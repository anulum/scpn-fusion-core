//! GPU-accelerated Grad-Shafranov solver via wgpu compute shaders.
//!
//! Provides Red-Black SOR on the GPU for the GS equation inner loop.
//! Falls back to CPU if no GPU adapter is available.

use bytemuck::{Pod, Zeroable};
use fusion_types::error::{FusionError, FusionResult};
use std::borrow::Cow;

/// Uniform parameters passed to the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    nr: u32,
    nz: u32,
    dr: f32,
    dz: f32,
    r_left: f32,
    omega: f32,
    color: u32,
    _pad: u32,
}

/// GPU-accelerated GS solver using wgpu compute shaders.
pub struct GpuGsSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pipeline_residual: wgpu::ComputePipeline,
    pipeline_restrict: wgpu::ComputePipeline,
    pipeline_prolong: wgpu::ComputePipeline,
    param_buffer: wgpu::Buffer,
    psi_buffer: wgpu::Buffer,
    source_buffer: wgpu::Buffer,
    _residual_buffer: wgpu::Buffer,
    coarse_psi_buffer: wgpu::Buffer,
    coarse_source_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_mg: wgpu::BindGroup,
    nr: usize,
    nz: usize,
    dr: f32,
    dz: f32,
    r_left: f32,
}

impl GpuGsSolver {
    /// Create a new GPU GS solver for the given grid.
    ///
    /// Returns `Err` if no suitable GPU adapter is found.
    pub fn new(
        nr: usize,
        nz: usize,
        r_left: f64,
        r_right: f64,
        z_bottom: f64,
        z_top: f64,
    ) -> FusionResult<Self> {
        if nr < 4 || nz < 4 {
            return Err(FusionError::ConfigError(
                "GPU GS solver requires nr >= 4 and nz >= 4".to_string(),
            ));
        }

        let dr = (r_right - r_left) / (nr as f64 - 1.0);
        let dz = (z_top - z_bottom) / (nz as f64 - 1.0);

        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| FusionError::ConfigError("No suitable GPU adapter found".to_string()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fusion-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| FusionError::ConfigError(format!("GPU device request failed: {e}")))?;

        // Load WGSL shader
        let shader_source = include_str!("gs_solver.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gs_solver"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        let grid_size = nr * nz;
        let buf_size = (grid_size * std::mem::size_of::<f32>()) as u64;

        // Create buffers
        let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let psi_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psi"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let source_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("source"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout and pipeline
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gs_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Residual buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Coarse source (for restriction)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Coarse destination (for restriction/prolongation)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gs_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gs_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_residual = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gs_residual_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("calculate_residual"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_restrict = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gs_restrict_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("restrict_to_coarse"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_prolong = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gs_prolong_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("prolong_and_add"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Residual and Coarse buffers
        let nr_c = (nr - 1) / 2 + 1;
        let nz_c = (nz - 1) / 2 + 1;
        let coarse_grid_size = nr_c * nz_c;
        let coarse_buf_size = (coarse_grid_size * std::mem::size_of::<f32>()) as u64;

        let residual_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residual"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let coarse_psi_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coarse_psi"),
            size: coarse_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let coarse_source_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coarse_source"),
            size: coarse_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gs_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: psi_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: residual_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: coarse_source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: coarse_psi_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_mg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gs_mg_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coarse_psi_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: coarse_source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: residual_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: psi_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_residual,
            pipeline_restrict,
            pipeline_prolong,
            param_buffer,
            psi_buffer,
            source_buffer,
            _residual_buffer: residual_buffer,
            coarse_psi_buffer,
            coarse_source_buffer,
            staging_buffer,
            bind_group,
            bind_group_mg,
            nr,
            nz,
            dr: dr as f32,
            dz: dz as f32,
            r_left: r_left as f32,
        })
    }

    /// Upload initial ψ and source arrays (both row-major, nz×nr).
    pub fn upload(&self, psi: &[f32], source: &[f32]) -> FusionResult<()> {
        let expected = self.nr * self.nz;
        if psi.len() != expected || source.len() != expected {
            return Err(FusionError::ConfigError(format!(
                "upload: expected {} elements, got psi={} source={}",
                expected,
                psi.len(),
                source.len()
            )));
        }
        self.queue
            .write_buffer(&self.psi_buffer, 0, bytemuck::cast_slice(psi));
        self.queue
            .write_buffer(&self.source_buffer, 0, bytemuck::cast_slice(source));
        Ok(())
    }

    /// Run `iterations` Red-Black SOR sweeps on the GPU.
    pub fn solve(&self, iterations: usize, omega: f32) -> FusionResult<()> {
        if !(1.0..2.0).contains(&omega) {
            return Err(FusionError::ConfigError(format!(
                "SOR omega must be in [1, 2), got {omega}"
            )));
        }

        // Workgroup dimensions: 16×16 threads per workgroup
        let wg_x = ((self.nr - 2) as u32).div_ceil(16);
        let wg_y = ((self.nz - 2) as u32).div_ceil(16);

        for _ in 0..iterations {
            for color in 0..2u32 {
                let params = GpuParams {
                    nr: self.nr as u32,
                    nz: self.nz as u32,
                    dr: self.dr,
                    dz: self.dz,
                    r_left: self.r_left,
                    omega,
                    color,
                    _pad: 0,
                };
                self.queue
                    .write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&params));

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("gs_encoder"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gs_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
        }

        // Wait for GPU to finish
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Download the solved ψ array from the GPU.
    pub fn download(&self) -> FusionResult<Vec<f32>> {
        let buf_size = (self.nr * self.nz * std::mem::size_of::<f32>()) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.psi_buffer, 0, &self.staging_buffer, 0, buf_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| FusionError::ConfigError(format!("GPU download channel error: {e}")))?
            .map_err(|e| FusionError::ConfigError(format!("GPU buffer map failed: {e}")))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }

    /// Convenience: upload, solve, download in one call.
    pub fn solve_full(
        &self,
        psi_init: &[f32],
        source: &[f32],
        iterations: usize,
        omega: f32,
    ) -> FusionResult<Vec<f32>> {
        self.upload(psi_init, source)?;
        self.solve(iterations, omega)?;
        self.download()
    }

    /// Run a 2-level V-cycle on the GPU.
    pub fn vcycle(&self, pre_sweeps: usize, post_sweeps: usize, omega: f32) -> FusionResult<()> {
        let nr_c = (self.nr - 1) / 2 + 1;
        let nz_c = (self.nz - 1) / 2 + 1;

        let wg_x_f = ((self.nr - 2) as u32).div_ceil(16);
        let wg_y_f = ((self.nz - 2) as u32).div_ceil(16);
        let wg_x_c = (nr_c as u32).div_ceil(16);
        let wg_y_c = (nz_c as u32).div_ceil(16);

        // 1. Pre-smoothing (Fine grid)
        self.solve(pre_sweeps, omega)?;

        // 2. Residual calculation (Fine grid)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mg_residual"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_residual);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x_f, wg_y_f, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // 3. Restriction (Fine -> Coarse)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mg_restrict"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_restrict);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x_c, wg_y_c, 1);
        }
        // Copy restricted residual to coarse source buffer
        // Note: The restrict kernel writes to coarse_psi_buffer (binding 5) which is coarse_dest
        // But for the coarse solve, it needs to be in coarse_source_buffer
        // Wait, the restrict kernel writes to binding 5 (coarse_dest).
        // We need to copy binding 5 -> coarse_source_buffer.
        self.queue.submit(Some(encoder.finish()));

        let coarse_grid_size = nr_c * nz_c;
        let coarse_buf_size = (coarse_grid_size * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mg_copy"),
            });
        encoder.copy_buffer_to_buffer(
            &self.coarse_psi_buffer,
            0,
            &self.coarse_source_buffer,
            0,
            coarse_buf_size,
        );
        self.queue.submit(Some(encoder.finish()));

        // Clear coarse psi for the error solve
        let zeros = vec![0.0f32; coarse_grid_size];
        self.queue
            .write_buffer(&self.coarse_psi_buffer, 0, bytemuck::cast_slice(&zeros));

        // 4. Coarse solve (using bind_group_mg which points to coarse buffers)
        // We use a few more sweeps on the coarse grid
        for _ in 0..10 {
            for color in 0..2u32 {
                let params = GpuParams {
                    nr: nr_c as u32,
                    nz: nz_c as u32,
                    dr: self.dr * 2.0,
                    dz: self.dz * 2.0,
                    r_left: self.r_left,
                    omega: 1.0, // usually 1.0 for coarse solve
                    color,
                    _pad: 0,
                };
                self.queue
                    .write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&params));
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.bind_group_mg, &[]);
                    pass.dispatch_workgroups(wg_x_c, wg_y_c, 1);
                }
                self.queue.submit(Some(encoder.finish()));
            }
        }

        // 5. Prolongation and Addition (Coarse -> Fine)
        // Restore fine params
        let fine_params = GpuParams {
            nr: self.nr as u32,
            nz: self.nz as u32,
            dr: self.dr,
            dz: self.dz,
            r_left: self.r_left,
            omega,
            color: 0,
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&fine_params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mg_prolong"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_prolong);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x_f, wg_y_f, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // 6. Post-smoothing (Fine grid)
        self.solve(post_sweeps, omega)?;

        Ok(())
    }

    /// Grid dimensions.
    pub fn grid_shape(&self) -> (usize, usize) {
        (self.nz, self.nr)
    }
}

/// Check if a GPU adapter is available without creating a full solver.
pub fn gpu_available() -> bool {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_some()
}

/// Get GPU adapter info string.
pub fn gpu_info() -> Option<String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let info = adapter.get_info();
    Some(format!(
        "{} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_params_size() {
        assert_eq!(std::mem::size_of::<GpuParams>(), 32);
    }

    #[test]
    fn test_gpu_available_does_not_panic() {
        // Just check it doesn't crash — may return false in CI
        let _ = gpu_available();
    }

    #[test]
    fn test_gpu_info_does_not_panic() {
        let _ = gpu_info();
    }

    #[test]
    fn test_solver_rejects_small_grid() {
        let result = GpuGsSolver::new(2, 2, 1.0, 3.0, -1.0, 1.0);
        assert!(result.is_err());
    }

    // Integration test: only runs if GPU is available
    #[test]
    fn test_gpu_solve_zero_source_preserves_boundary() {
        if !gpu_available() {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }

        let nr = 33;
        let nz = 33;
        let solver = GpuGsSolver::new(nr, nz, 1.0, 3.0, -1.0, 1.0)
            .expect("GPU solver creation should succeed");

        // Zero source, zero initial → should stay zero
        let psi = vec![0.0f32; nr * nz];
        let source = vec![0.0f32; nr * nz];

        let result = solver
            .solve_full(&psi, &source, 10, 1.3)
            .expect("GPU solve should succeed");

        assert_eq!(result.len(), nr * nz);
        let max_abs = result.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "Zero source should give zero solution, got max_abs={}",
            max_abs
        );
    }
}
