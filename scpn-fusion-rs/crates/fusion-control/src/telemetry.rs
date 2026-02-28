// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Circular Telemetry Buffer
// © 1998–2026 Miroslav Šotek. All rights reserved.
// ─────────────────────────────────────────────────────────────────────
//! Zero-allocation circular buffer for high-frequency telemetry.
//! Prevents memory fragmentation in 10kHz+ control loops.

use serde::{Deserialize, Serialize};

/// A fixed-size circular buffer for a single telemetry channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularChannel {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl CircularChannel {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            capacity,
            head: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Returns the data in chronological order (oldest to newest).
    pub fn get_view(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.count);
        if self.count < self.capacity {
            result.extend_from_slice(&self.data[0..self.count]);
        } else {
            // Read from head to end, then from 0 to head
            result.extend_from_slice(&self.data[self.head..self.capacity]);
            result.extend_from_slice(&self.data[0..self.head]);
        }
        result
    }

    pub fn latest(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let idx = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };
        self.data[idx]
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }
}

/// Multi-channel telemetry suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySuite {
    pub r_axis: CircularChannel,
    pub z_axis: CircularChannel,
    pub ip_ma: CircularChannel,
    pub beta: CircularChannel,
}

impl TelemetrySuite {
    pub fn new(capacity: usize) -> Self {
        Self {
            r_axis: CircularChannel::new(capacity),
            z_axis: CircularChannel::new(capacity),
            ip_ma: CircularChannel::new(capacity),
            beta: CircularChannel::new(capacity),
        }
    }

    pub fn record(&mut self, r: f64, z: f64, ip: f64, beta: f64) {
        self.r_axis.push(r);
        self.z_axis.push(z);
        self.ip_ma.push(ip);
        self.beta.push(beta);
    }

    pub fn clear(&mut self) {
        self.r_axis.clear();
        self.z_axis.clear();
        self.ip_ma.clear();
        self.beta.clear();
    }
}
