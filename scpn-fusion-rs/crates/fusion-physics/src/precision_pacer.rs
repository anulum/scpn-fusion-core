// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Precision Pacer (OS Scheduler Bypass)

//! High-precision timing pacer for real-time control loops.
//! 
//! Uses std::hint::spin_loop() to bypass the OS Completely Fair Scheduler (CFS)
//! and standard nanosleep jitter (typically 100-200 us).

use std::hint::spin_loop;
use std::time::{Duration, Instant};

/// Pacing mode for the real-time loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingMode {
    /// Yield to OS scheduler (standard nanosleep).
    Sleep,
    /// Refuse to yield core, burning cycles for microsecond precision.
    Spin,
}

/// Precision pacer state.
#[derive(Debug)]
pub struct PrecisionPacer {
    last_tick: Instant,
    target_interval: Duration,
    mode: PacingMode,
}

impl PrecisionPacer {
    pub fn new(frequency_hz: f64, mode: PacingMode) -> Self {
        Self {
            last_tick: Instant::now(),
            target_interval: Duration::from_nanos((1.0e9 / frequency_hz) as u64),
            mode,
        }
    }

    /// Wait until the next tick interval.
    pub fn wait_next(&mut self) -> u128 {
        let elapsed = self.last_tick.elapsed();
        if elapsed < self.target_interval {
            match self.mode {
                PacingMode::Sleep => {
                    std::thread::sleep(self.target_interval - elapsed);
                }
                PacingMode::Spin => {
                    let target_ns = self.target_interval.as_nanos();
                    while self.last_tick.elapsed().as_nanos() < target_ns {
                        spin_loop();
                    }
                }
            }
        }
        
        let actual_elapsed = self.last_tick.elapsed().as_nanos();
        self.last_tick = Instant::now();
        actual_elapsed
    }

    pub fn reset(&mut self) {
        self.last_tick = Instant::now();
    }
}
