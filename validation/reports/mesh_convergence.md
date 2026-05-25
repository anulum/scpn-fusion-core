# GS Solver Mesh Convergence Study

Determines the spatial order of accuracy for the Grad-Shafranov elliptic solver.

## Contract

- Contract: fixed-boundary manufactured Solov'ev GS solve shows second-order mesh convergence
- Status: PASS
- Required minimum adjacent-grid rate: 1.80
- Minimum observed adjacent-grid rate: 1.992859
- Rated grid transitions: 3

## Machine

- Platform: Linux-6.17.0-29-generic-x86_64-with-glibc2.39
- CPU count: 12
- Python: 3.12.3
- NumPy: 2.2.6

## Results

| Grid | h | NRMSE | Rate | Time (s) | Iters |
|------|---|-------|------|----------|-------|
| 17x17 | 1.2500e-01 | 8.1756e-05 | N/A | 0.0890 | 801 |
| 33x33 | 6.2500e-02 | 1.9818e-05 | 2.04 | 0.6779 | 2601 |
| 65x65 | 3.1250e-02 | 4.8780e-06 | 2.02 | 4.6114 | 10001 |
| 129x129 | 1.5625e-02 | 1.2256e-06 | 1.99 | 23.8719 | 25000 |
