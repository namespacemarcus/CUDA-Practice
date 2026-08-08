## RTX 4060

```
=== M=131072, K=16 ===
  hgemv_k16_f16                                         0.025 ms     165.8 GFLOPS
  torch_hgemv                                           0.020 ms     209.0 GFLOPS

=== M=262144, K=16 ===
  hgemv_k16_f16                                         0.062 ms     136.3 GFLOPS
  torch_hgemv                                           0.037 ms     224.4 GFLOPS

=== M=262140, K=128 ===
  hgemv_k32_f16                                         0.276 ms     243.4 GFLOPS
  hgemv_k128_f16x4                                      0.277 ms     242.5 GFLOPS
  torch_hgemv                                           0.304 ms     221.0 GFLOPS

=== M=131072, K=512 ===
  hgemv_k32_f16                                         0.709 ms     189.4 GFLOPS
  hgemv_k128_f16x4                                      0.547 ms     245.2 GFLOPS
  torch_hgemv                                           0.554 ms     242.3 GFLOPS

=== M=65536, K=2048 ===
  hgemv_k32_f16                                         1.096 ms     244.8 GFLOPS
  hgemv_k128_f16x4                                      1.094 ms     245.5 GFLOPS
  torch_hgemv                                           1.095 ms     245.1 GFLOPS

```
