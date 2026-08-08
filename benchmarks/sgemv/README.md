## RTX 4060

```
=== M=131072, K=16 ===
  sgemv_k16_f32                                         0.026 ms     162.0 GFLOPS
  torch_sgemv                                           0.032 ms     130.4 GFLOPS

=== M=262144, K=16 ===
  sgemv_k16_f32                                         0.048 ms     173.9 GFLOPS
  torch_sgemv                                           0.038 ms     218.1 GFLOPS

=== M=262140, K=128 ===
  sgemv_k32_f32                                         0.554 ms     121.2 GFLOPS
  sgemv_k128_f32x4                                      0.557 ms     120.4 GFLOPS
  torch_sgemv                                           0.552 ms     121.5 GFLOPS

=== M=131072, K=512 ===
  sgemv_k32_f32                                         1.158 ms     115.9 GFLOPS
  sgemv_k128_f32x4                                      1.095 ms     122.5 GFLOPS
  torch_sgemv                                           1.247 ms     107.6 GFLOPS

=== M=65536, K=2048 ===
  sgemv_k32_f32                                         2.181 ms     123.1 GFLOPS
  sgemv_k128_f32x4                                      2.264 ms     118.6 GFLOPS
  torch_sgemv                                           2.183 ms     123.0 GFLOPS

```
