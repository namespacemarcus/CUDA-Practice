>RTX 4060

```
=== M=N=K=128 ===
  sgemm_naive                                           0.011 ms     369.6 GFLOPS
  sgemm_tiling                                          0.028 ms     152.3 GFLOPS
  sgemm_at_tiling                                       0.027 ms     154.2 GFLOPS
  sgemm_at_tiling_bcf_swizzling                         0.021 ms     196.0 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore                  0.020 ms     212.2 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore_dbf              0.018 ms     232.7 GFLOPS
  sgemm_cublas                                          0.013 ms     322.5 GFLOPS
  sgemm_cublas_tf32                                     0.013 ms     323.5 GFLOPS

=== M=N=K=256 ===
  sgemm_naive                                           0.057 ms     587.5 GFLOPS
  sgemm_tiling                                          0.048 ms     695.7 GFLOPS
  sgemm_at_tiling                                       0.040 ms     829.6 GFLOPS
  sgemm_at_tiling_bcf_swizzling                         0.037 ms     910.2 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore                  0.035 ms     958.1 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore_dbf              0.030 ms    1111.3 GFLOPS
  sgemm_cublas                                          0.017 ms    1985.9 GFLOPS
  sgemm_cublas_tf32                                     0.014 ms    2432.9 GFLOPS

=== M=N=K=512 ===
  sgemm_naive                                           0.392 ms     685.0 GFLOPS
  sgemm_tiling                                          0.148 ms    1807.9 GFLOPS
  sgemm_at_tiling                                       0.077 ms    3481.3 GFLOPS
  sgemm_at_tiling_bcf_swizzling                         0.070 ms    3826.9 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore                  0.066 ms    4051.9 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore_dbf              0.061 ms    4398.4 GFLOPS
  sgemm_cublas                                          0.052 ms    5163.8 GFLOPS
  sgemm_cublas_tf32                                     0.036 ms    7426.2 GFLOPS

=== M=N=K=1024 ===
  sgemm_naive                                           3.569 ms     601.8 GFLOPS
  sgemm_tiling                                          0.521 ms    4120.1 GFLOPS
  sgemm_at_tiling                                       0.408 ms    5266.6 GFLOPS
  sgemm_at_tiling_bcf_swizzling                         0.450 ms    4768.4 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore                  0.370 ms    5799.6 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore_dbf              0.391 ms    5485.6 GFLOPS
  sgemm_cublas                                          0.415 ms    5173.0 GFLOPS
  sgemm_cublas_tf32                                     0.243 ms    8837.6 GFLOPS

=== M=N=K=2048 ===
  sgemm_naive                                          20.703 ms     829.8 GFLOPS
  sgemm_tiling                                          2.554 ms    6726.8 GFLOPS
  sgemm_at_tiling                                       1.972 ms    8713.4 GFLOPS
  sgemm_at_tiling_bcf_swizzling                         1.829 ms    9391.1 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore                  1.804 ms    9521.7 GFLOPS
  sgemm_at_tiling_bcf_swizzling_cstore_dbf              1.865 ms    9211.7 GFLOPS
  sgemm_cublas                                          1.960 ms    8764.2 GFLOPS
  sgemm_cublas_tf32                                     1.342 ms   12797.3 GFLOPS
```