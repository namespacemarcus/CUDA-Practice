## RTX 4060

```
=== M=N=K=1024 ===
  hgemm_gw_tiled                                        0.214 ms   10043.8 GFLOPS
  hgemm_gw_tiled_bcf                                    0.139 ms   15446.9 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                0.133 ms   16206.7 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         0.118 ms   18204.4 GFLOPS
  hgemm_cublas                                          0.118 ms   18252.0 GFLOPS

=== M=N=K=2048 ===
  hgemm_gw_tiled                                        1.277 ms   13450.8 GFLOPS
  hgemm_gw_tiled_bcf                                    1.073 ms   16011.8 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                1.031 ms   16663.9 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         1.011 ms   16999.9 GFLOPS
  hgemm_cublas                                          0.980 ms   17522.4 GFLOPS

=== M=N=K=4096 ===
  hgemm_gw_tiled                                        9.749 ms   14097.3 GFLOPS
  hgemm_gw_tiled_bcf                                    5.508 ms   24951.8 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                5.684 ms   24179.8 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         5.156 ms   26658.0 GFLOPS
  hgemm_cublas                                          4.766 ms   28836.1 GFLOPS

=== M=N=K=8192 ===
  hgemm_gw_tiled                                       51.561 ms   21324.7 GFLOPS
  hgemm_gw_tiled_bcf                                   44.498 ms   24709.4 GFLOPS
  hgemm_gw_tiled_bcf_dbf                               46.678 ms   23555.3 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                        41.396 ms   26560.7 GFLOPS
  hgemm_cublas                                         36.309 ms   30282.2 GFLOPS
```
