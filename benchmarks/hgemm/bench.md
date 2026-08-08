## RTX 4060

```
=== M=N=K=1024 ===
  hgemm_gw_tiled                                        0.168 ms   12756.4 GFLOPS
  hgemm_gw_tiled_bcf                                    0.139 ms   15421.7 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                0.133 ms   16171.2 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         0.122 ms   17652.8 GFLOPS
  hgemm_cublas                                          0.114 ms   18755.4 GFLOPS
  torch_hgemm                                           0.119 ms   18045.4 GFLOPS

=== M=N=K=2048 ===
  hgemm_gw_tiled                                        1.230 ms   13963.6 GFLOPS
  hgemm_gw_tiled_bcf                                    0.894 ms   19215.7 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                1.037 ms   16563.5 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         0.809 ms   21234.3 GFLOPS
  hgemm_cublas                                          0.919 ms   18687.6 GFLOPS
  torch_hgemm                                           0.854 ms   20112.4 GFLOPS

=== M=N=K=4096 ===
  hgemm_gw_tiled                                        8.530 ms   16111.8 GFLOPS
  hgemm_gw_tiled_bcf                                    6.248 ms   21998.3 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                5.293 ms   25963.9 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         4.730 ms   29054.6 GFLOPS
  hgemm_cublas                                          4.550 ms   30203.4 GFLOPS
  torch_hgemm                                           4.602 ms   29864.7 GFLOPS

=== M=N=K=8192 ===
  hgemm_gw_tiled                                       52.465 ms   20957.1 GFLOPS
  hgemm_gw_tiled_bcf                                   39.300 ms   27977.3 GFLOPS
  hgemm_gw_tiled_bcf_dbf                               40.656 ms   27044.2 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                        36.988 ms   29726.0 GFLOPS
  hgemm_cublas                                         36.085 ms   30469.9 GFLOPS
  torch_hgemm                                          36.081 ms   30473.2 GFLOPS
```
