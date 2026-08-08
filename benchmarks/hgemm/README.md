## RTX 4060

```
=== M=N=K=1024 ===
  hgemm_tiled                                           0.164 ms   13117.7 GFLOPS
  hgemm_gw_tiled                                        0.169 ms   12673.3 GFLOPS
  hgemm_gw_tiled_bcf                                    0.139 ms   15501.1 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                0.133 ms   16194.2 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         0.118 ms   18252.0 GFLOPS
  hgemm_cublas                                          0.115 ms   18659.5 GFLOPS
  torch_hgemm                                           0.115 ms   18628.4 GFLOPS

=== M=N=K=2048 ===
  hgemm_tiled                                           1.310 ms   13118.5 GFLOPS
  hgemm_gw_tiled                                        1.203 ms   14277.3 GFLOPS
  hgemm_gw_tiled_bcf                                    0.877 ms   19590.6 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                1.054 ms   16304.4 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         0.810 ms   21218.4 GFLOPS
  hgemm_cublas                                          0.888 ms   19357.6 GFLOPS
  torch_hgemm                                           0.924 ms   18583.5 GFLOPS

=== M=N=K=4096 ===
  hgemm_tiled                                           8.173 ms   16815.3 GFLOPS
  hgemm_gw_tiled                                        7.364 ms   18664.8 GFLOPS
  hgemm_gw_tiled_bcf                                    5.070 ms   27105.9 GFLOPS
  hgemm_gw_tiled_bcf_dbf                                5.241 ms   26224.2 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                         4.676 ms   29393.1 GFLOPS
  hgemm_cublas                                          4.513 ms   30451.4 GFLOPS
  torch_hgemm                                           4.628 ms   29700.2 GFLOPS

=== M=N=K=8192 ===
  hgemm_tiled                                          51.045 ms   21540.1 GFLOPS
  hgemm_gw_tiled                                       52.503 ms   20941.9 GFLOPS
  hgemm_gw_tiled_bcf                                   46.235 ms   23780.8 GFLOPS
  hgemm_gw_tiled_bcf_dbf                               45.969 ms   23918.4 GFLOPS
  hgemm_gw_tiled_bcf_dbf_cstore                        37.011 ms   29707.7 GFLOPS
  hgemm_cublas                                         36.071 ms   30481.6 GFLOPS
  torch_hgemm                                          36.084 ms   30470.5 GFLOPS
```
