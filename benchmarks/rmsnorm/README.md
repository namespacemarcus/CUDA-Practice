## RTX 4060

```
=== S=16384, D=1024 ===
  rms_norm_f32                                          0.669 ms
  rms_norm_f32x4                                        0.601 ms
  rms_norm_f16_f16                                      0.616 ms
  rms_norm_f16x2_f16                                    0.283 ms
  rms_norm_f16x8_f16                                    0.300 ms
  rms_norm_f16x8_pack_f16                               0.309 ms
  rms_norm_f16_f32                                      0.593 ms
  rms_norm_f16x8_f32                                    0.304 ms
  rms_norm_f16x8_pack_f32                               0.360 ms
  torch_rmsnorm                                         2.092 ms

=== S=65536, D=1024 ===
  rms_norm_f32                                          2.831 ms
  rms_norm_f32x4                                        2.436 ms
  rms_norm_f16_f16                                      2.668 ms
  rms_norm_f16x2_f16                                    1.164 ms
  rms_norm_f16x8_f16                                    1.213 ms
  rms_norm_f16x8_pack_f16                               1.218 ms
  rms_norm_f16_f32                                      2.362 ms
  rms_norm_f16x8_f32                                    1.233 ms
  rms_norm_f16x8_pack_f32                               1.224 ms
  torch_rmsnorm                                         8.466 ms

=== S=32768, D=512 ===
  rms_norm_f32                                          0.584 ms
  rms_norm_f32x4                                        0.614 ms
  rms_norm_f16_f16                                      0.389 ms
  rms_norm_f16x2_f16                                    0.291 ms
  rms_norm_f16x8_f16                                    0.294 ms
  rms_norm_f16x8_pack_f16                               0.293 ms
  rms_norm_f16_f32                                      0.460 ms
  rms_norm_f16x8_f32                                    0.309 ms
  rms_norm_f16x8_pack_f32                               0.362 ms
  torch_rmsnorm                                         2.093 ms

=== S=65536, D=256 ===
  rms_norm_f32                                          0.605 ms
  rms_norm_f32x4                                        0.603 ms
  rms_norm_f16_f16                                      0.360 ms
  rms_norm_f16x2_f16                                    0.325 ms
  rms_norm_f16x8_f16                                    0.309 ms
  rms_norm_f16x8_pack_f16                               0.310 ms
  rms_norm_f16_f32                                      0.351 ms
  rms_norm_f16x8_f32                                    0.308 ms
  rms_norm_f16x8_pack_f32                               0.309 ms
  torch_rmsnorm                                         2.345 ms

```
