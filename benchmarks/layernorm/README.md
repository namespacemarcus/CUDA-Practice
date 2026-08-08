## RTX 4060

```
=== S=16384, D=1024 ===
  layer_norm_f32                                        0.931 ms
  layer_norm_f32x4                                      0.617 ms
  layer_norm_f16_f16                                    0.884 ms
  layer_norm_f16x2_f16                                  0.317 ms
  layer_norm_f16x8_f16                                  0.299 ms
  layer_norm_f16x8_pack_f16                             0.309 ms
  layer_norm_f16_f32                                    0.853 ms
  layer_norm_f16x8_pack_f32                             0.308 ms
  torch_layernorm                                       0.607 ms

=== S=65536, D=1024 ===
  layer_norm_f32                                        3.563 ms
  layer_norm_f32x4                                      2.534 ms
  layer_norm_f16_f16                                    3.577 ms
  layer_norm_f16x2_f16                                  1.249 ms
  layer_norm_f16x8_f16                                  1.291 ms
  layer_norm_f16x8_pack_f16                             1.319 ms
  layer_norm_f16_f32                                    3.491 ms
  layer_norm_f16x8_pack_f32                             1.221 ms
  torch_layernorm                                       2.502 ms

=== S=32768, D=512 ===
  layer_norm_f32                                        0.616 ms
  layer_norm_f32x4                                      0.620 ms
  layer_norm_f16_f16                                    0.552 ms
  layer_norm_f16x2_f16                                  0.338 ms
  layer_norm_f16x8_f16                                  0.303 ms
  layer_norm_f16x8_pack_f16                             0.309 ms
  layer_norm_f16_f32                                    0.546 ms
  layer_norm_f16x8_pack_f32                             0.308 ms
  torch_layernorm                                       0.607 ms

=== S=65536, D=256 ===
  layer_norm_f32                                        0.592 ms
  layer_norm_f32x4                                      0.604 ms
  layer_norm_f16_f16                                    0.554 ms
  layer_norm_f16x2_f16                                  0.302 ms
  layer_norm_f16x8_f16                                  0.301 ms
  layer_norm_f16x8_pack_f16                             0.309 ms
  layer_norm_f16_f32                                    0.499 ms
  layer_norm_f16x8_pack_f32                             0.309 ms
  torch_layernorm                                       0.604 ms

```
