## RTX 4060

```
=== shape=(4096, 4096) (16777216 elems) ===
  sigmoid_f32                                           0.610 ms
  sigmoid_f32x4                                         0.624 ms
  sigmoid_f16                                           0.312 ms
  sigmoid_f16x2                                         0.311 ms
  sigmoid_f16x8                                         0.326 ms
  sigmoid_f16x8_pack                                    0.318 ms
  torch_sigmoid                                         0.631 ms

=== shape=(8192, 4096) (33554432 elems) ===
  sigmoid_f32                                           1.234 ms
  sigmoid_f32x4                                         1.327 ms
  sigmoid_f16                                           0.620 ms
  sigmoid_f16x2                                         0.625 ms
  sigmoid_f16x8                                         0.593 ms
  sigmoid_f16x8_pack                                    0.609 ms
  torch_sigmoid                                         1.231 ms

=== shape=(8192, 8192) (67108864 elems) ===
  sigmoid_f32                                           2.502 ms
  sigmoid_f32x4                                         2.555 ms
  sigmoid_f16                                           1.285 ms
  sigmoid_f16x2                                         1.224 ms
  sigmoid_f16x8                                         1.196 ms
  sigmoid_f16x8_pack                                    1.234 ms
  torch_sigmoid                                         2.579 ms

=== shape=(16384, 4096) (67108864 elems) ===
  sigmoid_f32                                           2.442 ms
  sigmoid_f32x4                                         2.516 ms
  sigmoid_f16                                           1.229 ms
  sigmoid_f16x2                                         1.273 ms
  sigmoid_f16x8                                         1.203 ms
  sigmoid_f16x8_pack                                    1.233 ms
  torch_sigmoid                                         2.485 ms

```
