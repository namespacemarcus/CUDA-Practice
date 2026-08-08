## RTX 4060

```
=== shape=(4096, 4096) (16777216 elems) ===
  elementwise_add_f32                                   0.923 ms
  elementwise_add_f32x4                                 0.911 ms
  elementwise_add_f16                                   0.432 ms
  elementwise_add_f16x2                                 0.454 ms
  elementwise_add_f16x8                                 0.511 ms
  elementwise_add_f16x8_pack                            0.527 ms
  torch_add                                             1.028 ms

=== shape=(8192, 4096) (33554432 elems) ===
  elementwise_add_f32                                   2.102 ms
  elementwise_add_f32x4                                 2.218 ms
  elementwise_add_f16                                   1.006 ms
  elementwise_add_f16x2                                 1.109 ms
  elementwise_add_f16x8                                 1.010 ms
  elementwise_add_f16x8_pack                            1.048 ms
  torch_add                                             2.082 ms

=== shape=(8192, 8192) (67108864 elems) ===
  elementwise_add_f32                                   4.273 ms
  elementwise_add_f32x4                                 4.144 ms
  elementwise_add_f16                                   2.010 ms
  elementwise_add_f16x2                                 2.145 ms
  elementwise_add_f16x8                                 1.931 ms
  elementwise_add_f16x8_pack                            1.958 ms
  torch_add                                             3.734 ms

=== shape=(16384, 4096) (67108864 elems) ===
  elementwise_add_f32                                   3.752 ms
  elementwise_add_f32x4                                 3.694 ms
  elementwise_add_f16                                   1.720 ms
  elementwise_add_f16x2                                 1.824 ms
  elementwise_add_f16x8                                 1.730 ms
  elementwise_add_f16x8_pack                            1.831 ms
  torch_add                                             3.768 ms

```
