## RTX 4060

```
=== N=16777216 ===
  dot_product_f32_f32                                   0.559 ms
  dot_product_f32x4_f32                                 0.553 ms
  dot_product_f16_f32                                   0.370 ms
  dot_product_f16x2_f32                                 0.280 ms
  dot_product_f16x8_pack_f32                            0.280 ms
  torch_dot                                             0.550 ms

=== N=33554432 ===
  dot_product_f32_f32                                   1.095 ms
  dot_product_f32x4_f32                                 1.088 ms
  dot_product_f16_f32                                   0.743 ms
  dot_product_f16x2_f32                                 0.548 ms
  dot_product_f16x8_pack_f32                            0.547 ms
  torch_dot                                             1.098 ms

=== N=67108864 ===
  dot_product_f32_f32                                   2.180 ms
  dot_product_f32x4_f32                                 2.255 ms
  dot_product_f16_f32                                   1.265 ms
  dot_product_f16x2_f32                                 1.154 ms
  dot_product_f16x8_pack_f32                            1.097 ms
  torch_dot                                             2.224 ms

=== N=134217728 ===
  dot_product_f32_f32                                   4.526 ms
  dot_product_f32x4_f32                                 4.643 ms
  dot_product_f16_f32                                   2.420 ms
  dot_product_f16x2_f32                                 2.238 ms
  dot_product_f16x8_pack_f32                            2.381 ms
  torch_dot                                             4.425 ms

```
