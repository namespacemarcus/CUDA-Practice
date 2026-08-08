## RTX 4060

```
=== shape=(2048, 4096) ===
  matrix_transpose_f32_loadcoal                         0.765 ms
  matrix_transpose_f32x4_loadcoal                       0.755 ms
  matrix_transpose_f32x4_loadcoal_smem                  0.403 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              0.419 ms
  torch_transpose                                       0.514 ms

=== shape=(4096, 4096) ===
  matrix_transpose_f32_loadcoal                         1.517 ms
  matrix_transpose_f32x4_loadcoal                       1.503 ms
  matrix_transpose_f32x4_loadcoal_smem                  0.817 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              0.853 ms
  torch_transpose                                       1.087 ms

=== shape=(4096, 8192) ===
  matrix_transpose_f32_loadcoal                         3.054 ms
  matrix_transpose_f32x4_loadcoal                       3.384 ms
  matrix_transpose_f32x4_loadcoal_smem                  1.517 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              1.564 ms
  torch_transpose                                       2.459 ms

=== shape=(8192, 4096) ===
  matrix_transpose_f32_loadcoal                         2.299 ms
  matrix_transpose_f32x4_loadcoal                       2.472 ms
  matrix_transpose_f32x4_loadcoal_smem                  1.512 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              1.616 ms
  torch_transpose                                       2.196 ms

=== shape=(8192, 8192) ===
  matrix_transpose_f32_loadcoal                         4.757 ms
  matrix_transpose_f32x4_loadcoal                       4.529 ms
  matrix_transpose_f32x4_loadcoal_smem                  3.062 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              3.194 ms
  torch_transpose                                       4.542 ms
```
