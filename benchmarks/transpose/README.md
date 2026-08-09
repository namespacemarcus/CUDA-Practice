## RTX 4060

```
=== shape=(2048, 4096) ===
  matrix_transpose_f32_loadcoal                         0.762 ms
  matrix_transpose_f32x4_loadcoal                       0.758 ms
  matrix_transpose_f32x4_loadcoal_smem                  0.078 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              0.061 ms
  torch_transpose                                       0.569 ms

=== shape=(4096, 4096) ===
  matrix_transpose_f32_loadcoal                         1.738 ms
  matrix_transpose_f32x4_loadcoal                       1.732 ms
  matrix_transpose_f32x4_loadcoal_smem                  0.152 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              0.143 ms
  torch_transpose                                       1.212 ms

=== shape=(4096, 8192) ===
  matrix_transpose_f32_loadcoal                         2.419 ms
  matrix_transpose_f32x4_loadcoal                       2.465 ms
  matrix_transpose_f32x4_loadcoal_smem                  0.235 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              0.182 ms
  torch_transpose                                       2.364 ms

=== shape=(8192, 4096) ===
  matrix_transpose_f32_loadcoal                         2.453 ms
  matrix_transpose_f32x4_loadcoal                       2.410 ms
  matrix_transpose_f32x4_loadcoal_smem                  0.232 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              0.180 ms
  torch_transpose                                       2.394 ms

=== shape=(8192, 8192) ===
  matrix_transpose_f32_loadcoal                         4.657 ms
  matrix_transpose_f32x4_loadcoal                       4.477 ms
  matrix_transpose_f32x4_loadcoal_smem                  0.449 ms
  matrix_transpose_f32x4_loadcoal_smem_bcf              0.402 ms
  torch_transpose                                       4.420 ms
```
