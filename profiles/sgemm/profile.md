![alt text](image.png)

在 $M=N=K=4096$ 的 shape 下，应用 tiled 分块 + aT + swizzling bcf + coal store + double buffer 后，性能超过 cuBLAS kernel。