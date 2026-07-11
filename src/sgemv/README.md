# SGEMV

$$
y = A \cdot x
$$
- 矩阵 $A \in \mathbb{R}^{M \times K}$（行主序）
- 向量 $x \in \mathbb{R}^{K \times 1}$
- 输出向量 $y \in \mathbb{R}^{M \times 1}$
$$
y_m = \sum_{k=0}^{K-1} A_{m,k} \cdot x_k, \quad m \in [0, M)
$$
本质就是 **$M$ 次长度为 $K$ 的点积**，每次点积产生 $y$ 的一个元素。
