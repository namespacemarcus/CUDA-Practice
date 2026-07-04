# Softmax

## Softmax

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

直接计算会遇到数值溢出问题：当 $x_i$ 较大时 $e^{x_i}$ 会溢出到 `+INF`。

## Safe Softmax

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}, \quad m = \max_k x_k
$$

所有指数减去全局最大值，保证指数项 $\le 1$，避免上溢。需要两遍扫描：第一遍求 `max`，第二遍求 `exp` 并累加 `sum`。

## Online Softmax

Safe softmax 需要两遍全局同步，online softmax 将求 `max` 和求 `sum` 合并为一遍归约。

定义 `MD` 结构体存储局部统计量：

```cpp
struct MD { float m; float d; };
// m = 当前局部最大值
// d = Σ exp(x_j - m)，基于当前局部 m 的指数和
```

**合并（Merge）更新公式**：当合并两组局部统计量 $(m_a, d_a)$ 和 $(m_b, d_b)$，假设 $m_a > m_b$：

$$
m_{\text{new}} = m_a
$$

$$
d_{\text{new}} = d_a + d_b \cdot e^{m_b - m_a}
$$

**解释**：$d_b$ 原本以 $m_b$ 为基准计算（$d_b = \sum e^{x_j - m_b}$）。合并后全局最大值变为 $m_a$，需要将 $d_b$ 中的每一项重新放缩：

$$
e^{x_j - m_a} = e^{x_j - m_b} \cdot e^{m_b - m_a}
$$

因此 $d_b$ 整体乘以 $e^{m_b - m_a}$ 即可。

**算法流程**（以 `online_safe_softmax_f32_per_token_kernel` 为例）：

1. 每个线程初始化自己的 `MD(m_i, d_i)`，其中 $m_i = x_i$，$d_i = 1$（因为 $\exp(x_i - x_i) = 1$）
2. Warp 内归约：调用 `warp_reduce_md_op`，每次用上述合并公式两两合并
3. 各 warp 结果写入 shared memory，再由 warp 0 归约得到最终的 `MD(m_final, d_final)`
4. 输出：$y_i = \exp(x_i - m_{\text{final}}) / d_{\text{final}}$

只需一次 global memory 读取，所有归约在 register 和 shared memory 中完成。
