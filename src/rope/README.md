# RoPE

Rotary Position Embedding (RoPE)，对位置 $m$ 的向量在每个二维子空间内施加旋转，把绝对位置信息以旋转角的形式编码进 query/key。

输入 $x \in \mathbb{R}^{S \times D}$，$S$ 为序列长度，$D$ 为隐藏维度（$D$ 为偶数）。将每个 token 的 $D$ 维向量按**相邻元素**组成 $D/2$ 个二维对（interleaved 布局）：

$$
(x_{2p},\ x_{2p+1}), \quad p = 0, 1, \dots, D/2 - 1
$$

对位置 $m$ 的第 $p$ 个对，定义角频率（$\theta = 10000$）：

$$
\theta_p = \theta^{-2p/D}
$$

旋转角度 $\phi_{m,p} = m \cdot \theta_p$，对二维对施加旋转矩阵：

$$
\begin{pmatrix}
y_{2p} \\ y_{2p+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos\phi_{m,p} & -\sin\phi_{m,p} \\
\sin\phi_{m,p} & \cos\phi_{m,p}
\end{pmatrix}
\begin{pmatrix}
x_{2p} \\ x_{2p+1}
\end{pmatrix}
$$

即：

$$
y_{2p} = x_{2p}\cos\phi_{m,p} - x_{2p+1}\sin\phi_{m,p}
$$

$$
y_{2p+1} = x_{2p}\sin\phi_{m,p} + x_{2p+1}\cos\phi_{m,p}
$$

其中位置 $m$ 取 token 在序列中的下标（`token_idx`）。
