# RMSNorm
$x \in \mathbb{R}^{S \times D}$, rms norm is applied per row (over the last dimension $D$):

$$
y = \frac{x}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}} \cdot \gamma
$$
- $\gamma \in \mathbb{R}$
- $\epsilon = 10^{-5}$
