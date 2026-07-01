# LayerNorm
$x \in \mathbb{R}^{S \times D}$, layer norm is applied per row (over the last dimension $D$):

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$
- $\mu = \frac{1}{D}\sum_{i=1}^{D} x_i$
- $\sigma^2 = \frac{1}{D}\sum_{i=1}^{D} (x_i - \mu)^2$
- $\gamma, \beta \in \mathbb{R}$
- $\epsilon$
