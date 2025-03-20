# DQNO: Learning Statistically Accurate Derived Quanties of Hasegawa Wakatani Model
Derived-Quanities Neural Operator (DQNO) is a library for learning various neural operator methods to predict statistically accurate derived quanities of the Hasegawa Wakatani Model.

Unlike other neural operators these focus on predicting accurate derived quanties of the system instead of pointwise accurate state predictions

Some of these quanties are defined below:

$$
\begin{align}
    \Gamma^n &= -     \iint{ \mathrm{d}^2x \space \left( n \space \partial_y \phi \right) }  \\\
    \Gamma^c &= c_1   \iint{ \mathrm{d}^2x \space \left(n - \phi \right)^2}  \\\
    E        &= \small \frac{1}{2} \normalsize \iint{\mathrm{d}^2 x \space \left(n^2 - \left|\nabla_\bot \phi \right|^2 \right)}  \\\
    U        &= \small \frac{1}{2} \normalsize \iint{\mathrm{d}^2 x \space \left(n-\nabla_\bot^2  \phi\right)^2} = \small \frac{1}{2} \normalsize \iint{\mathrm{d}^2 x \space \left(n-\Omega\right)^2}
\end{align}
$$


## The Hasegawa-Wakatani Model

The HW model describes drift-wave turbulence using two physical fields: the density $n$ and the potential $\phi$ using various gradients on these.

$$
\begin{align}
    \partial_t n &= c_1 \left( \phi - n \right)
                     - \left[ \phi, n \right]
                     - \kappa_n \partial_y \phi
                     - \nu \nabla^{2N} n
    \\\
    \partial_t \Omega &= c_1 \left( \phi - n \right)
                                      - \left[ \phi, \Omega \right]
                                      - \nu \nabla^{2N} \Omega
    \\\
    \Omega &= \nabla^2 \phi
\end{align}
$$

$\Omega$ is also (more) commonly written as $\zeta$ (The voriticity)

$$
\begin{align}
    \partial_t n &= c_1 \left( \phi - n \right)
                     - \left[ \phi, n \right]
                     - \kappa_n \partial_y \phi
                     - \nu \nabla^{2N} n
    \\\
    \partial_t \zeta &= c_1 \left( \phi - n \right)
                                      - \left[ \phi, \zeta \right]
                                      - \nu \nabla^{2N} \zeta
    \\\
    \zeta &= \nabla^2 \phi
\end{align}
$$