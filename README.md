# DQFNO
Derived-Quanities Fourier Neural Operator


### Important Things
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