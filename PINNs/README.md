# Physics-Informed Neural Networks (PINN) for the 1D Heat Equation

This repository provides an implementation of a **Physics-Informed Neural Network (PINN)** to solve the one-dimensional heat equation with asymmetric Dirichlet boundary conditions. The model is optimized using second-order **L-BFGS optimization** and validated against a classical **Finite Difference Method (FDM)** numerical solver.

---

## Mathematical Formulation

### 1. Governing Equation
We consider the one-dimensional heat equation describing temperature distribution $u(t, x)$ over time $t$ and space $x$:

$$
u_t(t, x) = u_{xx}(t, x), \quad t \in [0, T_{max}], \quad x \in [0, L]
$$

where the domain boundaries from the implementation are defined by:
- $T_{max} = 60$
- $L = 1$ (Length of the bar)

### 2. Boundary and Initial Conditions
The system is subject to asymmetric Dirichlet boundary conditions (where the left end is held ice-cold and the right end is heated) and a completely cold initial state:

- **Left Spatial Boundary ($x = 0$):**
  $$
  u(t, 0) = 0, \quad \forall t \in (0, T_{max}]
  $$

- **Right Spatial Boundary ($x = L$):**
  $$
  u(t, L) = 1, \quad \forall t \in (0, T_{max}]
  $$

- **Temporal Initial Condition ($t = 0$):**
  $$
  u(x, 0) = u_0(x) = 0, \quad \forall x \in [0, L]
  $$

---

## Physics-Informed Neural Network Framework

We approximate the underlying continuous solution $u: [0, T_{max}] \times [0, L] \mapsto \mathbb{R}$ using a deep feedforward neural network with tunable parameters $\theta$:

$$
u_\theta(t, x) \approx u(t, x)
$$

### 1. Residual Definitions
To enforce the governing physical law and conditions, we derive the following physics residuals:

* **Interior Residual:** Enforces the heat PDE inside the spatio-temporal domain.
    $$
    r_{int, \theta}(t, x) := u_{\theta, t}(t, x) - u_{\theta, xx}(t, x), \quad \forall t \in [0, T_{max}], \quad x \in [0, L]
    $$

* **Spatial Boundary Residuals:** Enforces compliance at the physical boundaries.
    $$
    r_{sb, \theta}^{(0)}(t) := u_{\theta}(t, 0) - 0, \quad \forall t \in (0, T_{max}]
    $$
    $$
    r_{sb, \theta}^{(L)}(t) := u_{\theta}(t, L) - 1, \quad \forall t \in (0, T_{max}]
    $$

* **Temporal Boundary (Initial) Residual:** Enforces compliance with the initial state at $t = 0$.
    $$
    r_{tb, \theta}(x) := u_{\theta}(0, x) - 0, \quad \forall x \in [0, L]
    $$

### 2. Continuous Objective Functions
Ideally, the parameters $\theta$ minimize the continuous $L^2$ norm of these residuals across their respective domains:

$$
L_{int}(\theta) = \int_{[0, T_{max}] \times [0, L]} r_{int, \theta}^2(t, x) \, dt dx
$$

$$
L_{sb}(\theta) = \int_{[0, T_{max}]} \left( r_{sb, \theta}^{(0)}(t) \right)^2 \, dt + \int_{[0, T_{max}]} \left( r_{sb, \theta}^{(L)}(t) \right)^2 \, dt
$$

$$
L_{tb}(\theta) = \int_{[0, L]} r_{tb, \theta}^2(x) \, dx
$$

---

## Discretization and Quasi-Monte Carlo Sampling

The integral terms are approximated using statistical sampling. Rather than standard pseudo-random numbers, we utilize low-discrepancy **Sobol sequences** to ensure an optimal space-filling distribution across the training domains.

We define the following discrete training sets:
- **Interior Points:** $$
  S_{int} = \{y_n\}_{n=1}^{N_{int}}, \quad y_n = (t, x)_n \in [0, T_{max}] \times [0, L]
  $$
- **Spatial Boundary Points:** $$
  S_{sb, 0} = \{t_n\}_{n=1}^{N_{sb}}, \quad t_n \in [0, T_{max}]
  $$
  $$
  S_{sb, L} = \{t_n\}_{n=1}^{N_{sb}}, \quad t_n \in [0, T_{max}]
  $$
- **Temporal Boundary Points:** $$
  S_{tb} = \{x_n\}_{n=1}^{N_{tb}}, \quad x_n \in [0, L]
  $$

The default sizes chosen in the implementation are $N_{int} = 2000$, $N_{sb} = 1000$, and $N_{tb} = 400$.

### Discrete Mean Squared Error Losses
The empirical mean squared error (MSE) loss metrics mapped to the code are given by:

$$
L_{int}(\theta) = \frac{1}{N_{int}} \sum_{n=1}^{N_{int}} r_{int, \theta}^2(y_n)
$$

$$
L_{sb}(\theta) = \frac{1}{N_{sb}} \sum_{n=1}^{N_{sb}} \left( u_\theta(t_n, 0) - 0 \right)^2 + \frac{1}{N_{sb}} \sum_{n=1}^{N_{sb}} \left( u_\theta(t_n, L) - 1 \right)^2
$$

$$
L_{tb}(\theta) = \frac{1}{N_{tb}} \sum_{n=1}^{N_{tb}} \left( u_\theta(0, x_n) - 0 \right)^2
$$

---

## Optimization Problem

The model parameters are trained simultaneously by solving the joint multi-objective optimization problem:

$$
\theta^\ast = \arg\min_{\theta} \Big( L_{int}(\theta) + \lambda L_u(\theta) \Big)
$$

where the combined boundary compliance penalty $L_u(\theta)$ is:

$$
L_u(\theta) = L_{tb}(\theta) + L_{sb}(\theta)
$$

and $\lambda = 10$ serves as the regularization scaling weight used to balance the physics residual constraints with the physical boundary constraints.

---

## Code Implementation Details

* **Architecture:** Fully connected architecture structured as `2` $\rightarrow$ `64` $\rightarrow$ `128` $\rightarrow$ `64` $\rightarrow$ `1` utilizing `Tanh` activation functions to guarantee smooth higher-order spatial and temporal derivatives.
* **Optimizer:** `torch.optim.LBFGS` with a learning rate of `1.0`, configured with `strong_wolfe` line-search criteria for stable and fast quasi-Newton convergence.
* **Validation:** Cross-verified directly against a Forward-Euler **Finite Difference Method (FDM)** grid using a stability threshold of $\Delta t \le 0.4 \Delta x^2$ to measure quantitative $L^2$ relative difference metrics.