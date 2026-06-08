# 1D Allen-Cahn Equation Solver using Neural Operators (FNO & CNO)

An implementation of Data-Driven Solution Operators for the 1D Allen-Cahn Partial Differential Equation (PDE) utilizing both Fourier Neural Operator (FNO) and Convolutional Neural Operator (CNO) frameworks in PyTorch.

---

## 1. Overview
Traditional numerical PDE solvers (e.g., Finite Difference, Finite Element Methods) require discrete, computationally expensive time-stepping loops to evolve a physical system from time $t=0$ to $t=1$. 

This project implements **Operator Learning** to map infinite-dimensional function spaces. Instead of learning a single solution, the neural operator learns the underlying physical law as a continuous operator, mapping any initial boundary profile directly to its future state in a single neural network forward pass.

$$\mathcal{G}: u(\cdot, t = 0) \mapsto  u(\cdot, t = 1)$$

### Key Features
- **Mesh/Resolution Independence:** Because the learning occurs globally within the frequency domain using continuous wave representations, a network trained on a coarse grid (e.g., $N=1001$) can generalize seamlessly to perform zero-shot super-resolution inference on significantly finer grids without retraining.
- **Fast Inference:** Bypasses sequential step-by-step temporal integration, yielding massive speedups over classical solvers.

---

## 2. Mathematical Background: The Allen-Cahn Equation
The Allen-Cahn equation is a well-known non-linear partial differential equation describing the thermodynamics of **phase separation** in multi-component alloy systems and diffuse interface phase-fields (e.g., crystalline grain growth, immiscible fluid separation):

$$\frac{\partial u}{\partial t} = \Delta u - \epsilon^2 u (u^2 - 1)$$

### Components of the equation:
* **$u(x, t)$ (Phase Field / Order Parameter):** A continuous variable where $u = 1$ denotes stable Phase A, $u = -1$ denotes stable Phase B, and intermediate values around $u = 0$ capture the thin, diffuse interface between them.
* **$\Delta u$ (Diffusion/Laplacian Term):** Drives global spatial smoothing. It acts to minimize surface gradients and interface boundary area.
* **$-\epsilon^2 u (u^2 - 1)$ (Reaction Term):** A non-linear point-wise driving force derived from the derivative of a double-well potential $F(u) = \frac{1}{4}(u^2 - 1)^2$ with energy minima exactly at $u = \pm 1$. This term forces intermediate state values to "choose a side" and rapidly transition into pure phases.
* **$\epsilon$ (Interface Width Parameter):** Controls the balance between diffusion and reaction. Smaller $\epsilon$ values produce ultra-sharp localized phase boundaries.

---

## 3. Neural Operator Architectures

### Section 1: Fourier Neural Operator (FNO)
The project utilizes a 1D FNO pipeline to approximate the solution operator $\mathcal{G}$. The model processes data through three key structural steps:

1. **Lifting Layer ($P$):** Pointwise maps the 2-channel physical spatial inputs (the initial condition $u(x, t=0)$ and grid coordinates $x$) up to a higher-dimensional latent space ($width=64$) using MLP.
2. **Fourier Layers ($\mathcal{K} + W$):** The core operator consisting of parallel tracks:
   * **Spectral Path ($\mathcal{K}$):** Executes a Real Fast Fourier Transform (`torch.fft.rfft`), truncates high-frequency features beyond a low-pass filter threshold (`n_modes=16`), performs complex feature mixing weights via Einstein summation (`torch.einsum`), and returns via an Inverse FFT. This models the continuous global diffusion physics.
   * **Physical Path ($W$):** A 1D Convolution with a kernel size of 1 acting point-by-point to safely preserve force term from equation.
3. **Projection Layer ($Q$):** Compresses the hidden width channels back down to a single physical 1-channel solution representing $\hat{u}(x, t=1)$.

---

### Section 2: Convolutional Neural Operator (CNO)
Formulated as an **iterative architecture** introduced in the paper *[Convolutional Neural Operators for robust and accurate learning of PDEs](https://arxiv.org/abs/2302.01178)*, the CNO presents an option optimized for the physical domain.

It utilizes a multi-scale learning framework designed to mitigate high-frequency mapping discrepancies natively in the physical space.

#### What is Aliasing and Why Does it Matter?
* **Frequency Leakage:** Every fundamental non-linear operation (like an activation function) creates additional high-frequency components that lead to **aliasing errors**.
* **Error Propagation:** These errors accumulate due to inconsistent frequency content and propagate through the network layers, degrading training stability and numerical precision.

#### How do the CNO Operations Work?
To alleviate these errors, CNO redefines core layers to maintain proper band-limiting features:
* **Up/Downsampling:** Utilizes strict low-pass filtering alongside spatial resize modes to permanently discard aliased signal noise.
* **Activations (`CON_ReLu`):** Employs a 3-step procedure to mitigate aliasing during non-linear operations:
  1. **Upsample:** Interpolates the input feature maps to a higher frequency resolution grid (e.g., double the input frequency sizes).
  2. **Activate:** Applies the standard activation function (e.g., `ReLU`) on the expanded space where high-frequency errors won't fold back into the signal.
  3. **Downsample:** Compresses the maps back down to the intended target output size with filtering.

#### The CNO Model Pipeline
The layout implements an alias-free multi-scale blueprint to scale across the following stages:
* **Lifting Layer:** The first layer maps input shapes into an initial latent continuous feature space by expanding the channel dimensions (`in_channels --> 8`).
* **Multi-Scale Encoder-Decoder Track:** Features a U-Net-like contracting and expanding network layout.
* **Residual Connections & Skip Paths:** Integrates additional ResNet blocks (`residual`) to directly connect matching spatial coordinates across matching hierarchical levels of the **encoder** and the **decoder**, heavily enhancing gradient training behavior.
* **Projection Layer:** The final output layer maps the internal multi-scale representations back down to the target single-channel physics space shape.

---

## 4. Dataset Structure
The data arrays are configured as follows:
* **`AC_data_input.npy`** (Shape: `(1000, 1001, 2)`): Consists of 1,000 independent physical simulations. Each sample has 1,001 grid mesh nodes containing coordinate values $x$ and corresponding initial mixtures $u(x, t=0)$.
* **`AC_data_output.npy`** (Shape: `(1000, 1001)`): Holds the target ground truth $u(x, t=1)$.
