# Strained 1D rod 
This was the first example we got started with as an introduction to PINNS, based on a simple exercise from TU Braunschweig. 

## Forward problem
The original task was to solve the following PDE with a Physics Informed Neural Network:
$$
\sigma (x) = E\cdot \frac{\partial u(x)}{\partial x}, \quad \frac{\partial \sigma (x)}{\partial x} = 0
$$
with boundary conditions:
$$
u(0) = 0,\quad \sigma (1) = \sigma_0
$$
which corresponds to a homogenous rod, which is fixed on the left end and subjected to an axial force $F=\sigma_0\cdot A$ at the right end. A simple PINN is perfectly capable of approximating the solution to this problem.



## Inverse problem
We are aiming to eventually utilize PINNs for system identification. This generally involves solving an inverse problem. To mimic this situation we altered the above problem. The Youngs modulus $E$, which is a prescribed constant in the former case, is now added as a variable to the output of the neural network. The system of PDEs can for this case be summarized with the equation
$$
\frac{\partial E(x)}{\partial x}\frac{\partial u(x)}{\partial x} + E(x)\frac{\partial^2u(x)}{\partial x^2} = 0
$$
The goal is to prescribe a certain $E$ profile and be able to identify this profile with the PINN based on artifical data on the displacement $u$ (drawn from the corresponding analytic solution), which adds a contribution to the loss function 
$$
\mathcal{L} = \mathcal{L}_\text{PDE} + \mathcal{L}_\text{BC} + \mathcal{L}_\text{data}
$$
For the case of a homogenous beam, the PINN is able to identify the prescribed value for $E$ with an almost constant profile. 
As a next step we tried a linearly increasing profile for $E$ with the same boundary conditions, which gives the solutions
$$
E(x) = 10\cdot(x+1)\\
\rightarrow u(x) = \frac{1}{2}\ln(x+1)
$$
With some tweeking of the parameters and sufficient up-scaling of $\mathcal{L}_\text{data}$, the PINN can identify an almost linear profile. However, the loss minimization always stagnates at a certain value. This might be an issue of the algorithm ending up in a dominating local minimum. Possible solution ansatzes are incorporation of BFGS optimizer instead of Adam and maybe going from tanh to swish activation.