# Euler beam 

To see if other physical problems would lead to the same problems we encountered with the axial beam, we followed the work of [Kapoor et al. 2023](https://arxiv.org/abs/2303.01055) who solved the time dependent 1D Euler beam equation with a PINN.
For our purpose we only deal with the time independent part of the PDE, which transverse beam deflection $u$ with the line load $f$
$$EIu_{xxxx}(x,t)=f(x,t)$$
and we set the boundary conditions
$$u(0)=u(1)=u_{xx}(0)=u_{xx}(1)=0$$
and the force profile with corresponding solution:
$$f(x) = EI\pi^4\sin(\pi x)$$
$$u(x) = \sin(\pi x)$$
## Forward problem

For the forward problem, the PINN is again able to fit the solution for $u(x)$ arbitrarily well given $f(x)$.

## Inverse problem
For the inverse problem, the NN is again able to fit the data on $u$ in case of appropriate loss scaling. However, it fails to remotely approximate the sinusoidal shape of the force profile. Instead it always ends up with an almost uniform line force, which seems to have a similar resulting force along the whole beam. 
Looking at the individual loss contributions reveals, that $\mathcal{L}_\text{PDE}$ is stagnating in the minimization. We suspect, that the 4th order derivative in the PDE complicates the minimization, since small changes in f can induce large changes for u. 

