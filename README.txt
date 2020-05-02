# README
# and-python project
# Version 1.0
# Author: Dominique Delande
# April, 30, 2020

I. Physics
In this section, I discuss physics, leaving numerical methods and implementation
details for section II.

I.1. What is it?
The and-python is a project written mainly in Python for numerical simulations
of disordered quantum systems, oriented towards transport properties
such as Anderson localization.
It provides different methods for solving the Schroedinger equation - either
time-independent or time-dependent - or the time-dependent Gross-Pitaevskii
equation. This allows to compute the Green function, and to deduce
from it different quantities such as eigenergies/eigenstates, density of states,
spectral function, total transmission across a sample, 
giving access to the localization length for the non-interacting (Schroedinger) 
system. It is aimed at computing quantities averaged over disorder realizations,
although it can also provide the statistical distributions of some quantities.
The present version is limited to 1D systems, although it is designed for 
easy extension to higher dimensions.

I.2. How does it work?
The Schroedinger or Gross-Pitaevskii is discretized on a regular grid in
configuration space. The kinetic energy (Laplace operator) is discretized 
with a 3-point (in 1D) approximation, altough higher degree approximations 
could be easily added.  
The resulting discretized Hamiltonian is a sparse matrix (tridiagonal in 1D),
which can be efficiently manipulated.
For the Gross-Pitaevskii, the additional nonlinear term is purely diagonal.
The choice of units is hbar=m=1.

I.3. Exact diagonalization
The simplest calculation is to diagonalize the discretized Hamliltonian to obtain
the energy levels and eigenfunctions, and to compute e.g. the Inverse Participatio
Ration or to study multifractality. The convergence properties are discussed in 
section XX. 
There are tow methods : 'lapack' for full diagonalization (scales as the cube
of the matrix dimension) with all eigenpairs and 'sparse' which computes 
few eigenvalues near a targeted energy. In the present version, only one eigenvalue is computed, but this is very easily changed in diag.py.
This is only for non-interacting systems.

I.4. Localization length (Lyapounov exponent)
The transfer matrix method makes it possible to solve the Schroedinger equation
at a given energy E (not necessarily an energy level) assuming some boundary 
condition on the left side and propagating to the right side. The propagation 
equation is in 1D a simple recurrence relation between \psi_{i-1}, \psi_i and 
\psi{i+1}, so that the propagation is trivial. In dimension d, one has to propagate 
a structure of dimension d-1 representing the wavefunction in an hyperplane 
perpendicular to the direction of propagation. The numerical method is intrisincally
unstable, so that it typically picks an exponentially growing solution, with a 
positive Lyapounov exponent. In 1D, the only Lyapounov exponent is the inverse of 
the localization length. 
For a certain system size (much longer than the localization length), the 
statistical distribution of the Lyapounov exponent is close to a Gaussian 
distribution whose width decays as the inverse square root of the system size.
Thus, the longer the system, the better. The same uncertainty is obtained by using
many small systems or a single long one, all what matters is the sum of all lengths.
As the number of Flops is proportional to the system size, the optimum is to use a
system size significantly longer than the localization length, but not much longer,
for a better use of the caches.
This is only for non-interacting systems.

I.5. Solving the time-dependent Schroedinger or Gross-Pitaevskii equation
One starts with an arbitrary initial state, which can be anything: a Gaussian 
wavepacket, a plane wave, a state localized on a single site, or whatever the
user builds.
There are two methods to solve the evolution equation: the slow "ODE" method
and the fast "Chebyshev" method.
I.5.1. ODE method
The discretized Schroedinger or Gross-Pitaevskii equation is a set of coupled
differential equations between the various \psi_i(t), which can be solved using
a standard Python package for ODEs. The advantage is that it works for arbitrary
interaction strengths in the GPE. The drawback is that it is slow. Especially 
for the Schroedinger equation, the equations are linear, but this is not used by the 
algorithm.
I.5.2. Chebyshev method
In the non-interacting case, the evolution operator over time dt is exp(-i*H*dt)
so that:
 \psi(dt) = \exp{-iHdt}\psi(t=0)
The evolution operator is never explicitly built, but it can formally be written
as an infinite series of Chebyshev polynomials of H. The coefficients of theis 
series decay with the order, so that the series can be efficiently propagated.
Thanks to the recurrence relations between Chebyshed polynomials, it is possible
to build recursively \psi(dt). If the series is truncated at order N, 
the calculation involves N matrix vector products of the type H|\phi>. 
Because of the sparse  structure of H, this can be computed efficiently. 
The number of terms in the series is roughly range(H)*dt, 
where range(H)=E_max_E_min is the spectral range of the 
Hamiltonian. This is valid when N is large, for smaller range(H)*dt, N has to be 
taken slightly larger, with a minimum value of the order of 10. As the CPU time 
needed scales like N, the cost of propagating over a long time (by chaining many
elementary Chebyshev propagations) is almost independent of the time step, provided
range(H)*dt is >> 1. A basic rule of thumb is not to take time steps making N 
smaller than 10.
The interacting case is slightly more complicated, as the non-linear term is time 
dependent and thus cannot be included in the Chebyshev expansion. The program
uses a split-step algorithm by interleaving elementary propagation operators
\exp{-iHdt} and \exp{-ig|\psi|^2dt}. Propagation with \exp{-iHdt} uses the 
previously described Chebyshev expansion while the non-linear propagation 
\exp{-i g|\psi|^2dt} is diagonal in configuration space. Of course, this is a good 
approximation only if g is sufficiently small. Hence, the larger g, the smaller the 
time step dt to be taken. This may be a drawback because the number N of terms in 
the Chebyshev expansion may become small, implying a loss of efficiency.
The program records tha maximum value of g\psi|^2dt reached during the all 
propagation. A rule of thumb is that it should be smaller than 0.1. Sometimes, 
it has been observed that smaller values, say 0.02-0.05, are needed for
good convergence.

I.6. Spectral function
The spectral function can be computed by Fourier transform of the autocorrelation
function C(t)=<\psi(0)|\psi(t)>, with \psi(0) an arbitrary state. For the standard
spectral function, \psi(0) is a plane wave, but the program allows any initial
state. Especially, if \psi(0) is localized on a single site, the Fourier transform
of C(t) is the local density of states, which, after configuration averaging, gives 
the average density of states per unit volume.
One has to specify the requires energy resolution (which will give the total
propagation time 2\pi/energy_resolution) and the total energy range: because of 
the performed FFT, the spectrum is folded in an interval Delta E = 2\pi/dt,
where dt is the elementary time step for propagation. Thus, the program uses 
dt=2\pi/energy_range. The spectrum is output as symmeytric around E=0, but this
is easy to change. Note that the CPU time spent in the Fourier transform 
time->energy is usually negligible.
The spectral function can be computed in the interacting case, simply as the
Fourier transform of the nonlinear C(t), but its physical meaning is not 
completely clear.

II. Implementation


