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
the energy levels and eigenfunctions, and to compute e.g. the Inverse Participation
Ratio or to study multifractality. 
There are tow methods : 'lapack' for full diagonalization (scales as the cube
of the matrix dimension) with all eigenpairs and 'sparse' which computes 
few eigenvalues near a targeted energy. In the present version, 
only one eigenvalue is computed, but this is very easily changed in diag.py.
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
II.1. Python modules
The software is almost entirely written in Python, using mostly the standard
modules numpy and scipy. However, few pieces have also pure C implementations,
for the inner heavy computational routines. These may speed up the calculation
by roughly one order of magnitude. Note that there is always a 
Python implementation available, which is 100% compatible. See section II.12 for
details.
The list of modules used is:
  os
  time
  math
  numpy
  getpass
  copy
  sys
  mkl
  configparser
  timeit
  mkl_random
  scipy.integrate.ode
  scipy.sparse.linalg
  scipy.special
  anderson
While the module mkl_random is necessary (one could also implement the random
part using numpy.random), the module mkl itself is optional. It have to be checked
whether it really speads up things.
All the modules are available using anaconda, which is recommended.
The last module "anderson" contains all the specific code of this software.

II.2. The "anderson" module
This module contains all the basic code for the calculations. It contains:
  __init.py__  for the basic structures and methods
  diag.py for exact diagonalization routines
  io.py for input/output routines
  lyapounov.py for calculation of the Lyapounov exponent
  propagation.py for temporal propagation

II.3. Classes
The code defines the following classes:
  Timing in __init__.py which is used for monitoring time spent in the various 
    calculations
  Potential in __init__.py which defines a disordered potential
  Hamitonian in __init__.py, daughter of Potential, which defines a full Hamitonian
    (note that the Potential class is almost useless and may disappear)
  Wavefunction in __init__.py which defines a wavefunction, discretized in 
    configuration space (but also has a momentum space component)
  Diagonalization in diag.py which defines the diagonalization to be used
  Lyapounov in lyapounov.py which defines where to compute the Lyapounov exponent
  Temporal_propagation in prop.py which defines the parameters of the 
    temporal propagation
  Measurement in prop.py which defines the various measurements performed during 
    the temporal propagation
  Spectral_function in prop.py which defines the parameters for the calculation
    of the spectral function
In a future release, we should document all methods of these classes.
At the moment, there are some routines which should be turned to methods in a
class. Cleaning up these things should be done...

II.4. Diagonalization
Everything is in the diag.py file. Very primitive calculation, computes basically
the IPR of a single state using the compute_IPR routine.
The diagonalization routine is either numpy.linalg.eigh (Lapack diagonalization)
or scipy.sparse.linalg (Sparse diagonalization), which internally uses Arpack.
Of course, sparse diagonalization is much faster for large matrices. It can easily
go to matrices of size 1.e7, the main limitation being memory allocation.
It should be good to explore the specialized module primme https://github.com/
primme/primme, not available in conda, but easy to get using pip.
Another interesting possibility would be to interface with Jadamilu.

II.5. Lyapounov
Everything is in the lyapounov.py file. The compute_lyapounov method applied on a
Lyapounov object creates a specific disorder and computes the lyapounov exponent
on a regular grid of energies. It has a specific C implementation, much much 
faster, see section II.12.

II.6. Temporal propagation
Everything is in the propagation.py file. The main routine is gpe_evolution, which
evolves an initial state in a disordered potential and measures some interesting 
quantities. If the Chebyshev method is used, there is a C implementation available,
about 10 times faster, see section II.12.

II.7. Spectral function
Everything is in the propagation.py file. It basically works by a temporal 
propagation (see section II.6) followed by a time->energy Fourier transform in the 
method Spectral_function.compute_spectral_function. It internally uses numpy FFT,
which is usually very fast.

II.8. Basic examples
There are basic examples in the files:
  diag/compute_IPR.py
  lyapounov/compute_lyapounov_vs_energy.py
  prop/compute_prop.py
  aek0/compute_spectral_function.py
All these examples are supposed to run in few seconds.
The structure is the same for each example:
  1. Read parameters of the calculation in file params.dat (see section II.9)
  2. Prepare the Hamiltonian structure
  3. Prepare the calculation to be performed, including numpy arrays for the 
     results and the "header strings" for output (see section II.10)
  4. Loop over various disorder configurations
  5. Gather the results averaged over disorder configurations in a single structure
  6. Output results in various files (see section II.10).
  7. Print a summary of CPU time and number of Flops.
These examples can be used as templates for your own calculations.
There should be comments in these files, but there are only few at the moment.

II.9. Input parameters
In the examples, the parameters are in a file name params.dat which is parsed
using the Python configparser module. Of course, it is always possible to
define the various parameters using other methods, e.g. direct assignement at the
beginning of the Python script or parsing a XML file (not implemented yet).
Using the configparser module, the input file (typically params.dat) is divided
in several sections, each section refering to a family of parameters. 
For example, the params.dat file for the Lyapounov example is:

#####################################################################
[System]
# System size in natural units
size = 1.e4
# Spatial discretization
delta_x = 0.25
# either periodic or open
boundary_condition = open

[Disorder]
#  Various disorder types can be used
type = anderson_gaussian
# type = regensburg
# type = konstanz
# type = singapore
# type = speckle
# Correlation length of disorder (ignored for discrete Anderson disorders)
sigma = 1.591549431
# Disorder strength
V0 = 0.01

[Nonlinearity]
# g is the nonlinear interaction
g = 0.0

[Wavefunction]
# Either plane_wave or gaussian_wave_packet
initial_state = plane_wave
# make sure that k_0_over_2_pi*my_system_size is an integer
k_0_over_2_pi = 0.16
#0.16
# Size of Gaussian wavepacket
sigma_0 = 10.0

[Propagation]
# Propagation method can be either 'ode' or 'che' (for Chebyshev)
method = che
# Internally, the wavefunction can be stored either as a complex numpy array
# or as a real numpy array (twice the size) containing the real parts
# followed by the imaginary parts
# 'real' is usually a bit faster (10%)
data_layout = real
# Total duration of the propagation
t_max = 2000.
# Elementary time step
delta_t = 2.

[Spectral]
# Range of energies covered
range = 20.0
# Resolution of the spectral function
resolution = 0.02

[Averaging]
n_config = 8

[Measurement]
delta_t_measurement = 2.
# Not presently used
#first_measurement = 0
# Whether to measure the density in configuration space
density = True
# Whether to measure the density in momentum space
density_momentum = True
# Whether to measure <x(t)> and <x^2(t)>
dispersion_position = True
# Whether to measure <p(t)>
dispersion_momentum = True
# Whether to compute the total energy (should be constant)
# and the nonlinear part of the energy (=0 if g=0)
dispersion_energy = True
# Whether to compute the average final wavefunction in configuration and momentum spaces
wavefunction_final = True
# Whether to compute the autocorrelation function <psi(0)|psi(t)>
autocorrelation = True
# Whether to compute the standard deviation (over disorder realizations) of
# x(t), x^2(t), p(t), E_total(t) and E_nonlinear(t)
# This is computed only for quantities whose average value is computed
dispersion_variance = True

[Diagonalization]
method = sparse
targeted_energy = 0.6
# Characteristics of the histogram of the IPR values
IPR_min = 0.0
IPR_max = 0.01
number_of_bins = 50

[Lyapounov]
# Minimum energy for which the Lyapounov is computed
e_min = 0.0
# Maximum energy for which the Lyapounov is computed
e_max = 0.40
number_of_e_steps = 20
# Characteristics of the histogram of the Lyapounov values
e_histogram = 0.22
lyapounov_min = 0.0
lyapounov_max = 0.02
number_of_bins = 50
#####################################################################

The comments should be self-explanatory. 
Each example uses only the sections it needs, the other ones, if present, are 
ignored.

The three obligatory sections are:
[System] which defines the geometric properties of the system
[Disorder] which defines the properties of the Hamiltonian
[Averaging] which defines on how many disorder configurations are averaged

WARNING: When MPI is used (see section II.12), the parameter n_config
refers to the TOTAL number of configurations. This is different from
the an1d_propxx.c programs, where it was the number of configurations
per MPI process.

Other sections are as follows:
[Nonlinearity] to define g when needed (propagation and spectral function)
[Wavefunction] when an initial state is needed (propagation and spectral function)
[Propagation] for temporal propagation (propagation)
[Spectral] for spectral function (spectral function)
[Measurement] for temporal propagation (propagation)
[Diagonalization] for diagonalization (diagonalization)
[Lyapounov] for computation of the Lyapounov exponent (lyapounov)

In each section, there are parameters, each defined on a single line:
name_of_parameter = value_of_parameter
For some parameters, there are default values. For examples, all measurements
are turned off by default.
If a parameter with no default value is missing, the program stops with an error
message. 
Be careful to correctly write the name of a parameter, it is a source of common 
errors.

II.10. Output
Routines which conveniently outputs the results of the calculations are available
in io.py. 
The general idea is that each output file contains all the information needed to 
reproduce it, that is the values of all input parameters, before the results 
themselves. In the header of each output file, there are also informations
on the script used to produce the results, the machine on which it has been ran, 
how long the run took, and whether several MPI processes were used 
(see section II.12).
At the moment, only ASCII human-readable files are used.  
In a next release, it is planned to add HDF5 outputs.
The output files are produced by the numpy.savetxt routine. It consists in a header
(using the "header" parameter of numpy.savetxt) followed by a series of lines, each 
line containing at least two numerical values. For a temporal propagation, it can
be for example, time in column 1 and <x> in column 2.
Normally, the header_string contains all the information on the content of each 
column. The header is conveniently created using the output_string routine (in 
io.py)  called with the relevant parameters. It automatically adjusts to produce a 
header from the transmitted parameters.
The output_density routine (in io.py) outputs data with one, two or three columns,
for the different types of data: 'density', 'density_momentum', 
'spectral_function', 'IPR', 'histogram_IPR', 'lyapounov', 'histogram_lyapounov',
with the third column used for the standard deviation, if relevant.
One can also print a complex function ('autocorrelation') or a complex 
wavefunction in configuration ('wavefunction') of momentum 
('wavefunction_momentum') space.
The more specialized output_dispersion routine is for output of the various results
of a temporal propagation, such as <x(t0>, <p(t)>, <E_total(t)>..., optionally
with the standard deviation.

II.11. C routines
The most numerically intensive routines have been also written in C. They should
be strictly equivalent to the corresponding Python routines.
II.11.1 CFFI interface
They use the CFFI interface. Basically, it defines a proper C-Python interface
and a C source code embedded in a Python wrapper.
For example, the lyapounov_build.py file contains at the beginning:
  from cffi import FFI
  ffibuilder = FFI()
  ffibuilder.cdef("double core_lyapounov_cffi(const int dim_x, const int loop_step, const double * disorder, const double energy, const double inv_tunneling);")
which defines a routine double core_lyapounov_cffi which can be called from Python.
The following of lyapounov_build.py defines the embedded C code:
  ffibuilder.set_source("_lyapounov",
  r"""
  #include <stdlib.h>
  #include <stdio.h>
  #include <string.h>
  ...
  double core_lyapounov_cffi(const int dim_x, const int loop_step, const double * disorder, const double energy, const double inv_tunneling)
  {
    ... calculation...
     return(gamma);
  }
  """)
  if __name__ == "__main__":
  ffibuilder.compile(verbose=True)

The code is compiled with the Makefile in the "anderson" directory.
The GNU, clang and Intel compilers may be used, just by setting the COMPILER 
environment variable (see in the Makefile). It seems that the Intel compiler
gives the fastest code, it is thus recommended to use it if available.

WARNING: The code is generated in a routine typicalled named:
_lyapounov.cpython-37m-x86_64-linux-gnu.so
It is compiled for the specific architecture of the host. It is thus not portable
and must be recompiled on each machine.
The compilation also produces intermediate _lyapounov.c and _lyapounov.o files,
which can be discarded.

II.11.2 On the Python side, the compute_lyapounov method in the Lyapounov class 
contains:
  try:
    from anderson._lyapounov import ffi,lib
    use_cffi = True
  #    print('Using CFFI version')
  except ImportError:
    use_cffi = False
    print("\n Warning, this uses the slow Python version, you should build the C version!\n")
which tries to load the CFFI version if available.
When the routine is needed, the code looks like:
      if use_cffi:
        tab_gamma[i_energy]=lib.core_lyapounov_cffi(dim_x, loop_step,ffi.cast('double *',ffi.from_buffer(H.disorder)), self.tab_energy[i_energy], inv_tunneling)
      else:
        tab_gamma[i_energy]=core_lyapounov(dim_x, loop_step, H.disorder, self.tab_energy[i_energy], inv_tunneling)
which uses the CFFI version if available, the Python version in the core_lyapounov
Python routine otherwise. 

II.11.3 Routines with a C version
There are two C codes in
lyapounov_build.py: core_lyapounov_cffi 
                    for the computation of the Lyapounov exponent
chebyshev_build.py: chebyshev_clenshaw_real and chebyshev_clenshaw_complex
                    for a single step of the Chebyshev propagation
                    the two routines correspond to the choice of the data_layout
One routine could be developped for the calculation of the total energy as it is a
bit costly. Computing <x>, <x^2>, <p>, <E_nonlinear> is significantly less
expensive, thus a C version is not so important. 

II.12. Parallelism using MPI
The code can be parallelized using MPI. The parallelization is only the trivial
one over disorder realizations (no use of the Python multiprocessing module). 
The reason is that it is both easier to program and slightly more efficient to use MPI.
This however requires the mpi4py module, which is not always available (not by default
on anaconda). Thus the code contains things like:
  try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    mpi_version = True
    environment_string += 'MPI version ran on '+str(nprocs)+' processes\n\n'
  except ImportError:
    mpi_version = False
    nprocs = 1
    rank = 0
    environment_string += 'Single processor version\n\n'

The MPI version is launched using e.g.:
  mpiexec -n 8 python compute_prop.py
If no mpi4py module is found, a standard sequential routine is used, which means
that the same calculation is performed 8 time |-(.

If MPI is activated, all I/O is done on process 0. Input data (parsed from the input
file) are broadcasted to other processes. Then, each process builds its own objects
and works with the defined methods, like without MPI.
After all configurations have been treated, the data are gathered/reduced on process 0,
which performs the remaining calculation and the output.
Altogether, it is no much change compared to the sequential code.

WARNING: When MPI is used, the parameter n_config refers to the TOTAL number of 
configurations. This is different from the an1d_propxx.c programs, 
where it was the number of configurations per MPI process.
  
  
  
  


