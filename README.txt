# README
# and-python project
# Version 4.1
# Author: Dominique Delande
# February, 21, 2023

WARNING: In this version, the following features are new or not yet implemented:
* Lyapounov. Not implemented at all for multidimensional systems. C interface in 1d is obsolete.
* Fast C versions of the most CPU intensive routines, using the ctypes interface are ony partially implemented
in dimension 3 and larger. 1d and 2d version are working well and are as fast as a pure C code.
* The spectral function calculation is completely new. It now uses the Kernel Polynomial Method (KPM) with
Chebyshev polynomials, rather than the old Time->Energy Fourier transform of the evolution operator. 
* There is a new optiun to choose between reproducible and irreproducible randomness. In the first case, 
  the same RNG is used between runs, so that the results should be identical.  
* There are inconsistencies on which file is used for reading parameters. See section II.9

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
The present version works in arbitrary dimension.

I.2. How does it work?
The Schroedinger or Gross-Pitaevskii is discretized on a regular grid in
configuration space. The kinetic energy (Laplace operator) is discretized 
with a 3-point (in 1D) approximation, altough higher degree approximations 
could be easily added.  
The resulting discretized Hamiltonian is a sparse matrix (tridiagonal in 1D),
which can be efficiently manipulated.
For the Gross-Pitaevskii, the additional nonlinear term is purely diagonal.
The choice of units is hbar=1, with the mass usually m=1, 
but can be adjusted with the parameter "one_over_mass".

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
** Currently does not work for multidimensional systems **
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
taken slightly larger. As the CPU time 
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
The program records the maximum value of g|\psi|^2dt reached during the whole
propagation (the so-called "maximum nonlinear phase"), and prints it. 
A rule of thumb for systems where the disorder is stronger than interaction
is that it should be smaller than 0.1. Sometimes, 
it has been observed that smaller values, say 0.02-0.05, are needed for
good convergence. When the interaction is larger than the disorder, the density |\psi|^2
is more uniform and larger values of the nonlinear phase can be used.
In any case, it is always good to check the conservation of the total energy, setting 
"dispersion_energy" to True in the parameter file.

I.6. Spectral function
NOTE: The spectral function calculation is completely new for version 4.0.
Previously it was using temporal propagation followed by a Time->Energy Fourier transform.
The new version uses direct calculation with the Kernel Polynomial Method (KPM) and Chebyshev polynomials.
It is very significantly faster, less noisy, meaning that less disorder realizations are needed.

The spectral function is defined as:
  A_\psi(E) = \overline{<\psi|\delta(H-E)|\psi>}
where the overline denotes averaging over disorder realizations.
In the KPM method, the function \delta(H-E) is computed as a sum over Chebyshev polynomials of the
Hamiltonian H acting on the initial state |\psi>. The method is more detailed in the documentation.

For the standard
spectral function, \psi is a plane wave, but the program allows any initial
state. Especially, if \psi is localized on a single site (option "point" for the "initial_state" variable
in the [Wavefunction] section of the parameter file, the spectral function is 
the local density of states, which, after configuration averaging, gives 
the average density of states per unit volume. 
A better option, leading to less noisy density of states is to use:
  intiial_state = random
which chooses uncorrelated complex Gaussian variables on each site and also provides the density of states.

One has to specify the requested energy resolution and the total energy range.
It is IMPORTANT that the energy range covers the full energy spectrum. Otherwise, disaster is guaranteed
at the edges of the energy range which spoils everything. This is in contrast with the versions before 4.0,
where the spectral function was computed by Fourier transform and the full spectral function was folded in 
the requested energy range.

The spectral function can be computed in the interacting case, where the Hamitonian used in \delta(H-E) is now
  H = H-0 + disorder + multiplicative_factor_for_interaction*g*|\psi|^2
where the "multiplicative_factor_for_interaction" (default"0) is defined in the [Spectral] section of the parametre file.
  multiplicative_factor_for_interaction = 0.5
ensures that the average energy (preserved during the nonlinear evolution) is given by \int{dE E A_\psi(E)} and constant during the 
temporal evolution of \psi.

  
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
  configparser
  argparse
  timeit
  scipy.integrate
  scipy.sparse
  scipy.special
  
In addition, there are 6 optional modules: mkl, mkl_random, mkl_fft, numba, ctypes and  numpy.ctypeslib.
The mkl module is only used to set the number of OpenMP threads, it is probably useless.
If the mkl_random module is present, the MKL random number generator is used. If not,
the program uses numpy.random. 
If the mkl_fft module is present, most FFTs (the ones which use a lot of CPU time) 
are performed using the MKL FFT routines. If not present, the program uses numpy.fft,
which is a bit slower.
The ctypes and numpy.ctypeslib modules (if available) make it possible to use a C version of some numerically
intensive routines. It is usually about 10 times faster than the Python code.
The numba module is used to speed up a bit the pure Python code. 
All the modules are available using anaconda.

The last module "anderson" contains all the specific code of this software.

II.2. The "anderson" module
This module contains all the basic code for the calculations. It contains:
  __init.py__  for the basic structures and methods
  diag.py for exact diagonalization routines
  io.py for input/output routines
  lyapounov.py for calculation of the Lyapounov exponent
  propagation.py for temporal propagation*

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
    of the spectral function or the density of states
In a future release, we should document all methods of these classes.
At the moment, there are some routines which should be turned to methods in a
class. Cleaning up these things should be done...

II.4. Diagonalization
Everything is in the diag.py file. Very primitive calculation, computes basically
the IPR of a single state using the compute_IPR routine.
The diagonalization routine is either numpy.linalg.eigh (Lapack diagonalization)
or scipy.sparse.linalg (Sparse diagonalization), which internally uses Arpack.
Of course, sparse diagonalization is much faster for large matrices. It can easily
go to matrices of size 1.e7 in 1d, the main limitation being memory allocation.
In 2d, it works quite well. Preliminary tests in 3d show that it does not behave too well.
It should be good to explore the specialized module primme https://github.com/primme/primme, not available in conda, 
but easy to get using pip.
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
Everything is in the propagation.py file, although it could probably be moved to a separate module.

II.8. Basic examples
There are basic examples in the files:
  diag/compute_IPR.py
  lyapounov/compute_lyapounov_vs_energy.py
  prop/compute_prop.py
  spectral_function/compute_spectral_function.py
All these examples are supposed to run in few seconds.
The structure is the same for each example:
  1. Read parameters of the calculation in the parameter file (see section II.9)
  2. Prepare the Hamiltonian structure
  3. Prepare the calculation to be performed, including numpy arrays for the 
     results and the "header strings" for output (see section II.10)
  4. Loop over various disorder configurations
  5. Gather the results averaged over disorder configurations in a single structure
  6. Output results in various files (see section II.10).
  7. Print a summary of CPU time and number of Flops.
These examples can be used as templates for your own calculations.
There should be comments in these files, but there are only few at the moment.
How to launch these scripts is explained in sections II.9 (serial version) and II.12 (MPI version).

II.9. Input parameters
In the examples, the parameters are read from a "parameter file".
The present stage is a bit inconsistent. For temporal propagation, it is required
to give as argument the name of the parameter file, i.e. something like:
  python compute_prop.py my_parameter_file
For other python scripts, the parameters are expected to be in the "params.dat" and the script is launched with e.g.
  python compute_IPR.py
In a future version, every script will require explicitly the name of the parameter file.
The parameter file is parsed
using the Python configparser module. Of course, it is always possible to
define the various parameters using other methods, e.g. direct assignement at the
beginning of the Python script or parsing a XML file (not implemented yet).
Using the configparser module, the input file (typically params.dat) is divided
in several sections, each section refering to a family of parameters. 
For example, the parameter file params.dat for the spectral_function example is:

#####################################################################
[System]
dimension = 2
# System size in natural units
size_1 = 20.
size_2 = 20.
size_3 = 5.
size_4 = 1.
# Spatial discretization
delta_1 = 0.5
delta_2 = 0.5
delta_3 = 0.25
delta_4 = 0.25
# either periodic or open
boundary_condition_1 = periodic
boundary_condition_2 = periodic
boundary_condition_3 = periodic
boundary_condition_4 = periodic
use_mkl_random = True
use_mkl_fft = True

[Disorder]
#  Various disorder types can be used
# type = anderson
# type = anderson_gaussian
# type = regensburg
type = konstanz
# type = singapore
# type = speckle
# Correlation length of disorder
sigma = 1.0
# Disorder strength
V0 = 1.0

[Nonlinearity]
# g is the nonlinear interaction
g = 0.0

[Wavefunction]
# Either plane_wave or gaussian_wave_packet
initial_state = plane_wave
#initial_state = gaussian_wave_packet
# make sure that k_0_over_2_pi_i*size_i is an integer
k_0_over_2_pi_1 = 0.0
k_0_over_2_pi_2 = 0.0
# Size of Gaussian wavepacket
sigma_0_1 = 10.0
sigma_0_2 = 10.0

[Propagation]
# Propagation method can be either 'ode' or 'che' (for Chebyshev)
method = che
data_layout = real
#data_layout = complex
# Total duration of the propagation
t_max = 25.0
# Elementary time step
delta_t = 1.0

[Averaging]
n_config = 10000

[Measurement]
delta_t_measurement = 1.0
#first_measurement = 0
density = True
density_momentum = True
dispersion_position = True
dispersion_position2 = True
#dispersion_momentum = True
#dispersion_energy = True
dispersion_variance = True
#wavefunction = True
#wavefunction_momentum = True
#autocorrelation = True


[Spectral]
range = 20.0
resolution = 0.01

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
the an{1,2,3}d_propyy.c programs, where it was the number of configurations
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
line containing at least two numerical values. For a 1D temporal propagation, it can
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
of a temporal propagation, such as <x(t)>, <p(t)>, <E_total(t)>..., optionally
with the standard deviation.

II.11. C routines
The most numerically intensive routines have been also written in C. They should
be strictly equivalent to the corresponding Python routines.
In the first versions, the CFFI interface between Python and C was used. 
Starting from version 3.0, the ctypes interface is used. In a near future, it may be that
the pybind11 interface between Python and C++ will be used.

II.11.1 Ctypes interface
The basic idea is to write a pure traditional C code implementing the calculation-intensive routines. 
For the temporal propagation using the Chebyshev method, there is for exemple in "anderson/ctypes/chebyshev.c" a C routine defined by:
  double chebyshev_real(const int dimension, const int * restrict tab_dim, const int max_order,  const int * restrict tab_boundary_condition, double * restrict wfc, double * restrict psi, double * restrict psi_old,  const double * restrict disorder, const double * restrict tab_coef, const double * tab_tunneling, const double two_over_delta_e, const double two_e0_over_delta_e, const double g_times_delta_t, const double e0_times_delta_t);
The code is compiled using "make" with the Makefile in the "anderson/ctypes" directory.
The GNU, clang, Intel and Intel Oneapi compilers may be used, just by setting the COMPILER 
environment variable (see in the Makefile). It seems that the Intel compiler
gives the fastest code, it is thus recommended to use it if available.
Note that, on recent distributions like Ubuntu 22.04, the old Intel-21 compiler "icc" does not work any longer. You may use the "icc" compiler from
Intel Oneapi, setting the variable COMPILER to "intel". As this "icc" compiler is no longer developed, it will probably fail in a near future.

WARNING: The code is generated in a routine typically name chebyshev.so
Depending on the options in the Makefile, it may run only on the specific machine where it has been compiled.
If you encounter some error like "Illegal instruction", you have te recompile the C code.
If using the Intel compiler on Linux, the code should be portable and optimized for all Intel/AMD architectures.

On the python side, nothing has to be modified in the main script. Only the code inside the "anderson/propagation.py" file
is modified to use C routines if available. For exemple, in the "gpe_evolution" routine, there are the following lines:
    if not H.spin_one_half and self.want_ctypes:
      try:
        import ctypes
        import numpy.ctypeslib as ctl
        self.has_specific_full_chebyshev_routine = True
        self.chebyshev_ctypes_lib=ctypes.CDLL(anderson.__path__[0]+"/ctypes/chebyshev.so")
        if self.data_layout=='real':
          self.chebyshev_ctypes_lib.chebyshev_real.argtypes = [ctypes.c_int, ctl.ndpointer(np.intc), ctypes.c_int, ctl.ndpointer(np.intc),\
            ctl.ndpointer(np.float64), ctl.ndpointer(np.float64), ctl.ndpointer(np.float64), ctl.ndpointer(np.float64), ctl.ndpointer(np.float64), ctl.ndpointer(np.float64),\
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
          self.chebyshev_ctypes_lib.chebyshev_real.restype = ctypes.c_double
          if not (hasattr(self.chebyshev_ctypes_lib,'chebyshev_real') and    hasattr(self.chebyshev_ctypes_lib,'elementary_clenshaw_step_real_'+str(H.dimension)+'d')):
            self.has_specific_full_chebyshev_routine = False
            self.chebyshev_ctypes_lib = None
            if H.seed == 0 :
              print("\nWarning, chebyshev C library found, but without routine for real data layout and dimension "+str(H.dimension)+", this uses the slow Python version\n")
        if self.data_layout=='complex':
          self.chebyshev_ctypes_lib.chebyshev_complex.argtypes = [ctypes.c_int, ctl.ndpointer(np.intc), ctypes.c_int, ctl.ndpointer(np.intc),\
            ctl.ndpointer(np.complex128), ctl.ndpointer(np.complex128), ctl.ndpointer(np.complex128), ctl.ndpointer(np.float64), ctl.ndpointer(np.float64), ctl.ndpointer(np.float64),\
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
          self.chebyshev_ctypes_lib.chebyshev_complex.restype = ctypes.c_double
          if not (hasattr(self.chebyshev_ctypes_lib,'chebyshev_real') and    hasattr(self.chebyshev_ctypes_lib,'elementary_clenshaw_step_complex_'+str(H.dimension)+'d')):
            self.has_specific_full_chebyshev_routine = False
            self.chebyshev_ctypes_lib = None
            if H.seed == 0 :
              print("\nWarning, chebyshev C library found, but without routine for complex data layout and dimension "+str(H.dimension)+", this uses the slow Python version\n")
      except:
        self.has_specific_full_chebyshev_routine = False
        self.chebyshev_ctypes_lib = None
        if H.seed == 0 :
          print("\nWarning, no ctypes module, no numpy.ctypeslib module or no chebyshev C library found, this uses the slow Python version!\n")
    self.use_ctypes = self.has_specific_full_chebyshev_routine and self.want_ctypes

which test if the proper routines exist in the "anderson/ctypes/chebyshev.so" library and, if succesful, defines the types of the arguments and return value of the useful routines. This uses the standard ctypes and numpy.ctypeslib modules.

Then, in the chebyshev_step routine, the C code is called if it is available with:
  nonlinear_phase = chebyshev_ctypes_lib.chebyshev_real(H.dimension, np.asarray(H.tab_dim,dtype=np.intc), max_order, H.array_boundary_condition,\
        local_wfc, psi, psi_old, H.disorder.ravel(), propagation.tab_coef, np.asarray(H.tab_tunneling),\
        H.two_over_delta_e, H.two_e0_over_delta_e, H.interaction*propagation.delta_t, H.medium_energy*propagation.delta_t)
so that the conversion between Python and C variables and pointers is properly done.
If the C code is not available, an equivalent Python code is used, but it is MUCH slower.
 
At the moment, there are only the two routines chebyshev_real and chebyshev_complex in chebyshev.c 
(the two routines correspond to the choice of the data_layout). These routines internally manage both the 1d and 2d cases.
The 3d case is only supported for the "complex" data_layout. In a near future, it could be that the "real" data layout will be completely remved,
as it is not siginificantly faster.
One routine could be developped for the calculation of the total energy as it is a
bit costly. Computing <x>, <x^2>, <p>, <E_nonlinear> is significantly less
expensive, thus a C version is not so important. 
       
II.11.2 Old CFFI interface (no longer supported)
The routines use the CFFI interface. Basically, it defines a proper C-Python interface
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

On the Python side, the compute_lyapounov method in the Lyapounov class 
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


II.12. Parallelism using MPI
The code can be parallelized using MPI. The parallelization is only the trivial
one over disorder realizations (no use of the Python multiprocessing module). 
The reason is that it is both easier to program and slightly more efficient to use MPI.
This however requires the mpi4py module, which is not always available (not by default
on anaconda). When used on the LKB machines or on the ponyo cluster, there is a specific environment
for the OpenMPI implementation of MPI, which can be activated using:
  conda activate openmpi
  
The code must contain near the beginning something like:

  mpi_version, comm, nprocs, rank, mpi_string = anderson.determine_if_launched_by_mpi()

which checks whether the python script is called inside a MPI process. 
If this is the case, mpi_version is True, the  MPI communicator is comm, the number of
MPI processes is nprocs, the rank of the current process is rank, and
mpi_string contains some minimal MPI information
If not run inside MPI, mpi_version=False, comm=None, nprocs=1, rank=0 and mpi_string contains proper information

The MPI version is launched using e.g.:
  mpiexec -n 8 python compute_prop.py my_parameter_file
If no mpi4py module is found, the program stops with an error message.

If MPI is activated, all I/O is done on process 0. Input data (parsed from the input
file) are broadcasted to other processes. Then, each process builds its own objects
and works with the defined methods, like without MPI.
After all configurations have been treated, the data are gathered/reduced on process 0,
which performs the remaining calculation and the output.
Altogether, it is not much change compared to the sequential code.

WARNING: When MPI is used, the parameter n_config refers to the TOTAL number of 
configurations. This is different from the anxd_propxx.c programs, 
where it was the number of configurations per MPI process.
  
  
  
  


