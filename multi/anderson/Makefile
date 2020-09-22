COMPILER?=intel

ifeq ($(COMPILER),intel)
export CC=icc
#export CFLAGS=-Ofast -xHost -ftz -fno-alias -qopt-prefetch=4 -qopt-report=4 -qopt-report-phase=vec -no-prec-div -IPF-fp-relaxed -fp-model fast=2 
# Do not put -ip, -ipo or -ipo2 as they semm useless
export CFLAGS=-Ofast -xAVX -axCORE-AVX2,CORE-AVX512 -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -fargument-noalias-global -complex-limited-range -qopt-calloc
# -qopt-mem-layout-trans=3
# -qopt-prefetch=2
#export CFLAGS= -O3 -xAVX -axCORE-AVX2,CORE-AVX512 -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -fargument-noalias-global -complex-limited-range  
#export CFLAGS=-Ofast -xHost -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -fargument-noalias-global -complex-limited-range
#-qopt-prefetch=2
#-qopt-mem-layout-trans=3 
# Option to print vectorization status -qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout
#-qopt-streaming-stores=auto
#-funroll-loops
export LDSHARED=icc
endif

ifeq ($(COMPILER),gnu)
export CC=gcc
export CFLAGS=-Ofast -march=native
export LDSHARED=gcc
endif

ifeq ($(COMPILER),llvm)
export CC=clang
export CFLAGS=-Ofast -march=native
export LDSHARED=clang
endif

#LDFLAGS=-lgsl -lgslcblas -lm
export LDFLAGS = -pthread -shared -B /opt/anaconda3/compiler_compat -L/opt/anaconda3/lib -Wl,-rpath=/opt/anaconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/
%: %.c

%: %.o
	$(CC) ${CFLAGS} $^ -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
%.so : %_build.py
	python $^

all: chebyshev.so lyapounov.so

clean :
	rm *.o *.so
	
