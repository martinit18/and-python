COMPILER?=intel

ifeq ($(COMPILER),intel)
export CC=icc
#export CFLAGS=-Ofast -xHost -ftz -fno-alias -qopt-prefetch=4 -qopt-report=4 -qopt-report-phase=vec -no-prec-div -IPF-fp-relaxed -fp-model fast=2 
# Do not put -ip, -ipo or -ipo2
export CFLAGS=-Ofast -xAVX -axCORE-AVX2,CORE-AVX512 -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -qopt-prefetch=5 -fargument-noalias-global -complex-limited-range -qopt-mem-layout-trans=3
#export CFLAGS=-Ofast -xCORE-AVX512 -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -qopt-prefetch=5 -fargument-noalias-global -complex-limited-range -qopt-mem-layout-trans=3
#-axCORE-AVX512 
#export CFLAGS=-Ofast -xHost -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -qopt-prefetch=5 -fargument-noalias-global -complex-limited-range -qopt-mem-layout-trans=3

#-qopt-streaming-stores=auto
#-qopt-mem-layout-trans=3
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
	
