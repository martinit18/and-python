COMPILER?=oneapi

SHELL := /bin/bash

export WARNING=""

ifeq ($(COMPILER),intel)
CC=icc
#CFLAGS=-Ofast -xHost -ftz -fno-alias -qopt-prefetch=4 -qopt-report=4 -qopt-report-phase=vec -no-prec-div -IPF-fp-relaxed -fp-model fast=2 
#CFLAGS=-Ofast -xHost -ftz -fno-alias -no-prec-div -IPF-fp-relaxed -fp-model fast=2 
#export CFLAGS=-Ofast -xAVX -axCORE-AVX2,CORE-AVX512 -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -qopt-report=4 -qopt-report-phase=vec -fargument-noalias-global -complex-limited-range -qopt-calloc
#export CFLAGS=-Ofast -xHost -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -qopt-report=4 -qopt-report-phase=vec -fargument-noalias-global -complex-limited-range -qopt-calloc
# The following options are somehow dangerous (do not know why...)
# -fargument-noalias-global -complex-limited-range -fargument-noalias -qopt-calloc
#export CFLAGS=-Ofast -xHost -ftz -qopt-report=4 -qopt-report-phase=vec -finline -fno-alias -IPF-fp-relaxed -restrict 
export CFLAGS=-Ofast -xCORE-AVX2 -mauto-arch=core-avx512 -ftz -qopt-report=4 -qopt-report-phase=vec -qopt-zmm-usage=high 
export WARNING="\nWARNING, compiling with the Intel compiler icc does not work if conda is activated\n  If this is the case, run 'conda deactivate' and try again\n"
#export CFLAGS=-Ofast -xCORE-AVX2 -qopt-report=4 -qopt-report-phase=vec -ftz
#export CFLAGS=-Ofast -xHost -qopt-report=4 -qopt-report-phase=vec -ftz
#export CFLAGS=-O0
endif

ifeq ($(COMPILER),oneapi)
CC=icx 
CFLAGS=-O3 -xavx -axcore-avx2 -axskylake-avx512 -Wunknown-pragmas -fp-model fast -fno-alias -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize --gcc-toolchain=/usr
#CFLAGS=-O3 -xskylake-avx512 -Wunknown-pragmas -fp-model fast -fno-alias -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
#CFLAGS=-O3 -xHost -Wunknown-pragmas -fp-model fast -fno-alias -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize  -ftz
#CFLAGS=-O0 
endif

ifeq ($(COMPILER),gnu)
CC=gcc
CFLAGS=-Ofast -march=native 
#-Wall 
#-fopt-info-vec
endif

ifeq ($(COMPILER),llvm)
CC=clang
CFLAGS=-Ofast -march=native -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize 
CFLAGS=-Ofast -march=native
endif

all: chebyshev.so lyapounov.so
	
%: %.c

%: %.o
	$(CC) ${CFLAGS} $^ -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
%.so : %.c
	@echo -e ${WARNING}
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

clean :
	rm *.o *.so
	
