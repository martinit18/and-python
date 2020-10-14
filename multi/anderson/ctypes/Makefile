COMPILER?=intel

ifeq ($(COMPILER),intel)
CC=icc
#CFLAGS=-Ofast -xHost -ftz -fno-alias -qopt-prefetch=4 -qopt-report=4 -qopt-report-phase=vec -no-prec-div -IPF-fp-relaxed -fp-model fast=2 
#CFLAGS=-Ofast -xHost -ftz -fno-alias -no-prec-div -IPF-fp-relaxed -fp-model fast=2 
export CFLAGS=-Ofast -xAVX -axCORE-AVX2,CORE-AVX512 -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -fargument-noalias-global -complex-limited-range -qopt-calloc -qopt-report=4 -qopt-report-phase=vec
#export CFLAGS=-Ofast -xHost -ftz -finline -fno-alias -IPF-fp-relaxed -restrict -fargument-noalias-global -complex-limited-range -qopt-report=4 -qopt-report-phase=vec -qopt-calloc 
endif

ifeq ($(COMPILER),gnu)
CC=gcc
CFLAGS=-Ofast -march=native -Wall -fopt-info-vec
endif

ifeq ($(COMPILER),llvm)
CC=clang
CFLAGS=-Ofast -march=znver2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
endif

all: chebyshev.so

%: %.c

%: %.o
	$(CC) ${CFLAGS} $^ -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
%.so : %.c
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@

clean :
	rm *.o *.so
	
