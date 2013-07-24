################################################################################
# 24.07.13 J. Rugis
################################################################################

OS_SIZE = 64
OS_ARCH = x86_64

CUDA_PATH       ?= /share/apps/NVIDIA/CUDA-5.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# MPI check and binaries
MPICXX           = $(shell which mpicxx)
ifeq ($(MPICXX),)
      $(error MPI not found.)
endif

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
EXTRA_CCFLAGS   ?=

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30)

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      NVCCFLAGS += -lineinfo
      TARGET    := release
endif

# OS-specific build flags
LDFLAGS := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS := -m64
NVCCFLAGS := -m64

# Target rules
all: FDTD_02

utils.o: utils.cpp utils.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $< 

cudaEH3d.o: cudaEH3d.cu cudaEH3d.cuh
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

source.o: source.cpp source.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $< 

spaceEH3d.o: spaceEH3d.cpp spaceEH3d.h
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

model3d.o: model3d.cpp model3d.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

main.o: FDTD_02.cpp
	$(MPICXX) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

FDTD_02: main.o utils.o cudaEH3d.o model3d.o spaceEH3d.o source.o
	$(MPICXX) $(CCFLAGS) -o $@ $+ $(LDFLAGS) $(EXTRA_LDFLAGS)

clean:
	rm -f FDTD_02 main.o utils.o cudaEH3d.cu.o cudaEH3d.o model3d.o spaceEH3d.o source.o
