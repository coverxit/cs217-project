BINDIR = bin
EXECUTABLE = lzss

GCC = g++
NVCC = nvcc

CFLAGS = -c -O3 -std=c++11 -g
GCC_LDFLAGS = -pthread
NVCC_LDFLAGS = -Xcompiler="-pthread"

GCC_TARGET = $(BINDIR)/$(EXECUTABLE)_gcc
NVCC_TARGET = $(BINDIR)/$(EXECUTABLE)_nvcc

CPP_SOURCES =						\
	main.cpp						\
	LZSSFactory.cpp					\
	CPU/ST/CPUSingleThreadLZSS.cpp	\
	CPU/MT/CPUMultiThreadLZSS.cpp

COMMON_SOURCES =					\
	MatchHelper/MatchHelper.cu		\
	CPU/BlockHelper.cu

CUDA_SOURCES =						\
	CUDA/CUDALZSS.cu

CPP_OBJECTS = $(patsubst %.cpp,%.o,$(CPP_SOURCES))
COMMON_OBJECTS = $(patsubst %.cu,%.o,$(COMMON_SOURCES))
CUDA_OBJECTS = $(patsubst %.cu,%.o,$(CUDA_SOURCES))

.PHONY: all
all:
	@echo "Specify a target (gcc or nvcc)!"

.PHONY: gcc-setup
gcc-setup:
	$(eval CC = $(GCC))
	$(eval CFLAGS = -x c++ -DGCC_TARGET $(CFLAGS))
	$(eval LDFLAGS = $(GCC_LDFLAGS))

.PHONY: gcc
gcc: clean gcc-setup $(GCC_TARGET)

.PHONY: nvcc-setup
nvcc-setup:
	$(eval CC = $(NVCC))
	$(eval CFLAGS = -dc $(CFLAGS))
	$(eval LDFLAGS = $(NVCC_LDFLAGS))

.PHONY: nvcc
nvcc: clean nvcc-setup $(NVCC_TARGET)
	
%.o: %.cpp
	@echo '[$(CC)] compiling cpp:' $<
	@$(CC) $(CFLAGS) -o $@ $<

%.o: %.cu
	@echo '[$(CC)] compiling cu:' $<
	@$(CC) $(CFLAGS) -o $@ $<

$(GCC_TARGET): $(BINDIR) $(CPP_OBJECTS) $(COMMON_OBJECTS)
	@echo '[$(CC)] linking:' $@
	@$(CC) $(LDFLAGS) -o $@ $(CPP_OBJECTS) $(COMMON_OBJECTS)

$(NVCC_TARGET): $(BINDIR) $(CPP_OBJECTS) $(COMMON_OBJECTS) $(CUDA_OBJECTS)
	@echo '[$(CC)] linking:' $@
	@$(CC) $(LDFLAGS) -o $@ $(CPP_OBJECTS) $(COMMON_OBJECTS) $(CUDA_OBJECTS)

$(BINDIR):
	@echo 'mkdir:' $@
	@mkdir -p $@

.PHONY: clean
clean:
	@echo 'cleaning...'
	@rm -f $(CPP_OBJECTS)
	@rm -f $(COMMON_OBJECTS)
	@rm -f $(CUDA_OBJECTS)
	@rm -rf $(BINDIR)
