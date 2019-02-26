BINDIR = bin
EXECUTABLE = lzss

GCC = g++
NVCC = nvcc

CFLAGS = -c -O3 -std=c++11 -g
GCC_LDFLAGS = -pthread
NVCC_LDFLAGS = -Xcompiler="-pthread"

TARGET = $(BINDIR)/$(EXECUTABLE)

CPP_SOURCES =						\
	main.cpp						\
	LZSSFactory.cpp					\
	CPU/ST/CPUSingleThreadLZSS.cpp	\
	CPU/MT/CPUMultiThreadLZSS.cpp

CU_SOURCES =						\
	MatchHelper/MatchHelper.cu		\
	CPU/BlockHelper.cu

CPP_OBJECTS = $(patsubst %.cpp,%.o,$(CPP_SOURCES))
CU_OBJECTS = $(patsubst %.cu,%.o,$(CU_SOURCES))

.PHONY: all
all:
	@echo "Specify a target (cpu or gpu)!"

.PHONY: cpu-cc
cpu-cc:
	$(eval CC = $(GCC))
	$(eval CFLAGS = -x c++ $(CFLAGS))
	$(eval LDFLAGS = $(GCC_LDFLAGS))

.PHONY: cpu
cpu: clean cpu-cc $(TARGET)

.PHONY: gpu-cc
gpu-cc:
	$(eval CC = $(NVCC))
	$(eval LDFLAGS = $(NVCC_LDFLAGS))

.PHONY: gpu
gpu: clean gpu-cc $(TARGET)
	
%.o: %.cpp
	@echo 'compiling cpp:' $<
	@$(CC) $(CFLAGS) -o $@ $<

%.o: %.cu
	@echo 'compiling cu:' $<
	@$(CC) $(CFLAGS) -o $@ $<

$(TARGET): $(BINDIR) $(CPP_OBJECTS) $(CU_OBJECTS)
	@echo 'linking:' $@
	@$(CC) $(LDFLAGS) -o $@ $(CPP_OBJECTS) $(CU_OBJECTS)

$(BINDIR):
	@echo 'mkdir:' $@
	@mkdir -p $@

.PHONY: clean
clean:
	@echo 'cleaning...'
	@rm -f $(CPP_OBJECTS)
	@rm -f $(CU_OBJECTS)
	@rm -rf $(BINDIR)
