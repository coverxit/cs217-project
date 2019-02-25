BINDIR = bin
EXECUTABLE = lzss

GCC = g++
NVCC = nvcc

CFLAGS = -c -O3 -std=c++11 -g
LDFLAGS = -pthread

SOURCES =							\
	main.cpp						\
	LZSSFactory.cpp					\
	Helper/Helper.cpp				\
	CPU/BlockHelper.cpp				\
	CPU/ST/CPUSingleThreadLZSS.cpp	\
	CPU/MT/CPUMultiThreadLZSS.cpp

OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
TARGET = $(BINDIR)/$(EXECUTABLE)

.PHONY: all
all:
	@echo "Specify a target (cpu or gpu)!"

.PHONY: cpu-cc
cpu-cc:
	$(eval CC = $(GCC))

.PHONY: cpu
cpu: cpu-cc clean $(TARGET)

.PHONY: gpu-cc
gpu-cc:
	$(eval CC = $(NVCC))

.PHONY: gpu
gpu: gpu-cc clean $(TARGET)

$(OBJECTS): %.o : %.cpp
	$(CC) $(CFLAGS) -o $@ $<

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS)

$(BINDIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -f $(OBJECTS)
	rm -rf $(BINDIR)
