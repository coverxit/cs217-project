BINDIR = bin
EXECUTABLE = lzss

CC = g++
CFLAGS = -c -O0 -std=c++11 -g
LDFLAGS = -pthread

SOURCES = 							\
	main.cpp						\
	LZSSFactory.cpp 				\
	Helper/Helper.cpp				\
	CPUST/CPUSingleThreadLZSS.cpp

OBJECTS = $(patsubst %.cpp,%.o,$(SOURCES))
TARGET = $(BINDIR)/$(EXECUTABLE)

.PHONY: all
all: clean $(TARGET)

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
