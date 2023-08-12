# Default values, you can modify these as needed
BOOST := ${HOME}/boost_1_82_0/
EZINI := ${HOME}/ezini/

# Include directories
INCLUDES := -Iinclude/ -I$(BOOST) -I$(EZINI)

# Compiler and flags
CXX := g++
CXXFLAGS := -g -std=c++17
LDFLAGS := -lflexiblas -lmkl_rt -fopenmp -lquadmath

# Target
TARGET := octopode

# Source files
SRCS := src/main.cpp

# Object files
OBJS := $(SRCS:.cpp=.o) $(EZINI)/ezini.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)