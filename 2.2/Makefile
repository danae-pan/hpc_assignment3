TARGET= poisson
OBJS	= main.o \
          jacobi_offload.o \
		  jacobi.o \
          alloc3d.o \
          d_alloc3d.o \
		  warmup.o \
		  

CC	= nvc    # Use nvc for C files
CXX	= nvc++  # Use nvc++ for C++ files

OPT	= -g -fast -Msafeptr -Minfo=accel -acc -mp=gpu -gpu=mem:separate:pinnedalloc -gpu=cc90 -gpu=lineinfo -cuda -mp=noautopar
PARA	= -mp -fopenmp  # Enable OpenMP

CFLAGS = $(OPT) $(PARA)
CXXFLAGS = $(OPT) $(PARA)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

# Compile C++ files using nvc++
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C files using nvc (alloc3d.c specifically)
alloc3d.o: alloc3d.c alloc3d.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@/bin/rm -f *.o $(TARGET)