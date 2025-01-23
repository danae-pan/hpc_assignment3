TARGET= libmatmult.so
OBJS	=  matmult.o

CC	= nvc
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -mp=gpu -gpu=mem:separate:pinnedalloc -gpu=lineinfo -gpu=cc90 -cuda -mp=noautopar -cudalib=cublas
PIC   = -fpic -shared
ISA	=
PARA	= -fopenmp
INC = 

LIBS = -L/appl9/nvhpc/2024_2411/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux/lib -lcublas -lcublasLt


CXXFLAGS= $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	@/bin/rm -f $(TARGET) $(OBJS)
