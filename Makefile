CXX = clang++
UNAME := $(shell uname)

muller_objs = muller.o oclFunctions.o helperFunctions.o
multest_objs = multest.o mullib.o armaFunctions.o naiveFunctions.o helperFunctions.o oclFunctions.o

ocl_lib = -framework OpenCL
ifeq ($(UNAME), Linux)
ocl_lib = -lOpenCL
endif

.PHONY: all clean

all: multest muller

%.o: %.cpp
	$(CXX) -c -o $@ $<

#multest: $(multest_objs)
#	$(CXX) -framework OpenCL -o $@ $^ -larmadillo
multest: $(multest_objs)
	$(CXX) $(ocl_lib) -o $@ $^ -larmadillo

muller: $(muller_objs)
	$(CXX) -framework OpenCL -o $@ $^

clean:
	-rm -f *.o multest muller
