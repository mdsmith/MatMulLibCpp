muller: muller.cpp
	clang++ -framework OpenCL -o $@ $< -larmadillo
