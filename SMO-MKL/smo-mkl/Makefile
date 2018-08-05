CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -U_FORTIFY_SOURCE
LDFLAGS = -lrt
SHVER = 1

all: svm-train svm-predict svm-scale

lib: svm.o
	$(CXX) -shared svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o 
	$(CXX) $(CFLAGS) $(LDFLAGS) svm-predict.c svm.o timer.o -o svm-predict -lm
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) $(LDFLAGS) svm-train.c svm.o timer.o -o svm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) $(LDFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h timer.o
	$(CXX) $(CFLAGS) -c svm.cpp
timer.o: timer.cpp timer.hpp
	$(CXX) $(CFLAGS) -c timer.cpp
clean:
	rm -f *~ svm.o timer.o svm-train svm-predict svm-scale
