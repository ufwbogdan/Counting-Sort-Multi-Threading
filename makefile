n=10000000
p=$(shell nproc)
k=500

run: compile
	bin/main $(n)

compile:
	@mkdir -p bin && rm -f bin/*
	g++ -O3 -Wall -DK=$(k) -DNPROC=$(p) -DLEVEL1_DCACHE_LINESIZE=`getconf LEVEL1_DCACHE_LINESIZE` -fopenmp countingsort.cpp -o bin/main

clean:
	@rm -rf bin
