- Counting sort is an algorithm for sorting values belonging to a specific range [0, k). The file "countingsort.cpp" contains a sequential version, a naive parallel version and a correct parallel solution that minimizes the effect of the false sharing problem.
- It uses proper memory alignment using and computing the minimal padding value required to avoid different processors accessing the same cache-lines for writing.
- In modern multiprocessor CPU caches, where memory is cached in lines of small power of two size, the problem relies where two processors operate on independent data in the same memory address region, the cache coherency mechanisms in the system may force the
  whole line across the bus or interconnect with every data write, forcing memory stalls in addition to wasting system bandwith. 
- The solution also provides a makefile for easy testing.
