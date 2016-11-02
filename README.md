# Gemini
A computation-centric distributed graph processing system.

## Quick Start
Gemini uses **MPI** for inter-process communication and **libnuma** for NUMA-aware memory allocation.
A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

Implementations of five graph analytics applications (PageRank, Connected Components, Single-Source Shortest Paths, Breadth-First Search, Betweenness Centrality) are inclulded in the *toolkits/* directory.

To build:
```
make
```

The input parameters of these applications are as follows:
```
./toolkits/pagerank [path] [vertices] [iterations]
./toolkits/cc [path] [vertices]
./toolkits/sssp [path] [vertices] [root]
./toolkits/bfs [path] [vertices] [root]
./toolkits/bc [path] [vertices] [root]
```

*[path]* gives the path of an input graph, i.e. a file stored on a *shared* file system, consisting of *|E|* \<source vertex id, destination vertex id, edge data\> tuples in binary.
*[vertices]* gives the number of vertices *|V|*. Vertex IDs are represented with 32-bit integers and edge data can be omitted for unweighted graphs (e.g. the above applications except SSSP).
Note: CC makes the input graph undirected by adding a reversed edge to the graph for each loaded one; SSSP uses *float* as the type of weights.

If Slurm is installed on the cluster, you may run jobs like this, e.g. 20 iterations of PageRank on the *twitter-2010* graph:
```
srun -N 8 ./toolkits/pagerank /path/to/twitter-2010.binedgelist 41652230 20
```

## Resources

Xiaowei Zhu, Wenguang Chen, Weimin Zheng, and Xiaosong Ma.
Gemini: A Computation-Centric Distributed Graph Processing System.
12th USENIX Symposium on Operating Systems Design and Implementation (OSDI '16).

