# Makefile是给make命令执行时会读取的批处理脚本
# 不具有跨平台的特性，如果需要跨平台，可以使用cmake，通过编写CMakeLists.txt
# 来实现跨平台或者导出为某个平台特定的Makefile

ROOT_DIR= $(shell pwd)
TARGETS= toolkits/bc toolkits/bfs toolkits/cc toolkits/pagerank toolkits/sssp toolkits/convert_csv_to_binary
MACROS= 
# MACROS= -D PRINT_DEBUG_MESSAGES

MPICXX= mpicxx
CXXFLAGS= -O3 -Wall -std=c++11 -g -fopenmp -march=native -I$(ROOT_DIR) $(MACROS)
SYSLIBS= -lnuma
HEADERS= $(shell find . -name '*.hpp')

all: $(TARGETS)

toolkits/%: toolkits/%.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

clean: 
	rm -f $(TARGETS)

