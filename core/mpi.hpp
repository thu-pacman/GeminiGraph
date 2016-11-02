/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef MPI_HPP
#define MPI_HPP

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

template <typename T>
MPI_Datatype get_mpi_data_type() {
  if (std::is_same<T, char>::value) {
    return MPI_CHAR;
  } else if (std::is_same<T, unsigned char>::value) {
    return MPI_UNSIGNED_CHAR;
  } else if (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if (std::is_same<T, unsigned>::value) {
    return MPI_UNSIGNED;
  } else if (std::is_same<T, long>::value) {
    return MPI_LONG;
  } else if (std::is_same<T, unsigned long>::value) {
    return MPI_UNSIGNED_LONG;
  } else if (std::is_same<T, float>::value) {
    return MPI_FLOAT;
  } else if (std::is_same<T, double>::value) {
    return MPI_DOUBLE;
  } else {
    printf("type not supported\n");
    exit(-1);
  }
}

class MPI_Instance {
  int partition_id;
  int partitions;
public:
  MPI_Instance(int * argc, char *** argv) {
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id);
    MPI_Comm_size(MPI_COMM_WORLD, &partitions);
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("thread support level provided by MPI: ");
      switch (provided) {
        case MPI_THREAD_MULTIPLE:
          printf("MPI_THREAD_MULTIPLE\n"); break;
        case MPI_THREAD_SERIALIZED:
          printf("MPI_THREAD_SERIALIZED\n"); break;
        case MPI_THREAD_FUNNELED:
          printf("MPI_THREAD_FUNNELED\n"); break;
        case MPI_THREAD_SINGLE:
          printf("MPI_THREAD_SINGLE\n"); break;
        default:
          assert(false);
      }
    }
    #endif
  }
  ~MPI_Instance() {
    MPI_Finalize();
  }
  void pause() {
    if (partition_id==0) {
      getchar();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
};

#endif
