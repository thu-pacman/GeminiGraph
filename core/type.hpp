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

#ifndef TYPE_HPP
#define TYPE_HPP

#include <stdint.h>

struct Empty { };

typedef uint32_t VertexId;
typedef uint64_t EdgeId;

template <typename EdgeData>
struct EdgeUnit {
  VertexId src;
  VertexId dst;
  EdgeData edge_data;
} __attribute__((packed));

template <>
struct EdgeUnit <Empty> {
  VertexId src;
  union {
    VertexId dst;
    Empty edge_data;
  };
} __attribute__((packed));

template <typename EdgeData>
struct AdjUnit {
  VertexId neighbour;
  EdgeData edge_data;
} __attribute__((packed));

template <>
struct AdjUnit <Empty> {
  union {
    VertexId neighbour;
    Empty edge_data;
  };
} __attribute__((packed));

struct CompressedAdjIndexUnit {
  EdgeId index;
  VertexId vertex;
} __attribute__((packed));

template <typename EdgeData>
struct VertexAdjList {
  AdjUnit<EdgeData> * begin;
  AdjUnit<EdgeData> * end;
  VertexAdjList() : begin(nullptr), end(nullptr) { }
  VertexAdjList(AdjUnit<EdgeData> * begin, AdjUnit<EdgeData> * end) : begin(begin), end(end) { }
};

#endif
