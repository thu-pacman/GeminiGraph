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

// core/type.hpp定义了点，边ID以及邻接关系等
#ifndef TYPE_HPP
#define TYPE_HPP

#include <stdint.h>
#include "bitmap.hpp"

struct Empty { };
typedef Bitmap VertexSubset;

// 使用uint32_t之类的指定固定大小无符号整数类型
// 提供良好的移植性（同为int，但是不同平台下int大小可能不同，可能为2字节，4字节等）
typedef uint32_t VertexId;
typedef uint64_t EdgeId;

// 这里使用了模板，EdgeData只是代表抽象的边数据
// 比如最短路径里面边的权重，就是这里的EdgeData
// 类似Java里面的泛型
template <typename EdgeData>
struct EdgeUnit {
  VertexId src;
  VertexId dst;
  EdgeData edge_data;
} __attribute__((packed));

// 代表没有数据的边，比如PageRank的输入数据的边就只有起点终点，并没有数据
template <>
struct EdgeUnit <Empty> {
  VertexId src;
  union {
    VertexId dst;
    Empty edge_data;
  };
} __attribute__((packed));

// 代表一条邻接边关系，包含边的另一端的顶点id和边自身含有的数据
template <typename EdgeData>
struct AdjUnit {
  VertexId neighbour;
  EdgeData edge_data;
} __attribute__((packed));

// 代表一条邻接关系，只有邻居顶点id，但是邻接边没有数据属性，比如PageRank的输入数据
template <>
struct AdjUnit <Empty> {
  union {
    VertexId neighbour;
    Empty edge_data;
  };
} __attribute__((packed));

// 压缩的邻接索引单元，这里应该是要和其他的矩阵之类的结合理解
struct CompressedAdjIndexUnit {
  EdgeId index;
  VertexId vertex;
} __attribute__((packed));

// 顶点的邻接链表，即所有的邻接边关系
// 使用两个指针代表链表的起点和终点
// 使用初始化列表语法进行数据成员的初始化
template <typename EdgeData>
struct VertexAdjList {
  AdjUnit<EdgeData> * begin;
  AdjUnit<EdgeData> * end;
  VertexAdjList() : begin(nullptr), end(nullptr) { }
  VertexAdjList(AdjUnit<EdgeData> * begin, AdjUnit<EdgeData> * end) : begin(begin), end(end) { }
};
#endif
