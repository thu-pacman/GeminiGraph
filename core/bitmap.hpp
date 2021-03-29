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

// 自己实现了一套位图
#include <stddef.h>
#ifndef BITMAP_HPP
#define BITMAP_HPP

// 位图，本身是使用二进制位来压缩数据的一种方式，在编程里面，没有直接表示二进制位的数据(boolean?)
// 所以，往往使用32位的int，64位的long等类型的数组来进行间接表示，即封装了多个二进制位
// 当需要定位索引为 x
// 的二进制位的位置时，要先使用整数除法定位x在哪个数值上，然后使用余数
// 定位在对应数值上的具体位置，比如定义索引号为2237的二进制位在以int32数组array上的位置时
// 其对应的可表示数值为 2237 / 32 = 69，即在array[69]上，接下来定位二进制位
//  2237 % 32 = 29，即对应的二进制位在arr[69]的整数的29位上。
// 
#define WORD_OFFSET(i) ((i) >> 6) // 除以64，其实一个long放64个二进制位
#define BIT_OFFSET(i) ((i)&0x3f) // 和0011 1111相与，其实为和64取余，找到其二进制位的偏移

/**
 * 用unsigned long数组存放位图，实现时使用指针 + size表示数组
 *
 */
class Bitmap {
public:
  size_t size;
  unsigned long *data;

  // 初始化列表语法
  Bitmap() : size(0), data(NULL) {}  
  Bitmap(size_t size) : size(size) {
    data = new unsigned long[WORD_OFFSET(size) + 1];
    clear();
  }

  // 析构，清理data数组
  ~Bitmap() { delete[] data; }

  /**
   * 数组所有元素置为0
   */
  void clear() {
    size_t bm_size = WORD_OFFSET(size);
#pragma omp parallel for
    for (size_t i = 0; i <= bm_size; i++) {
      data[i] = 0;
    }
  }

  /**
   * 数组所有元素置为0xffffffffffffffff，即64位全为1
   */
  void fill() {
    size_t bm_size = WORD_OFFSET(size);
#pragma omp parallel for
    for (size_t i = 0; i < bm_size; i++) {
      data[i] = 0xffffffffffffffff;
    }
    data[bm_size] = 0;
    for (size_t i = (bm_size << 6); i < size; i++) {
      data[bm_size] |= 1ul << BIT_OFFSET(i);
    }
  }

  /**
   * 获取对应的二进制位的值，返回0代表该位为0，否则1
   * @param i 二进制位的index值
   */
  unsigned long get_bit(size_t i) {
    return data[WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
  }
  
  /**
   * 设置对应二进制位的值为1，使用cas来实现原子更新
   * @param i 二进制位的index值
   */  
  void set_bit(size_t i) {
    __sync_fetch_and_or(data + WORD_OFFSET(i), 1ul << BIT_OFFSET(i));
  }
};

// 没有必要放在这里，既然有一个统一的type.hpp来放置定义
// typedef Bitmap VertexSubset;
#endif
