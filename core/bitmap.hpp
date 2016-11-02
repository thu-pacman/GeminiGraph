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

#ifndef BITMAP_HPP
#define BITMAP_HPP

#define WORD_OFFSET(i) ((i) >> 6)
#define BIT_OFFSET(i) ((i) & 0x3f)

class Bitmap {
public:
  size_t size;
  unsigned long * data;
  Bitmap() : size(0), data(NULL) { }
  Bitmap(size_t size) : size(size) {
    data = new unsigned long [WORD_OFFSET(size)+1];
    clear();
  }
  ~Bitmap() {
    delete [] data;
  }
  void clear() {
    size_t bm_size = WORD_OFFSET(size);
    #pragma omp parallel for
    for (size_t i=0;i<=bm_size;i++) {
      data[i] = 0;
    }
  }
  void fill() {
    size_t bm_size = WORD_OFFSET(size);
    #pragma omp parallel for
    for (size_t i=0;i<bm_size;i++) {
      data[i] = 0xffffffffffffffff;
    }
    data[bm_size] = 0;
    for (size_t i=(bm_size<<6);i<size;i++) {
      data[bm_size] |= 1ul << BIT_OFFSET(i);
    }
  }
  unsigned long get_bit(size_t i) {
    return data[WORD_OFFSET(i)] & (1ul<<BIT_OFFSET(i));
  }
  void set_bit(size_t i) {
    __sync_fetch_and_or(data+WORD_OFFSET(i), 1ul<<BIT_OFFSET(i));
  }
};

typedef Bitmap VertexSubset;

#endif
