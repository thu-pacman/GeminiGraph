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

#ifndef ATOMIC_HPP
#define ATOMIC_HPP

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

template <class T>
inline bool cas(T * ptr, T old_val, T new_val) {
  if (sizeof(T) == 8) {
    return __sync_bool_compare_and_swap((long*)ptr, *((long*)&old_val), *((long*)&new_val));
  } else if (sizeof(T) == 4) {
    return __sync_bool_compare_and_swap((int*)ptr, *((int*)&old_val), *((int*)&new_val));
  } else {
    assert(false);
  }
}

template <class T>
inline bool write_min(T * ptr, T val) {
  volatile T curr_val; bool done = false;
  do {
    curr_val = *ptr;
  } while (curr_val > val && !(done = cas(ptr, curr_val, val)));
  return done;
}

template <class T>
inline void write_add(T * ptr, T val) {
  volatile T new_val, old_val;
  do {
    old_val = *ptr;
    new_val = old_val + val;
  } while (!cas(ptr, old_val, new_val));
}

#endif
