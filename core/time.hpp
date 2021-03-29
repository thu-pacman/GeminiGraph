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

#ifndef TIME_HPP
#define TIME_HPP

#include <sys/time.h>

/**
 * 获取当前时间
 * tv_sec为从1970-1-1零点零分到创建struct timeval时的秒数，tv_usec为微秒数
 */
inline double get_time() {
  struct timeval tv;
  // C++ 11里面并不推荐使用NULL来表示空指针，最好是使用nullptr
  // gettimeofday(&tv, NULL);
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + (tv.tv_usec / 1e6);
}

#endif
