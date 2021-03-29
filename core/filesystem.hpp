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

#ifndef FILESYSTEM_HPP
#define FILESYSTEM_HPP

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>

// 这里就是简单检查文件是否存在以及文件的大小
// 注意输入文件是二进制格式的，没有空隙，
// 所以这里获取的文件大小才能用于后面的计算
inline bool file_exists(std::string filename) {
  struct stat st;
  return stat(filename.c_str(), &st)==0;
}

inline long file_size(std::string filename) {
  struct stat st;
  assert(stat(filename.c_str(), &st)==0);
  return st.st_size;
}

#endif
