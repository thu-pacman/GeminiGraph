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

// 定义了用到的常量
#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// CHUNKSIZE 表示划分的大小
// 1 << 20 => 1048576，即百万级别
#define CHUNKSIZE (1 << 20)

// PAGESIZE 表示单页大小
// 1 << 12 => 4096
#define PAGESIZE (1<<12)

#endif
