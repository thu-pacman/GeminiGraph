/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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

#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

#define COMPACT 0

void compute(Graph<Empty> * graph, VertexId root) {
  double exec_time = 0;
  exec_time -= get_time();

  double * num_paths = graph->alloc_vertex_array<double>();
  double * dependencies = graph->alloc_vertex_array<double>();
  VertexSubset * active_all = graph->alloc_vertex_subset();
  active_all->fill();
  VertexSubset * visited = graph->alloc_vertex_subset();
  std::vector<VertexSubset *> levels;
  VertexSubset * active_in = graph->alloc_vertex_subset();

  VertexId active_vertices = 1;
  visited->clear();
  visited->set_bit(root);
  active_in->clear();
  active_in->set_bit(root);
  levels.push_back(active_in);
  graph->fill_vertex_array(num_paths, 0.0);
  num_paths[root] = 1.0;
  VertexId i_i;
  if (graph->partition_id==0) {
    printf("forward\n");
  }
  for (i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    VertexSubset * active_out = graph->alloc_vertex_subset();
    active_out->clear();
    graph->process_edges<VertexId,double>(
      [&](VertexId src){
        graph->emit(src, num_paths[src]);
      },
      [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (!visited->get_bit(dst)) {
            if (num_paths[dst]==0) {
              active_out->set_bit(dst);
            }
            write_add(&num_paths[dst], msg);
          }
        }
        return 0;
      },
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        if (visited->get_bit(dst)) return;
        double sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_in->get_bit(src)) {
            sum += num_paths[src];
          }
        }
        if (sum > 0) {
          graph->emit(dst, sum);
        }
      },
      [&](VertexId dst, double msg) {
        if (!visited->get_bit(dst)) {
          active_out->set_bit(dst);
          write_add(&num_paths[dst], msg);
        }
        return 0;
      },
      active_in, visited
    );
    active_vertices = graph->process_vertices<VertexId>(
      [&](VertexId vtx) {
        visited->set_bit(vtx);
        return 1;
      },
      active_out
    );
    levels.push_back(active_out);
    active_in = active_out;
  }

  double * inv_num_paths = num_paths;
  graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      inv_num_paths[vtx] = 1 / num_paths[vtx];
      dependencies[vtx] = 0;
      return 1;
    },
    active_all
  );
  visited->clear();
  graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      visited->set_bit(vtx);
      dependencies[vtx] += inv_num_paths[vtx];
      return 1;
    },
    levels.back()
  );
  graph->transpose();
  if (graph->partition_id==0) {
    printf("backward\n");
  }
  while (levels.size() > 1) {
    graph->process_edges<VertexId,double>(
      [&](VertexId src){
        graph->emit(src, dependencies[src]);
      },
      [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (!visited->get_bit(dst)) {
            write_add(&dependencies[dst], msg);
          }
        }
        return 0;
      },
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        if (visited->get_bit(dst)) return;
        double sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (levels.back()->get_bit(src)) {
            sum += dependencies[src];
          }
        }
        graph->emit(dst, sum);
      },
      [&](VertexId dst, double msg) {
        if (!visited->get_bit(dst)) {
          write_add(&dependencies[dst], msg);
        }
        return 0;
      },
      levels.back(), visited
    );
    delete levels.back();
    levels.pop_back();
    graph->process_vertices<VertexId>(
      [&](VertexId vtx){
        visited->set_bit(vtx);
        dependencies[vtx] += inv_num_paths[vtx];
        return 1;
      },
      levels.back()
    );
  }

  graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      dependencies[vtx] = (dependencies[vtx] - inv_num_paths[vtx]) / inv_num_paths[vtx];
      return 1;
    },
    active_all
  );
  graph->transpose();

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(dependencies, 0);
  graph->gather_vertex_array(inv_num_paths, 0);
  if (graph->partition_id==0) {
    for (VertexId v_i=0;v_i<20;v_i++) {
      printf("%lf %lf\n", dependencies[v_i], 1 / inv_num_paths[v_i]);
    }
  }

  graph->dealloc_vertex_array(dependencies);
  graph->dealloc_vertex_array(inv_num_paths);
  delete visited;
  delete active_all;
}

// an implementation which uses an array to store the levels instead of multiple bitmaps
void compute_compact(Graph<Empty> * graph, VertexId root) {
  double exec_time = 0;
  exec_time -= get_time();

  double * num_paths = graph->alloc_vertex_array<double>();
  double * dependencies = graph->alloc_vertex_array<double>();
  VertexSubset * active_all = graph->alloc_vertex_subset();
  active_all->fill();
  VertexSubset * visited = graph->alloc_vertex_subset();
  VertexId * level = graph->alloc_vertex_array<VertexId>();
  VertexSubset * active_in = graph->alloc_vertex_subset();
  VertexSubset * active_out = graph->alloc_vertex_subset();

  visited->clear();
  visited->set_bit(root);
  active_in->clear();
  active_in->set_bit(root);
  VertexId active_vertices = graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      if (active_in->get_bit(vtx)) {
        level[vtx] = 0;
        return 1;
      } else {
        level[vtx] = graph->vertices;
        return 0;
      }
    },
    active_all
  );
  graph->fill_vertex_array(num_paths, 0.0);
  num_paths[root] = 1.0;
  VertexId i_i;
  if (graph->partition_id==0) {
    printf("forward\n");
  }
  for (i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_out->clear();
    graph->process_edges<VertexId,double>(
      [&](VertexId src){
        graph->emit(src, num_paths[src]);
      },
      [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (!visited->get_bit(dst)) {
            if (num_paths[dst]==0) {
              active_out->set_bit(dst);
            }
            write_add(&num_paths[dst], msg);
          }
        }
        return 0;
      },
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        if (visited->get_bit(dst)) return;
        double sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_in->get_bit(src)) {
            sum += num_paths[src];
          }
        }
        if (sum > 0) {
          graph->emit(dst, sum);
        }
      },
      [&](VertexId dst, double msg) {
        if (!visited->get_bit(dst)) {
          active_out->set_bit(dst);
          write_add(&num_paths[dst], msg);
        }
        return 0;
      },
      active_in, visited
    );
    active_vertices = graph->process_vertices<VertexId>(
      [&](VertexId vtx) {
        visited->set_bit(vtx);
        level[vtx] = i_i + 1;
        return 1;
      },
      active_out
    );
    std::swap(active_in, active_out);
  }

  double * inv_num_paths = num_paths;
  graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      inv_num_paths[vtx] = 1 / num_paths[vtx];
      dependencies[vtx] = 0;
      return 1;
    },
    active_all
  );
  visited->clear();
  active_in->clear();
  graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      if (level[vtx]==i_i) {
        active_in->set_bit(vtx);
        return 1;
      }
      return 0;
    },
    active_all
  );
  graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      visited->set_bit(vtx);
      dependencies[vtx] += inv_num_paths[vtx];
      return 1;
    },
    active_in
  );
  graph->transpose();
  if (graph->partition_id==0) {
    printf("backward\n");
  }
  while (i_i > 0) {
    graph->process_edges<VertexId,double>(
      [&](VertexId src){
        graph->emit(src, dependencies[src]);
      },
      [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          if (!visited->get_bit(dst)) {
            write_add(&dependencies[dst], msg);
          }
        }
        return 0;
      },
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        if (visited->get_bit(dst)) return;
        double sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          if (active_in->get_bit(src)) {
            sum += dependencies[src];
          }
        }
        graph->emit(dst, sum);
      },
      [&](VertexId dst, double msg) {
        if (!visited->get_bit(dst)) {
          write_add(&dependencies[dst], msg);
        }
        return 0;
      },
      active_in, visited
    );
    i_i--;
    active_in->clear();
    active_vertices = graph->process_vertices<VertexId>(
      [&](VertexId vtx){
        if (level[vtx]==i_i) {
          active_in->set_bit(vtx);
          return 1;
        }
        return 0;
      },
      active_all
    );
    graph->process_vertices<VertexId>(
      [&](VertexId vtx){
        visited->set_bit(vtx);
        dependencies[vtx] += inv_num_paths[vtx];
        return 1;
      },
      active_in
    );
  }

  graph->process_vertices<VertexId>(
    [&](VertexId vtx){
      dependencies[vtx] = (dependencies[vtx] - inv_num_paths[vtx]) / inv_num_paths[vtx];
      return 1;
    },
    active_all
  );
  graph->transpose();

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(dependencies, 0);
  graph->gather_vertex_array(inv_num_paths, 0);
  if (graph->partition_id==0) {
    for (VertexId v_i=0;v_i<20;v_i++) {
      printf("%lf %lf\n", dependencies[v_i], 1 / inv_num_paths[v_i]);
    }
  }

  graph->dealloc_vertex_array(dependencies);
  graph->dealloc_vertex_array(inv_num_paths);
  delete visited;
  delete active_all;
  delete active_in;
  delete active_out;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("bc [file] [vertices] [root]\n");
    exit(-1);
  }

  Graph<Empty> * graph;
  graph = new Graph<Empty>();
  VertexId root = std::atoi(argv[3]);
  graph->load_directed(argv[1], std::atoi(argv[2]));

  #if COMPACT
  compute_compact(graph, root);
  #else
  compute(graph, root);
  #endif
  for (int run=0;run<5;run++) {
    #if COMPACT
    compute_compact(graph, root);
    #else
    compute(graph, root);
    #endif
  }

  delete graph;
  return 0;
}
