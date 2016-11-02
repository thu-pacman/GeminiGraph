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

typedef float Weight;

void compute(Graph<Weight> * graph, VertexId root) {
  double exec_time = 0;
  exec_time -= get_time();

  Weight * distance = graph->alloc_vertex_array<Weight>();
  VertexSubset * active_in = graph->alloc_vertex_subset();
  VertexSubset * active_out = graph->alloc_vertex_subset();
  active_in->clear();
  active_in->set_bit(root);
  graph->fill_vertex_array(distance, (Weight)1e9);
  distance[root] = (Weight)0;
  VertexId active_vertices = 1;
  
  for (int i_i=0;active_vertices>0;i_i++) {
    if (graph->partition_id==0) {
      printf("active(%d)>=%u\n", i_i, active_vertices);
    }
    active_out->clear();
    active_vertices = graph->process_edges<VertexId,Weight>(
      [&](VertexId src){
        graph->emit(src, distance[src]);
      },
      [&](VertexId src, Weight msg, VertexAdjList<Weight> outgoing_adj){
        VertexId activated = 0;
        for (AdjUnit<Weight> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          Weight relax_dist = msg + ptr->edge_data;
          if (relax_dist < distance[dst]) {
            if (write_min(&distance[dst], relax_dist)) {
              active_out->set_bit(dst);
              activated += 1;
            }
          }
        }
        return activated;
      },
      [&](VertexId dst, VertexAdjList<Weight> incoming_adj) {
        Weight msg = 1e9;
        for (AdjUnit<Weight> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          // if (active_in->get_bit(src)) {
            Weight relax_dist = distance[src] + ptr->edge_data;
            if (relax_dist < msg) {
              msg = relax_dist;
            }
          // }
        }
        if (msg < 1e9) graph->emit(dst, msg);
      },
      [&](VertexId dst, Weight msg) {
        if (msg < distance[dst]) {
          write_min(&distance[dst], msg);
          active_out->set_bit(dst);
          return 1;
        }
        return 0;
      },
      active_in
    );
    std::swap(active_in, active_out);
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  graph->gather_vertex_array(distance, 0);
  if (graph->partition_id==0) {
    VertexId max_v_i = root;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (distance[v_i] < 1e9 && distance[v_i] > distance[max_v_i]) {
        max_v_i = v_i;
      }
    }
    printf("distance[%u]=%f\n", max_v_i, distance[max_v_i]);
  }

  graph->dealloc_vertex_array(distance);
  delete active_in;
  delete active_out;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("sssp [file] [vertices] [root]\n");
    exit(-1);
  }

  Graph<Weight> * graph;
  graph = new Graph<Weight>();
  graph->load_directed(argv[1], std::atoi(argv[2]));
  VertexId root = std::atoi(argv[3]);

  compute(graph, root);
  for (int run=0;run<5;run++) {
    compute(graph, root);
  }

  delete graph;
  return 0;
}
