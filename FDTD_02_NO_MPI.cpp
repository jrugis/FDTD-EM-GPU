/*

fdtd_02
An MPI+CUDA Finite-Difference-Time-Domain modelling and simulation program.

Copyright (c)2013 J.Rugis
Institute of Earth Science and Engineering
University of Auckland, New Zealand

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

j.rugis@auckland.ac.nz

notes:
1) The overall model space is a cube.
2) This overall space is divided into eight model blocks.
3) The eight model blocks are distrubuted, one each, to eight MPI nodes.
3) The program takes advantage of the uniform regular symmetry of the blocks.

*/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <time.h>
using namespace std;

#include "defs.h"
#include "model3d.h"

time_t start_time;
char host_name[80];

// get start time
void get_start_time(){
  start_time = time(NULL);
}

// show elapsed time
void show_elapsed_time(){
  time_t elapsed_time = time(NULL) - start_time;
  cout << "time: " << elapsed_time << endl;
}

// the main program
int main(int argc, char* argv[])
{
  CModel3D *model;

  if(argc < 2){
	cout << "Required: size_model & total_time_steps" << endl;
	exit(1);
  }
  int size_model = atoi(argv[1]); // overall model edge size, AN EVEN NUMBER!
  if(size_model == 0) {
	  size_model = 100;
	  cout << "Default size_model = " << size_model << endl;
  }
  int total_time_steps = atoi(argv[2]); // simulation time steps
  if(total_time_steps == 0) {
	  total_time_steps = 120;
	  cout << "Default total_time_steps = " << total_time_steps << endl;
  }
  bool use_cuda = false;
  if(strcmp(argv[4], "CUDA") == 0) use_cuda = true;
  bool use_2nd_cuda = false;
  if(strcmp(argv[5], "2ND_CUDA") == 0) use_2nd_cuda = true;
  bool save_final_field = false;
  if(strcmp(argv[6], "SAVE_FIELD") == 0) save_final_field = true;

  get_start_time();

  model = new CModel3D(0, size_model, use_cuda);

  // simulation time steps
  for(unsigned long t = 0; t < total_time_steps; t++){
    model->stepA();
    model->stepB();
  }

  if(save_final_field) model->save_data();
  delete model;
  show_elapsed_time();
  return 0;
}

