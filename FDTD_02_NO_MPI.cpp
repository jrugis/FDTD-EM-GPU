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
3) The eight model blocks are distributed, one each, to eight MPI nodes.
3) The program takes advantage of the uniform regular symmetry of the blocks.

*/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include<stdio.h>
using namespace std;

#include "defs.h"
#include "utils.h"
#include "model3d.h"

char host_name[80];
int commSize;
int commRank = 0; // default to zero for non-MPI builds

// get host name
void get_host_name(){
  FILE *in;
  in = popen("hostname", "r");
  fgets(host_name, sizeof(host_name), in);
  pclose(in);
}

// get settings
void get_settings(int argc, char* argv[], CSettings *settings)
{
  if(argc < 8){
	cout << "Useage:";
	cout << " fdtd_02 model_size_x_y_z time_steps op_code";
	cout << " cuda_blocks_x_y cuda_blocks_z cuda_threads_x_y_z cuda_gap" << endl;
	exit(1);
  }
  settings->size = atoi(argv[1]);         // overall model edge size, AN EVEN NUMBER!
  settings->steps = atoi(argv[2]);        // simulation time steps
  settings->op_code = atoi(argv[3]);      // settings op-code
  settings->cuda_bx = atoi(argv[4]);      // cuda blocks x
  settings->cuda_by = settings->cuda_bx;  // cuda blocks y
  settings->cuda_bz = atoi(argv[5]);    // cuda blocks z
  settings->cuda_tx = atoi(argv[6]);      // cuda threads x
  settings->cuda_ty = settings->cuda_tx;  // cuda threads y
  settings->cuda_tz = settings->cuda_ty;  // cuda threads y
  settings->cuda_gap = atoi(argv[7]);     // cuda a gap

  settings->cuda_x = (settings->cuda_tx * settings->cuda_bx) - 2;
  settings->cuda_y = (settings->cuda_ty * settings->cuda_by) - 2;
  settings->cuda_z = (settings->cuda_tz * settings->cuda_bz) - 2;

  int temp;
  temp = settings->size - settings->cuda_x;
  if(temp < 4) {cout << "ERROR: cuda_bx too big." << endl; exit(1);}
  temp = settings->size - settings->cuda_y;
  if(temp < 4) {cout << "ERROR: cuda_by too big." << endl; exit(1);}
  temp = settings->size - (2 * settings->cuda_z) - settings->cuda_gap;
  if(temp < 4) {cout << "ERROR: cuda_bz too big." << endl; exit(1);}
}


// the main program function for each mpi node
int main(int argc, char* argv[])
{
  CModel3D *model;
  CSettings settings;
  CTimer main_timer, item_timer;
  double step_time, step_time_0, step_time_1, step_time_2, step_time_3;

  cout.precision(2);
  cout << scientific;

  get_settings(argc, argv, &settings);
  get_host_name();  

  size_t size_model = settings.size;
  size_t total_time_steps = settings.steps;
  bool save_final_field = settings.op_code & SAVE;
  bool use_mpi = settings.op_code & MPI;

  cout << commRank << " host: " << host_name;

  model = new CModel3D(commRank, size_model, settings);
  cout << commRank << "          time(sec) setup: " << item_timer.get_elapsed_time() << endl;

  // simulation time steps
  step_time_0 = step_time_1 = step_time_2 = step_time_3 = 0.0;
  for(unsigned long t = 0; t < total_time_steps; t++){
    model->stepA();
    step_time_0 += item_timer.get_elapsed_time();
    model->stepB();
    step_time_1 += item_timer.get_elapsed_time();
  }
  step_time_0 /= total_time_steps;
  step_time_1 /= total_time_steps;
  step_time = step_time_0 + step_time_1;
  cout << commRank << "           time(sec) step: " << step_time
    << " A(" << step_time_0 << ") B(" << step_time_1 << ")" << endl;

  if(save_final_field) model->save_data();

  delete model;

  cout << commRank << "         time(sec) finish: " << item_timer.get_elapsed_time() << endl;
  cout << commRank << "          time(sec) total: " << main_timer.get_elapsed_time() << endl;
  return 0;
}

