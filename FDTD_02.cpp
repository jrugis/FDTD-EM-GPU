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
#include <iomanip>
#include <cstdlib>
#include <string>
#include <mpi.h>
using namespace std;

#include "defs.h"
#include "utils.h"
#include "model3d.h"

#define MPI_NODES 8 // mpi model blocks, DO NOT CHANGE

#define MPI_ERROR_ABORT -101
#define MPI_NODES_ABORT -102

char host_name[80];
int commSize;
int commRank = 0; // default to zero for non-MPI builds
int mesh[8][3] = // [node][face] sharing connectivity
  {{1,4,2},
   {0,5,3},
   {3,6,0},
   {2,7,1},
   {5,0,6},
   {4,1,7},
   {7,2,4},
   {6,3,5}};

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

// shutdown mpi on error
void mpi_abort(int err)
{
  cout << "MPI shutdown: " << err << endl;
  MPI_Abort(MPI_COMM_WORLD, err);
}

// mpi error checking macro
#define MPI_CHECK(call) \
  if((call) != MPI_SUCCESS) { \
    cout << "MPI error calling: " << call << endl; \
    mpi_abort(MPI_ERROR_ABORT); }

// initialize mpi
void mpi_init(int argc, char* argv[]){
  MPI_CHECK(MPI_Init(&argc, &argv));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));
  if(commSize != MPI_NODES) mpi_abort(MPI_NODES_ABORT); // check mpirun node count
  cout << commRank << "/" << commSize << " host: " << host_name;
}
 
// exchange shared face data between nodes
//  notes: 
//    1) non-blocking used to "bypass buffering"
//    2) implements two exchange types: low-to-high nodes & high-to-low nodes
void exchange_faces(CModel3D *model, bool lo2hi){
  MPI_Status mpi_status;
  MPI_Request mpi_request[3] = {NULL, NULL , NULL};
  for(int face = 0; face < 3; face++){ // receive up to three
    if(lo2hi && (commRank < mesh[commRank][face])) continue;  // lo2hi (don't receive from higher)
    if(!lo2hi && (commRank > mesh[commRank][face])) continue; // hi2lo (don't receive from lower) 
    MPI_CHECK(MPI_Irecv( 
      model->share_in_face[face], model->size_face, MPI_F_TYPE,
      mesh[commRank][face], 0, MPI_COMM_WORLD, &mpi_request[face]));
  }
  for(int face = 0; face < 3; face++){ // send up to three
    if(lo2hi && (commRank > mesh[commRank][face])) continue;  // lo2hi (don't send to lower)
    if(!lo2hi && (commRank < mesh[commRank][face])) continue; // hi2lo (don't send to higher) 
    MPI_CHECK(MPI_Send(  
      model->share_out_face[face], model->size_face, MPI_F_TYPE,
      mesh[commRank][face], 0, MPI_COMM_WORLD));
  }
  for(int face = 0; face < 3; face++) // wait until all received
    if(mpi_request[face] != NULL) MPI_CHECK(MPI_Wait(&mpi_request[face], &mpi_status));
}

// shutdown mpi
void mpi_shutdown(){
  MPI_CHECK(MPI_Finalize());
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

  if(use_mpi){
    size_model /= 2;
    mpi_init(argc, argv);
  }
  else cout << commRank << " host: " << host_name;

  model = new CModel3D(commRank, size_model, settings);
  cout << commRank << "          time(sec) setup: " << item_timer.get_elapsed_time() << endl;

  // simulation time steps
  step_time_0 = step_time_1 = step_time_2 = step_time_3 = 0.0;
  for(unsigned long t = 0; t < total_time_steps; t++){
    model->stepA();
    step_time_0 += item_timer.get_elapsed_time();
    if(use_mpi) exchange_faces(model, true); // send & receive h-field
    step_time_1 += item_timer.get_elapsed_time();
    model->stepB();
    step_time_2 += item_timer.get_elapsed_time();
    if(use_mpi) exchange_faces(model, false);// send & receive e-field
    step_time_3 += item_timer.get_elapsed_time();
  }
  step_time_0 /= total_time_steps;
  step_time_1 /= total_time_steps;
  step_time_2 /= total_time_steps;
  step_time_3 /= total_time_steps;
  step_time = step_time_0 + step_time_1 + step_time_2 + step_time_3;
  cout << commRank << "           time(sec) step: " << step_time
    << " A(" << step_time_0 << ") MPI(" << step_time_1 << ") B(" << step_time_2 << ") MPI(" << step_time_3 << ")" << endl;

  if(save_final_field) model->save_data();

  delete model;

  if(use_mpi) mpi_shutdown();

  cout << commRank << "         time(sec) finish: " << item_timer.get_elapsed_time() << endl;
  cout << commRank << "          time(sec) total: " << main_timer.get_elapsed_time() << endl;
  return 0;
}

