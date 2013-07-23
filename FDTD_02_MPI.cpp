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
#include <string>
#include <time.h>
#include <mpi.h>
using namespace std;

#include "defs.h"
#include "model3d.h"

#define MPI_NODES 8 // mpi model blocks, DO NOT CHANGE

#define MPI_ERROR_ABORT -101
#define MPI_NODES_ABORT -102

time_t start_time;
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

// get start time
void get_start_time(){
  start_time = time(NULL);
} 

// show elapsed time
void show_elapsed_time(){
  time_t elapsed_time = time(NULL) - start_time;
  cout << "rank/time: " << commRank << "/" << elapsed_time << endl;
}

// get host name
void get_host_name(){
  FILE *in;
  in = popen("hostname", "r");
  fgets(host_name, sizeof(host_name), in);
  pclose(in);
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
  cout << "host: " << host_name << "rank/size: " << commRank << "/" << commSize << endl;
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
  bool use_mpi = false;
  if(strcmp(argv[3], "MPI") == 0) use_mpi = true;
  bool use_cuda = false;
  if(strcmp(argv[4], "CUDA") == 0) use_cuda = true;
  bool use_2nd_cuda = false;
  if(strcmp(argv[5], "2ND_CUDA") == 0) use_2nd_cuda = true;
  bool save_final_field = false;
  if(strcmp(argv[6], "SAVE_FIELD") == 0) save_final_field = true;

  get_start_time();
  get_host_name();  

  if(use_mpi){
    size_model /= 2;
    mpi_init(argc, argv);
  }
  else cout << "host: " << host_name << endl;

  model = new CModel3D(commRank, size_model, use_cuda);

  // simulation time steps
  for(unsigned long t = 0; t < total_time_steps; t++){
    model->stepA();
    if(use_mpi) exchange_faces(model, true); // send & receive h-field
    model->stepB();
    if(use_mpi) exchange_faces(model, false);// send & receive e-field
  }

  if(save_final_field) model->save_data();

  delete model;

  if(use_mpi) mpi_shutdown();

  show_elapsed_time();
  return 0;
}

