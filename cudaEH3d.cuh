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

*/

#ifndef CUDAEH3D_H
#define CUDAEH3D_H

#include <stdio.h>

#include "defs.h"

class CCudaEH3d
{
public:
  CCudaEH3d(int rank, int device, CSettings settings);
  ~CCudaEH3d();

  size_t sX, sY, sZ, sXY, sXYZ;  // stride sizes
  size_t fsX, fsY, fsZ;          // face sizes
  F_TYPE *h_buf;            // host field or material buffer pointer
  F_TYPE *h_fbuf_xy, *h_fbuf_yz, *h_fbuf_xz; // host cuda face buffers

  void reset();             // reset fields
  void device_sync();       // cuda synchronize
  void stream_sync();
  void set_material(int m); // copy material from space
  void get_eh(int m);       // retrieve field
  void update_e();          // update fields
  void update_h();
  void clear_h_fbuf();

  void cuda_e_in();         // cuda face swapping
  void cuda_e_out();
  void cuda_h_in();
  void cuda_h_out();

private:
  int node;
  int cuda_device;
  cudaStream_t cuda_stream;
  size_t buf_size; // size (in bytes) of device buffers
  size_t fbuf_size_xy, fbuf_size_yz, fbuf_size_xz; // size (in bytes) of device face buffers
  F_TYPE **d0;     // device field and material buffer pointers
  F_TYPE *d_fbuf_xy, *d_fbuf_yz, *d_fbuf_xz;  // device cuda face buffers
  size_t cuda_bx, cuda_by, cuda_bz;
  size_t cuda_tx, cuda_ty, cuda_tz;

  void checkDeviceInfo(); // device properties sufficient?
};

#endif // CUDAEH3D_H
