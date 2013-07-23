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

#ifndef SPACEEH3D_H
#define SPACEEH3D_H

#include <stdio.h>

#include "defs.h"

class CCudaEH3d;

class CSpaceEH3d
{
public:
  CSpaceEH3d(int rank, size_t sx, size_t sy, size_t sz, CSettings settings);
  ~CSpaceEH3d();

  size_t sX, sY, sZ;  // size
  size_t sXY, sXYZ;   // size
  F_TYPE **yc;   // field and material buffer pointers (Yee cell)

  void reset();
  void set_cuda_material();
  void update_e();
  void update_h();
  void write_hi_ex(F_TYPE *face);
  void write_hi_ey(F_TYPE *face);
  void write_hi_ez(F_TYPE *face);
  void write_lo_hx(F_TYPE *face);
  void write_lo_hy(F_TYPE *face);
  void write_lo_hz(F_TYPE *face);
  void read_lo_ex(F_TYPE *face);
  void read_lo_ey(F_TYPE *face);
  void read_lo_ez(F_TYPE *face);
  void read_hi_hx(F_TYPE *face);
  void read_hi_hy(F_TYPE *face);
  void read_hi_hz(F_TYPE *face);

  void save(FILE *f);

private:
  int node;
  bool use_cuda0, use_cuda1;
  size_t cuda_min_x, cuda_min_y, cuda_min_z0, cuda_min_z1;
  size_t cuda_max_x, cuda_max_y, cuda_max_z0, cuda_max_z1;
  CCudaEH3d *cuda3d0, *cuda3d1;

  void update_e_block(size_t x_lo, size_t x_hi, size_t y_lo, size_t y_hi, size_t z_lo, size_t z_hi);
  void update_h_block(size_t x_lo, size_t x_hi, size_t y_lo, size_t y_hi, size_t z_lo, size_t z_hi);
};

#endif // SPACEEH3D_H
