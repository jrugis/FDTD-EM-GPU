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

#ifndef DEFS_H_INCLUDED
#define DEFS_H_INCLUDED

#include <cstdlib>
#include <time.h>

/* Sample ./FDTD_02 command line arguments
 * size, steps, op_code, cuda_x_y, cuda_z, cuda_bx_by, cuda_bz, cuda_t, cuda_gap
 *
 * 100	120	0	0	0	0	0	0	0	// no mpi, no cuda
 * 200	120	1	0	0	0	0	0	0	// mpi, no cuda
 *
 * 100 	120	2	8	8	1	1	10	8	// no mpi, min cuda
 * 200	120	3	8	8	1	1	10	8	// mpi, min cuda
 *
 * 430	2	2	418	208	42	21	10	8	// no mpi, max cuda m2090
 * 860	2	3	418	208	42	21	10	8	// mpi, max cuda m2090
 *
 * 450	2	2	428	218	43	22	10	8	// no mpi, max cuda k20Xm
 *
 */

// settings opcode bits
#define MPI 0x0001
#define CUDA 0x0002
#define CUDA2 0x0004
#define SAVE 0x0008

struct CSettings
{
  size_t size;
  size_t steps;
  unsigned int op_code;
  size_t cuda_x, cuda_y, cuda_z;
  size_t cuda_bx, cuda_by, cuda_bz;
  size_t cuda_tx, cuda_ty, cuda_tz;
  size_t cuda_gap;
};

// Yee-cell elements
enum YeeElementType
{
  ex, ey, ez, hx, hy, hz,
  cexe, ceye, ceze, cexh, ceyh, cezh,
  chxe, chye, chze, chxh, chyh, chzh
};
#define FIELD_FIRST ex
#define FIELD_LAST hz
#define MAT_FIRST cexe
#define MAT_LAST chzh
#define FIELD_DIM 3

// the simulation data type, SET AS A PAIR!
//#define F_TYPE float
//#define MPI_F_TYPE MPI_FLOAT
#define F_TYPE double
#define MPI_F_TYPE MPI_DOUBLE

// ***********************************************************************
// math constants
// ***********************************************************************
#define PI 3.14159265358979323846

// ***********************************************************************
// electrical constants
// ***********************************************************************
#define IMP0 377.0
#define DTDS2D (1.0 / sqrt(2.0))
#define DTDS3D (1.0 / sqrt(3.0))

// ***********************************************************************

#endif // DEFS_H_INCLUDED
