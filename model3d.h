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

#ifndef MODEL3D_H
#define MODEL3D_H

#include "defs.h"

#define M1000
#define F1000

class CSpaceEH3d;
//class CAbc1o3d;
//class CTfsf3d;

class CModel3D
{
public:
  CModel3D(int rank, size_t sizeL, CSettings settings);
  ~CModel3D();

  F_TYPE *share_out_face[3], *share_in_face[3]; // shared face data
  size_t size_face;                             // size of each shared face, elements are the
                                                //   three components of either e or h field
  void reset();
  void stepA();
  void stepB();
  void save_data();

private:
  CSpaceEH3d *space3d;        // 3d simulation space
  //CAbc1o3d *abc1o3d;          // first order abc
  //CTfsf3d *tfsf3d;            // tfsf in 3d space

  size_t time_step;           // time step
  int node;                   // mpi node for this model
  size_t sizeL, sizeF, sizeT; // model block size
  bool use_cuda, use_mpi;

  void set_material();
  void push_e(); // push shared e-field to eh-space
  void push_h(); // push shared h-field to eh-space
  void pull_e(); // pull shared e-field from eh-space
  void pull_h(); // pull shared h-field from eh-space
};

#endif // MODEL3D_H
