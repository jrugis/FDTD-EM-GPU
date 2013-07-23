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

#include <stdio.h>
#include <math.h>
//#include <stdlib.h>
//#include <limits.h>

#include "defs.h"
#include "spaceEH3d.h"
#include "source.h"
//#include "abc1o3d.h"
//#include "tfsf3d.h"
#include "model3d.h"

CModel3D::CModel3D(int rank, size_t l, CSettings settings)
{
  node = rank;
  sizeL = l + 2;     // block size includes extra outer faces (three shared)
  sizeF = sizeL * sizeL;
  sizeT = sizeF * sizeL;
  use_cuda = settings.op_code & CUDA;
  use_mpi = settings.op_code & MPI;

  // shared face data
  //   three (e or h) field components 
  size_face = 3 * l * l; 
  //size_face = 3 * sizeF; 

  //abc1o3d = NULL;
  //tfsf3d = NULL;

  for(int i = 0; i < 3; i++){ // 3x shared face data
    share_in_face[i] = new F_TYPE[size_face];
    share_out_face[i] = new F_TYPE[size_face];
  }

  space3d = new CSpaceEH3d(node, sizeL, sizeL, sizeL, settings);
  set_material(); // space & material
  reset();        // space & material
}

CModel3D::~CModel3D()
{
  delete space3d;
  //if(abc1o2d != NULL) delete abc1o3d;  
  //if(tfsf2d != NULL) delete tfsf3d;  
}

// ***********************************************************************
// model materials
// ***********************************************************************
void CModel3D::set_material()
{
#ifdef M1000
  for(size_t i = 0; i < sizeT; i++) {
    space3d->yc[cexe][i] = space3d->yc[ceye][i] = space3d->yc[ceze][i] = 1.0; // free-space
    space3d->yc[cexh][i] = space3d->yc[ceyh][i] = space3d->yc[cezh][i] = DTDS3D * IMP0;
    space3d->yc[chxh][i] = space3d->yc[chyh][i] = space3d->yc[chzh][i] = 1.0;
    space3d->yc[chxe][i] = space3d->yc[chye][i] = space3d->yc[chze][i] = DTDS3D / IMP0;
  }
  if(use_cuda) space3d->set_cuda_material();
  // set tfsf and abc's after material initialization!!!
  //abc1o3d = new CAbc1o3d(space3d);
#endif
}

// ***********************************************************************
// model field step
// ***********************************************************************
void CModel3D::stepA()
{
  push_e(); // to eh-space because e-field has been updated and shared
  space3d->update_h();  // update magnetic field
  pull_h(); // from eh-space because h-field has been updated and is ready to be shared
}
void CModel3D::stepB()
{
  push_h(); // to eh-space because h-field has been updated and shared
  #ifdef F1000  // central dipole source
  //#define SRCXYZ (sizeL - 2) // mpi lower center
  //#else
  ///#define SRCXYZ ((sizeL / 2) - 1) // non-mpi lower center
  //#endif
  //#define SRC (SRCXYZ + SRCXYZ * sizeL + SRCXYZ * sizeF)
  //#define SRC ((4 * sizeL / 5) * (1 + sizeL + sizeF))
  //#define SRC (45 * (1 + sizeL + sizeF))
  size_t srcxyz = use_mpi ? (sizeL - 2) : ((sizeL / 2) - 1);
  size_t src = srcxyz + srcxyz * sizeL + srcxyz * sizeF;
  #define WTS 100
  #define DTS (WTS * 2) // delay time steps
  #define NWTSS (-1.0 * pow(WTS, 2)) // negative ((width time steps) squared)
  space3d->update_e();  // update electric field
  if(node == 0) sourceRicker(false, 30.0, &(space3d->yc[ez][src]), time_step, WTS);
  //if(node == 0) sourceGaussian(false, 30.0, &(space3d->c[SRC].ey), time_step, DTS, NWTSS);
  //abc1o3d->update_e();
  pull_e(); // from eh-space because e-field has been updated and is ready to be shared
  #endif
  time_step++;
}

// only three pushes total for each node
void CModel3D::push_e() // share-in data to hi-side faces
{
  if(~node & 0x01) space3d->write_hi_ex(share_in_face[0]);
  if(~node & 0x02) space3d->write_hi_ez(share_in_face[2]);
  if(~node & 0x04) space3d->write_hi_ey(share_in_face[1]);
}  
void CModel3D::push_h() // share-in data to low-side faces
{
  if(node & 0x01) space3d->write_lo_hx(share_in_face[0]);
  if(node & 0x02) space3d->write_lo_hz(share_in_face[2]);
  if(node & 0x04) space3d->write_lo_hy(share_in_face[1]);
}

// only three pulls total for each node
void CModel3D::pull_e() // share-out data from lo-side faces
{
  if(node & 0x01) space3d->read_lo_ex(share_out_face[0]);
  if(node & 0x02) space3d->read_lo_ez(share_out_face[2]);
  if(node & 0x04) space3d->read_lo_ey(share_out_face[1]);
}
void CModel3D::pull_h()  // share-out data from hi-side faces
{
  if(~node & 0x01) space3d->read_hi_hx(share_out_face[0]);
  if(~node & 0x02) space3d->read_hi_hz(share_out_face[2]);
  if(~node & 0x04) space3d->read_hi_hy(share_out_face[1]);
}

void CModel3D::reset()
{
  time_step = 0;
  for(int i = 0; i < 3; i++) // 3x shared face data
    for(size_t m = 0; m < size_face; m++)
      share_in_face[i][m] = share_out_face[i][m] = 0.0;
  space3d->reset();
  //if(abc1o3d != NULL) abc1o3d->reset();
  //if(tfsf3d != NULL) tfsf3d->reset();
}

void CModel3D::save_data()
{
  FILE *f_out;
  char fname[20];
  sprintf(fname, "data%d.bin", node);
  f_out = fopen(fname, "wb");
  space3d->save(f_out);
  fclose(f_out);
}
