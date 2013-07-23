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
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "defs.h"
#include "cudaEH3d.cuh"
#include "spaceEH3d.h"

CSpaceEH3d::CSpaceEH3d(int rank, size_t sx, size_t sy, size_t sz, CSettings settings)
{
  node = rank;
  sX = sx;   // size
  sY = sy;
  sZ = sz;
  sXY = sx * sy;
  sXYZ = sx * sy * sz;
  use_cuda0 = settings.op_code & CUDA;
  use_cuda1 = settings.op_code & CUDA2;

  // indices for corners of embedded CUDA blocks
  cuda_min_x = ((sX - settings.cuda_x) / 2);
  cuda_min_y = ((sY - settings.cuda_y) / 2);
  cuda_min_z0 = ((sZ / 2) - settings.cuda_z - (settings.cuda_gap / 2));
  cuda_min_z1 = (cuda_min_z0 + settings.cuda_z + settings.cuda_gap);
  cuda_max_x = (cuda_min_x + settings.cuda_x - 1);
  cuda_max_y = (cuda_min_y + settings.cuda_y - 1);
  cuda_max_z0 = (cuda_min_z0 + settings.cuda_z - 1);
  cuda_max_z1 = (cuda_min_z1 + settings.cuda_z - 1);

  yc = new F_TYPE*[MAT_LAST - FIELD_FIRST + 1]; // Yee cell element buffers
  for(int i = FIELD_FIRST; i <= MAT_LAST; i++) yc[i] = new F_TYPE[sXYZ];

  if(use_cuda0) cuda3d0 = new CCudaEH3d(node, 0, settings); // cuda compute object
  if(use_cuda1) cuda3d1 = new CCudaEH3d(node, 1, settings);
}

CSpaceEH3d::~CSpaceEH3d()
{
  if(use_cuda0) delete cuda3d0;
  if(use_cuda1) delete cuda3d1;

  for(int i = FIELD_FIRST; i <= MAT_LAST; i++) delete[] yc[i];
  delete[] yc;
}

void CSpaceEH3d::reset()
{
  for(int j = FIELD_FIRST; j <= FIELD_LAST; j++) {
	for(int i = 0; i < sXYZ; i++) {
      yc[j][i] = 0.0;
	}
  }
  if(use_cuda0) cuda3d0->reset();
  if(use_cuda1) cuda3d1->reset();
}

void CSpaceEH3d::set_cuda_material()
{
  F_TYPE *p;
  for(int m = MAT_FIRST; m <= MAT_LAST; m++) {
    p = cuda3d0->h_buf;
    for(size_t k = (cuda_min_z0 - 1); k <= (cuda_max_z0 + 1); k++){   // include halo
      for(size_t j = (cuda_min_y - 1); j <= (cuda_max_y + 1); j++){
        for(size_t i = (cuda_min_x - 1); i <= (cuda_max_x + 1); i++){
          size_t n = i + j * sX + k * sXY;
          *p++ = yc[m][n];
        }
      }
    }
    cuda3d0->set_material(m);

    if(!use_cuda1) continue;
    p = cuda3d1->h_buf;
    for(size_t k = (cuda_min_z1 - 1); k <= (cuda_max_z1 + 1); k++){   // include halo
      for(size_t j = (cuda_min_y - 1); j <= (cuda_max_y + 1); j++){
        for(size_t i = (cuda_min_x - 1); i <= (cuda_max_x + 1); i++){
          size_t n = i + j * sX + k * sXY;
          *p++ = yc[m][n];
        }
      }
    }
    cuda3d1->set_material(m);
  }
}

void CSpaceEH3d::save(FILE *f)
{
  if(use_cuda0) {
    F_TYPE *p;
    for(int f = FIELD_FIRST; f <= FIELD_LAST; f++) {
      cuda3d0->get_eh(f);
      p = cuda3d0->h_buf;
      for(size_t k = 0; k < (cuda3d0->sZ - 2); k++){    // exclude halo
        for(size_t j = 0; j < (cuda3d0->sY - 2); j++){
          for(size_t i = 0; i < (cuda3d0->sX - 2); i++){
            size_t m = (i + cuda_min_x) + (j + cuda_min_y) * sX + (k + cuda_min_z0) * sXY;
            size_t n = (i + 1) + (j + 1) * cuda3d0->sX + (k + 1) * cuda3d0->sXY;
            yc[f][m] = p[n];
          }
        }
      }
    }
  }
  if(use_cuda1) {
    F_TYPE *p;
    for(int f = FIELD_FIRST; f <= FIELD_LAST; f++) {
      cuda3d1->get_eh(f);
      p = cuda3d1->h_buf;
      for(size_t k = 0; k < (cuda3d1->sZ - 2); k++){    // exclude halo
        for(size_t j = 0; j < (cuda3d1->sY - 2); j++){
          for(size_t i = 0; i < (cuda3d1->sX - 2); i++){
            size_t m = (i + cuda_min_x) + (j + cuda_min_y) * sX + (k + cuda_min_z1) * sXY;
            size_t n = (i + 1) + (j + 1) * cuda3d1->sX + (k + 1) * cuda3d1->sXY;
            yc[f][m] = p[n];
          }
        }
      }
    }
  }

  fprintf(f, "%ld %ld %ld\n", sX - 2, sY - 2, sZ - 2);
  for(size_t k = 1; k < sZ - 1; k++){
    for(size_t j = 1; j < sY - 1; j++){
      for(size_t i = 1; i < sX - 1; i++){
        size_t n = i + j * sX + k * sXY;
        for(int m = FIELD_FIRST; m <= FIELD_LAST; m++)
          fwrite(&yc[m][n], 1, sizeof(F_TYPE), f);
      }
    }
  }
}
/*
	////////////////////////// TEMP
    size_t i, j, k, n;
	i = cuda_min_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;
	i = cuda_min_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[0][n] = 0.03;

	i = cuda_min_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;
	i = cuda_min_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[1][n] = 0.03;

	i = cuda_min_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;
	i = cuda_min_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[2][n] = 0.03;

	i = cuda_min_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;
	i = cuda_min_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[3][n] = 0.03;

	i = cuda_min_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;
	i = cuda_min_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[4][n] = 0.03;

	i = cuda_min_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
	i = cuda_min_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
	i = cuda_max_x; j = cuda_min_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
	i = cuda_min_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_min_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
	i = cuda_max_x; j = cuda_max_y; k = cuda_max_z0; n = i + j * sX + k * sXY; yc[5][n] = 0.03;
*/

#define CUDA_WRITE_E \
  size_t n = i + j * sX + k * sXY;\
  *(pf++) = yc[ex][n];\
  *(pf++) = yc[ey][n];\
  *(pf++) = yc[ez][n];
#define CUDA_READ_E \
  size_t n = i + j * sX + k * sXY;\
  yc[ex][n] = *(pf++);\
  yc[ey][n] = *(pf++);\
  yc[ez][n] = *(pf++);

void CSpaceEH3d::update_e_block
  (size_t x_lo, size_t x_hi, size_t y_lo, size_t y_hi, size_t z_lo, size_t z_hi)
{
  for (size_t k = z_lo; k < z_hi; k++) { // don't update lower & upper e-fields
    for (size_t j = y_lo; j < y_hi; j++) {
      for (size_t i = x_lo; i < x_hi; i++) {
        size_t n = i + j * sX + k * sXY;
        yc[ex][n] = yc[cexe][n] * yc[ex][n]
          + yc[cexh][n] * ((yc[hz][n] - yc[hz][n - sX]) - (yc[hy][n] - yc[hy][n - sXY]));
        yc[ey][n] = yc[ceye][n] * yc[ey][n]
          + yc[ceyh][n] * ((yc[hx][n] - yc[hx][n - sXY]) - (yc[hz][n] - yc[hz][n - 1]));
        yc[ez][n] = yc[ceze][n] * yc[ez][n]
          + yc[cezh][n] * ((yc[hy][n] - yc[hy][n - 1]) - (yc[hx][n] - yc[hx][n - sX]));
      }
    }
  }
}

void CSpaceEH3d::update_e()
{
  ///cudaEvent_t t1, t2;
  ///float time;
  ///cudaEventCreate(&t1);
  ///cudaEventCreate(&t2);
/*
  for (size_t k = 1; k < sZ - 1; k++) { // don't update lower & upper e-fields
    for (size_t j = 1; j < sY - 1; j++) {
      for (size_t i = 1; i < sX - 1; i++) {
       	if((use_cuda0 or use_cuda1) and  // skip cuda cells
       	  (i >= cuda_min_x and i <= cuda_max_x) and
      	  (j >= cuda_min_y and j <= cuda_max_y)) {
        	if(use_cuda0 and (k >= cuda_min_z0 and k <= cuda_max_z0)) continue;
        	if(use_cuda1 and (k >= cuda_min_z1 and k <= cuda_max_z1)) continue;
       	}
        size_t n = i + j * sX + k * sXY;
        yc[ex][n] = yc[cexe][n] * yc[ex][n]
          + yc[cexh][n] * ((yc[hz][n] - yc[hz][n - sX]) - (yc[hy][n] - yc[hy][n - sXY]));
        yc[ey][n] = yc[ceye][n] * yc[ey][n]
          + yc[ceyh][n] * ((yc[hx][n] - yc[hx][n - sXY]) - (yc[hz][n] - yc[hz][n - 1]));
        yc[ez][n] = yc[ceze][n] * yc[ez][n]
          + yc[cezh][n] * ((yc[hy][n] - yc[hy][n - 1]) - (yc[hx][n] - yc[hx][n - sX]));
      }
    }
  }*/

  //if(use_cuda0) cuda3d0->device_sync();
  //if(use_cuda1) cuda3d1->device_sync();

  ///cudaEventRecord(t1, 0);
  F_TYPE *pf;
  if(use_cuda0) cuda3d0->update_e(); // update e-field in cuda
  if(use_cuda1) cuda3d1->update_e(); // update e-field in cuda

  if(not use_cuda0 and not use_cuda1) {
	update_e_block(1, sX - 1, 1, sY - 1, 1, sZ -1);
  }
  else {
	update_e_block(1, cuda_min_x, cuda_min_y, cuda_max_y + 1, 1, sZ - 1);                            // 0
	update_e_block(cuda_max_x + 1, sX - 1, cuda_min_y, cuda_max_y + 1, 1, sZ - 1);                   // 1
	update_e_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, 1, cuda_min_z0);          // 2
	update_e_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_max_z1 + 1, sZ - 1); // 3
	update_e_block(1, sX - 1, 1, cuda_min_y, 1, sZ - 1);                                         // 4
	update_e_block(1, sX - 1, cuda_max_y + 1, sY - 1, 1, sZ - 1);                                    // 5
    if(use_cuda0 and use_cuda1) { // 6
      update_e_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_max_z0 + 1, cuda_min_z1);
    }
    else if(use_cuda0) {
      update_e_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_max_z0 + 1, cuda_max_z1 + 1);
    }
    else if(use_cuda1) {
      update_e_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_min_z0, cuda_min_z1);
    }
  }

  if(use_cuda0) {
	// send lower e-fields to cuda
    pf = cuda3d0->h_fbuf_yz;
    size_t i = cuda_max_x + 1; // yz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_WRITE_E}
    }
    pf = cuda3d0->h_fbuf_xz;
    size_t j = cuda_max_y + 1; // xz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_E}
    }
    pf = cuda3d0->h_fbuf_xy;
    size_t k = cuda_max_z0 + 1; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_E}
    }
    cuda3d0->cuda_e_in(); // cuda pull in faces
  }
  if(use_cuda1) {
	// send lower e-fields to cuda
    pf = cuda3d1->h_fbuf_yz;
    size_t i = cuda_max_x + 1; // yz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_WRITE_E}
    }
    pf = cuda3d1->h_fbuf_xz;
    size_t j = cuda_max_y + 1; // xz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_E}
    }
    pf = cuda3d1->h_fbuf_xy;
    size_t k = cuda_max_z1 + 1; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_E}
    }
    cuda3d1->cuda_e_in(); // cuda pull in faces
  }

  if(use_cuda0) {
    // get upper e-fields from cuda
    cuda3d0->cuda_e_out(); // cuda push out faces
  }
  if(use_cuda1) {
    // get upper e-fields from cuda
    cuda3d1->cuda_e_out(); // cuda push out faces
  }

  if(use_cuda0) {
    cuda3d0->stream_sync();

    pf = cuda3d0->h_fbuf_yz;
    size_t i = cuda_min_x; // yz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_READ_E}
    }
    pf = cuda3d0->h_fbuf_xz;
    size_t j = cuda_min_y; // xz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_E}
    }
    pf = cuda3d0->h_fbuf_xy;
    size_t k = cuda_min_z0; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_E}
    }
  }
  if(use_cuda1) {
    cuda3d1->stream_sync();

    pf = cuda3d1->h_fbuf_yz;
    size_t i = cuda_min_x; // yz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_READ_E}
    }
    pf = cuda3d1->h_fbuf_xz;
    size_t j = cuda_min_y; // xz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_E}
    }
    pf = cuda3d1->h_fbuf_xy;
    size_t k = cuda_min_z1; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_E}
    }
  }
  ///cudaEventRecord(t2, 0);

  ///cudaEventSynchronize(t2);
  ///cudaEventElapsedTime(&time, t1, t2);
  ///std::cout << "E-field update time: " << time << "ms" << std::endl;
}

#define CUDA_WRITE_H \
  size_t n = i + j * sX + k * sXY;\
  *(pf++) = yc[hx][n];\
  *(pf++) = yc[hy][n];\
  *(pf++) = yc[hz][n];
#define CUDA_READ_H \
  size_t n = i + j * sX + k * sXY;\
  yc[hx][n] = *(pf++);\
  yc[hy][n] = *(pf++);\
  yc[hz][n] = *(pf++);

void CSpaceEH3d::update_h_block
  (size_t x_lo, size_t x_hi, size_t y_lo, size_t y_hi, size_t z_lo, size_t z_hi)
{
  for (size_t k = z_lo; k < z_hi; k++) { // don't update lower & upper e-fields
    for (size_t j = y_lo; j < y_hi; j++) {
      for (size_t i = x_lo; i < x_hi; i++) {
        size_t n = i + j * sX + k * sXY;
        yc[hx][n] = yc[chxh][n] * yc[hx][n]
          + yc[chxe][n] * ((yc[ey][n + sXY] - yc[ey][n]) - (yc[ez][n + sX] - yc[ez][n]));
        yc[hy][n] = yc[chyh][n] * yc[hy][n]
          + yc[chye][n] * ((yc[ez][n + 1] - yc[ez][n]) - (yc[ex][n + sXY] - yc[ex][n]));
        yc[hz][n] = yc[chzh][n] * yc[hz][n]
          + yc[chze][n] * ((yc[ex][n + sX] - yc[ex][n]) - (yc[ey][n + 1] - yc[ey][n]));
      }
    }
  }
}

void CSpaceEH3d::update_h()
{ /*
  for (size_t k = 0; k < sZ - 1; k++) {  // don't update upper h-fields
    for (size_t j = 0; j < sY - 1; j++) {
      for (size_t i = 0; i < sX - 1; i++) {
       	if((use_cuda0 or use_cuda1) and  // skip cuda cells
       	  (i >= cuda_min_x and i <= cuda_max_x) and
      	  (j >= cuda_min_y and j <= cuda_max_y)) {
        	if(use_cuda0 and (k >= cuda_min_z0 and k <= cuda_max_z0)) continue;
        	if(use_cuda1 and (k >= cuda_min_z1 and k <= cuda_max_z1)) continue;
       	}
        size_t n = i + j * sX + k * sXY;
        yc[hx][n] = yc[chxh][n] * yc[hx][n]
          + yc[chxe][n] * ((yc[ey][n + sXY] - yc[ey][n]) - (yc[ez][n + sX] - yc[ez][n]));
        yc[hy][n] = yc[chyh][n] * yc[hy][n]
          + yc[chye][n] * ((yc[ez][n + 1] - yc[ez][n]) - (yc[ex][n + sXY] - yc[ex][n]));
        yc[hz][n] = yc[chzh][n] * yc[hz][n]
          + yc[chze][n] * ((yc[ex][n + sX] - yc[ex][n]) - (yc[ey][n + 1] - yc[ey][n]));
      }
    }
  }*/

  //if(use_cuda0) cuda3d0->device_sync();
  //if(use_cuda1) cuda3d1->device_sync();

  F_TYPE *pf;
  if(use_cuda0) cuda3d0->update_h(); // update h-field in cuda
  if(use_cuda1) cuda3d1->update_h(); // update h-field in cuda

  if(not use_cuda0 and not use_cuda1) {
	update_h_block(0, sX - 1, 0, sY - 1, 0, sZ -1);
  }
  else {
	update_h_block(0, cuda_min_x, cuda_min_y, cuda_max_y + 1, 0, sZ - 1);                            // 0
	update_h_block(cuda_max_x + 1, sX - 1, cuda_min_y, cuda_max_y + 1, 0, sZ - 1);                   // 1
	update_h_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, 0, cuda_min_z0);          // 2
	update_h_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_max_z1 + 1, sZ - 1); // 3
	update_h_block(0, sX - 1, 0, cuda_min_y, 0, sZ - 1);                                             // 4
	update_h_block(0, sX - 1, cuda_max_y + 1, sY - 1, 0, sZ - 1);                                    // 5
    if(use_cuda0 and use_cuda1) { // 6
      update_h_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_max_z0 + 1, cuda_min_z1);
    }
    else if(use_cuda0) {
      update_h_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_max_z0 + 1, cuda_max_z1 + 1);
    }
    else if(use_cuda1) {
      update_h_block(cuda_min_x, cuda_max_x + 1, cuda_min_y, cuda_max_y + 1, cuda_min_z0, cuda_min_z1);
    }
  }

  if(use_cuda0) {
    // send upper h-fields to cuda
    pf = cuda3d0->h_fbuf_yz;
    size_t i = cuda_min_x - 1; // yz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_WRITE_H}
    }
    pf = cuda3d0->h_fbuf_xz;
    size_t j = cuda_min_y - 1; // xz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_H}
    }
    pf = cuda3d0->h_fbuf_xy;
    size_t k = cuda_min_z0 - 1; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_H}
    }
	cuda3d0->cuda_h_in(); // cuda pull in faces
  }
  if(use_cuda1) {
    // send upper h-fields to cuda
    pf = cuda3d1->h_fbuf_yz;
    size_t i = cuda_min_x - 1; // yz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_WRITE_H}
    }
    pf = cuda3d1->h_fbuf_xz;
    size_t j = cuda_min_y - 1; // xz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_H}
    }
    pf = cuda3d1->h_fbuf_xy;
    size_t k = cuda_min_z1 - 1; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_WRITE_H}
    }
	cuda3d1->cuda_h_in(); // cuda pull in faces
  }

  if(use_cuda0) {
	// get lower h-fields from cuda
	cuda3d0->cuda_h_out(); // cuda push out faces
  }
  if(use_cuda1) {
	// get lower h-fields from cuda
	cuda3d1->cuda_h_out(); // cuda push out faces
  }

  if(use_cuda0) {
    cuda3d0->stream_sync();

    pf = cuda3d0->h_fbuf_yz;
    size_t i = cuda_max_x; // yz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_READ_H}
    }
    pf = cuda3d0->h_fbuf_xz;
    size_t j = cuda_max_y; // xz plane face
    for(size_t k = cuda_min_z0; k <= cuda_max_z0 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_H}
    }
    pf = cuda3d0->h_fbuf_xy;
    size_t k = cuda_max_z0; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_H}
    }
  }
  if(use_cuda1) {
    cuda3d1->stream_sync();

    pf = cuda3d1->h_fbuf_yz;
    size_t i = cuda_max_x; // yz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t j = cuda_min_y; j <= cuda_max_y; j++){CUDA_READ_H}
    }
    pf = cuda3d1->h_fbuf_xz;
    size_t j = cuda_max_y; // xz plane face
    for(size_t k = cuda_min_z1; k <= cuda_max_z1 ; k++) {
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_H}
    }
    pf = cuda3d1->h_fbuf_xy;
    size_t k = cuda_max_z1; // xy plane face
    for(size_t j = cuda_min_y; j <= cuda_max_y; j++){
      for(size_t i = cuda_min_x; i <= cuda_max_x; i++){CUDA_READ_H}
    }
  }
}

#define WRITE_E \
  size_t n = i + j * sX + k * sXY;\
  yc[ex][n] = *(pf++);\
  yc[ey][n] = *(pf++);\
  yc[ez][n] = *(pf++);
#define WRITE_H \
  size_t n = i + j * sX + k * sXY;\
  yc[hx][n] = *(pf++);\
  yc[hy][n] = *(pf++);\
  yc[hz][n] = *(pf++);

// shared faces are smaller by one each edge!!
void CSpaceEH3d::write_hi_ex(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t i = sX - 1;
  for(size_t k = 1; k < sZ - 1; k++) 
    for(size_t j = 1; j < sY - 1; j++) {WRITE_E}
}
void CSpaceEH3d::write_hi_ey(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t j = sY - 1;
  for(size_t k = 1; k < sZ - 1; k++)
    for(size_t i = 1; i < sX - 1; i++) {WRITE_E}
}
void CSpaceEH3d::write_hi_ez(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t k = sZ - 1;
  for(size_t j = 1; j < sY - 1; j++)
    for(size_t i = 1; i < sX - 1; i++) {WRITE_E}
}
void CSpaceEH3d::write_lo_hx(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t i = 0;
  for(size_t k = 1; k < sZ - 1; k++)
    for(size_t j = 1; j < sY - 1; j++) {WRITE_H}
}
void CSpaceEH3d::write_lo_hy(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t j = 0;
  for(size_t k = 1; k < sZ - 1; k++)
    for(size_t i = 1; i < sX - 1; i++) {WRITE_H}
}
void CSpaceEH3d::write_lo_hz(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t k = 0;
  for(size_t j = 1; j < sY - 1; j++)
    for(size_t i = 1; i < sX - 1; i++) {WRITE_H}
}

#define READ_E \
  size_t n = i + j * sX + k * sXY;\
  *(pf++) = yc[ex][n];\
  *(pf++) = yc[ey][n];\
  *(pf++) = yc[ez][n];
#define READ_H \
  size_t n = i + j * sX + k * sXY;\
  *(pf++) = yc[hx][n];\
  *(pf++) = yc[hy][n];\
  *(pf++) = yc[hz][n];

// shared faces are smaller by one each edge!!
void CSpaceEH3d::read_lo_ex(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t i = 1;
  for(size_t k = 1; k < sZ - 1; k++)
    for(size_t j = 1; j < sY - 1; j++) {READ_E}
}
void CSpaceEH3d::read_lo_ey(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t j = 1;
  for(size_t k = 1; k < sZ - 1; k++)
    for(size_t i = 1; i < sX - 1; i++) {READ_E}
}
void CSpaceEH3d::read_lo_ez(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t k = 1;
  for(size_t j = 1; j < sY - 1; j++)
    for(size_t i = 1; i < sX - 1; i++) {READ_E}
}
void CSpaceEH3d::read_hi_hx(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t i = sX - 2;
  for(size_t k = 1; k < sZ - 1; k++)
    for(size_t j = 1; j < sY - 1; j++) {READ_H}
}
void CSpaceEH3d::read_hi_hy(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t j = sY - 2;
  for(size_t k = 1; k < sZ - 1; k++)
    for(size_t i = 1; i < sX - 1; i++) {READ_H}
}
void CSpaceEH3d::read_hi_hz(F_TYPE *face)
{
  F_TYPE *pf = face;
  size_t k = sZ - 2;
  for(size_t j = 1; j < sY - 1; j++)
    for(size_t i = 1; i < sX - 1; i++) {READ_H}
}

