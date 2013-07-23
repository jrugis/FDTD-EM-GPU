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
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "defs.h"
#include "cudaEH3d.cuh"

using namespace std;

// Cuda constant memory
//__constant__ unsigned_long cuda_sXY, cuda_sXYZ;

////////////////////////////////////////////////////////////////////////////////
// CUDA device kernel(s)
////////////////////////////////////////////////////////////////////////////////
__global__ void e_in_Kernel_xy(
  const F_TYPE *d_fbuf_xy,
  F_TYPE *exG, F_TYPE *eyG, F_TYPE *ezG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + (blockIdx.x + 1) * sX + (sZ - 1) * sXY; // z = max

  exG[n] = d_fbuf_xy[m];
  eyG[n] = d_fbuf_xy[m + 1];
  ezG[n] = d_fbuf_xy[m + 2];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void e_in_Kernel_yz(
  const F_TYPE *d_fbuf_yz,
  F_TYPE *exG, F_TYPE *eyG, F_TYPE *ezG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (sX - 1) + (threadIdx.x + 1) * sX + (blockIdx.x + 1) * sXY; // x = max

  exG[n] = d_fbuf_yz[m];
  eyG[n] = d_fbuf_yz[m + 1];
  ezG[n] = d_fbuf_yz[m + 2];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void e_in_Kernel_xz(
  const F_TYPE *d_fbuf_xz,
  F_TYPE *exG, F_TYPE *eyG, F_TYPE *ezG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + (sY - 1) * sX + (blockIdx.x + 1) * sXY; // y = max

  exG[n] = d_fbuf_xz[m];
  eyG[n] = d_fbuf_xz[m + 1];
  ezG[n] = d_fbuf_xz[m + 2];
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void e_out_Kernel_xy(
  F_TYPE *d_fbuf_xy,
  const F_TYPE *exG, const F_TYPE *eyG, const F_TYPE *ezG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + (blockIdx.x + 1) * sX + sXY; // z = 1

  d_fbuf_xy[m] = exG[n];
  d_fbuf_xy[m + 1] = eyG[n];
  d_fbuf_xy[m + 2] = ezG[n];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void e_out_Kernel_yz(
  F_TYPE *d_fbuf_yz,
  const F_TYPE *exG, const F_TYPE *eyG, const F_TYPE *ezG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = 1 + (threadIdx.x + 1) * sX + (blockIdx.x + 1) * sXY;  // x = 1

  d_fbuf_yz[m] = exG[n];
  d_fbuf_yz[m + 1] = eyG[n];
  d_fbuf_yz[m + 2] = ezG[n];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void e_out_Kernel_xz(
  F_TYPE *d_fbuf_xz,
  const F_TYPE *exG, const F_TYPE *eyG, const F_TYPE *ezG,
  size_t sX, size_t sY, size_t sz, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + sX + (blockIdx.x + 1) * sXY; // y = 1

  d_fbuf_xz[m] = exG[n];
  d_fbuf_xz[m + 1] = eyG[n];
  d_fbuf_xz[m + 2] = ezG[n];
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void h_in_Kernel_xy(
  const F_TYPE *d_fbuf_xy,
  F_TYPE *hxG, F_TYPE *hyG, F_TYPE *hzG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + (blockIdx.x + 1) * sX; // z = 0

  hxG[n] = d_fbuf_xy[m];
  hyG[n] = d_fbuf_xy[m + 1];
  hzG[n] = d_fbuf_xy[m + 2];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void h_in_Kernel_yz(
  const F_TYPE *d_fbuf_yz,
  F_TYPE *hxG, F_TYPE *hyG, F_TYPE *hzG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) * sX + (blockIdx.x + 1) * sXY;  // x = 0

  hxG[n] = d_fbuf_yz[m];
  hyG[n] = d_fbuf_yz[m + 1];
  hzG[n] = d_fbuf_yz[m + 2];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void h_in_Kernel_xz(
  const F_TYPE *d_fbuf_xz,
  F_TYPE *hxG, F_TYPE *hyG, F_TYPE *hzG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + (blockIdx.x + 1) * sXY; // y = 0

  hxG[n] = d_fbuf_xz[m];
  hyG[n] = d_fbuf_xz[m + 1];
  hzG[n] = d_fbuf_xz[m + 2];
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void h_out_Kernel_xy(
  F_TYPE *d_fbuf_xy,
  const F_TYPE *hxG, const F_TYPE *hyG, const F_TYPE *hzG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + (blockIdx.x + 1) * sX + (sZ - 2) * sXY; // z = max - 1

  d_fbuf_xy[m] = hxG[n];
  d_fbuf_xy[m + 1] = hyG[n];
  d_fbuf_xy[m + 2] = hzG[n];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void h_out_Kernel_yz(
  F_TYPE *d_fbuf_yz,
  const F_TYPE *hxG, const F_TYPE *hyG, const F_TYPE *hzG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (sX - 2) + (threadIdx.x + 1) * sX + (blockIdx.x + 1) * sXY; // x = max - 1

  d_fbuf_yz[m] = hxG[n];
  d_fbuf_yz[m + 1] = hyG[n];
  d_fbuf_yz[m + 2] = hzG[n];
}
////////////////////////////////////////////////////////////////////////////////
__global__ void h_out_Kernel_xz(
  F_TYPE *d_fbuf_xz,
  const F_TYPE *hxG, const F_TYPE *hyG, const F_TYPE *hzG,
  size_t sX, size_t sY, size_t sZ, size_t sXY)
{
  size_t m = (threadIdx.x + (blockIdx.x * blockDim.x)) * 3;
  size_t n = (threadIdx.x + 1) + (sY - 2) * sX + (blockIdx.x + 1) * sXY; // y = max - 1

  d_fbuf_xz[m] = hxG[n];
  d_fbuf_xz[m + 1] = hyG[n];
  d_fbuf_xz[m + 2] = hzG[n];
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void update_e_Kernel(
  F_TYPE *exG, F_TYPE *eyG, F_TYPE *ezG,
  const F_TYPE *hxG, const F_TYPE *hyG, const F_TYPE *hzG,
  const F_TYPE *cexeG, const F_TYPE *ceyeG, const F_TYPE *cezeG,
  const F_TYPE *cexhG, const F_TYPE *ceyhG, const F_TYPE *cezhG,
  const size_t sX, const size_t sXY)
{
  const size_t n =
	(((blockIdx.z * blockDim.z) + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y)
  + (((blockIdx.y * blockDim.y) + threadIdx.y) * gridDim.x * blockDim.x)
  +   (blockIdx.x * blockDim.x) + threadIdx.x;

  // skip halo
  if( (blockIdx.x == 0 and threadIdx.x == 0) or
	  (blockIdx.y == 0 and threadIdx.y == 0) or
	  (blockIdx.z == 0 and threadIdx.z == 0) or
	  (blockIdx.x == (gridDim.x - 1) and threadIdx.x == (blockDim.x - 1)) or
	  (blockIdx.y == (gridDim.y - 1) and threadIdx.y == (blockDim.y - 1)) or
	  (blockIdx.z == (gridDim.z - 1) and threadIdx.z == (blockDim.z - 1))) return;

  ////////////////// TEMP
//    exG[n] = 0.0;
//    eyG[n] = 0.0;
//    ezG[n] = 0.0;

  exG[n] = cexeG[n] * exG[n]
	+ cexhG[n] * ((hzG[n] - hzG[n - sX]) - (hyG[n] - hyG[n - sXY]));
  eyG[n] = ceyeG[n] * eyG[n]
    + ceyhG[n] * ((hxG[n] - hxG[n - sXY]) - (hzG[n] - hzG[n - 1]));
  ezG[n] = cezeG[n] * ezG[n]
    + cezhG[n] * ((hyG[n] - hyG[n - 1]) - (hxG[n] - hxG[n - sX]));

}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void update_h_Kernel(
  F_TYPE *hxG, F_TYPE *hyG, F_TYPE *hzG,
  const F_TYPE *exG, const F_TYPE *eyG, const F_TYPE *ezG,
  const F_TYPE *chxhG, const F_TYPE *chyhG, const F_TYPE *chzhG,
  const F_TYPE *chxeG, const F_TYPE *chyeG, const F_TYPE *chzeG,
  const size_t sX, const size_t sXY)
{
  const size_t n =
	(((blockIdx.z * blockDim.z) + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y)
  + (((blockIdx.y * blockDim.y) + threadIdx.y) * gridDim.x * blockDim.x)
  +   (blockIdx.x * blockDim.x) + threadIdx.x;

  //  skip halo
  if( (blockIdx.x == 0 and threadIdx.x == 0) or
	  (blockIdx.y == 0 and threadIdx.y == 0) or
	  (blockIdx.z == 0 and threadIdx.z == 0) or
          (blockIdx.x == (gridDim.x - 1) and threadIdx.x == (blockDim.x - 1)) or
 	  (blockIdx.y == (gridDim.y - 1) and threadIdx.y == (blockDim.y - 1)) or
  	  (blockIdx.z == (gridDim.z - 1) and threadIdx.z == (blockDim.z - 1))) return;

  ////////////////// TEMP
//    hxG[n] = 0.0;
//    hyG[n] = 0.0;
//    hzG[n] = 0.0;

  hxG[n] = chxhG[n] * hxG[n]
    + chxeG[n] * ((eyG[n + sXY] - eyG[n]) - (ezG[n + sX] - ezG[n]));
  hyG[n] = chyhG[n] * hyG[n]
    + chyeG[n] * ((ezG[n + 1] - ezG[n]) - (exG[n + sXY] - exG[n]));
  hzG[n] = chzhG[n] * hzG[n]
    + chzeG[n] * ((exG[n + sX] - exG[n]) - (eyG[n + 1] - eyG[n]));

}
////////////////////////////////////////////////////////////////////////////////


// utility functions
void abort(string message)
{
  cout << "ABORT: " << message << endl;
  exit(1);
}

void cudaKill(cudaError error_id, string message)
{
  cout << "ERROR: " << message << " returned ";
  cout << (int)error_id << " " << cudaGetErrorString(error_id) << endl;
  exit(1);
}

// CUDA error checking macro
#define CUDA_CHECK(cuda_function, message) \
  error_id = cuda_function; \
  if(error_id != cudaSuccess) cudaKill(error_id, message);


CCudaEH3d::CCudaEH3d(int rank, int device, CSettings settings)
{
  cudaError error_id;
  node = rank;

  // check CUDA device
  cuda_device = device;
  checkDeviceInfo();

  cuda_bx = settings.cuda_bx;
  cuda_by = settings.cuda_by;
  cuda_bz = settings.cuda_bz;
  cuda_tx = settings.cuda_tx;
  cuda_ty = settings.cuda_ty;
  cuda_tz = settings.cuda_tz;

  sX = settings.cuda_x + 2;   // stride sizes
  sY = settings.cuda_y + 2;
  sZ = settings.cuda_z + 2;
  sXY = sX * sY;
  sXYZ = sX * sY * sZ;

  fsX = settings.cuda_x;  // face sizes
  fsY = settings.cuda_y;
  fsZ = settings.cuda_z;

  fbuf_size_xy = fsX * fsY * FIELD_DIM * sizeof(F_TYPE); // face sizes in bytes
  fbuf_size_yz = fsY * fsZ * FIELD_DIM * sizeof(F_TYPE); // face sizes in bytes
  fbuf_size_xz = fsX * fsZ * FIELD_DIM * sizeof(F_TYPE); // face sizes in bytes
  buf_size = sXYZ * sizeof(F_TYPE); // field and material elements size in bytes

  d0 = new F_TYPE*[1 + MAT_LAST - FIELD_FIRST];  // field and material device buffers

  cudaSetDevice(cuda_device);
  cudaDeviceReset();
  //CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared), "cudaDeviceSetCacheConfig")
  CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), "cudaDeviceSetCacheConfig")
  CUDA_CHECK(cudaStreamCreate(&cuda_stream), "cudaStreamCreate")

  // allocate device buffers
  for(int i = FIELD_FIRST; i <= MAT_LAST; i++)
	CUDA_CHECK(cudaMalloc(&d0[i], buf_size), "cudaMalloc")
  CUDA_CHECK(cudaMalloc(&d_fbuf_xy, fbuf_size_xy), "cudaMalloc")
  CUDA_CHECK(cudaMalloc(&d_fbuf_yz, fbuf_size_yz), "cudaMalloc")
  CUDA_CHECK(cudaMalloc(&d_fbuf_xz, fbuf_size_xz), "cudaMalloc")
  size_t mem_free, mem_total;
  cudaMemGetInfo(&mem_free, &mem_total);
  cout << node << " memory device/free/total: " << device << "/" << mem_free << "/" << mem_total << endl;

  // allocate host buffers (after the devices are setup!)
  //h_buf = (double *)malloc(buf_size);
  CUDA_CHECK(cudaMallocHost(&h_buf, buf_size, cudaHostAllocPortable), "cudaMallocHost")
  CUDA_CHECK(cudaMallocHost(&h_fbuf_xy, fbuf_size_xy, cudaHostAllocPortable), "cudaMallocHost")
  CUDA_CHECK(cudaMallocHost(&h_fbuf_yz, fbuf_size_yz, cudaHostAllocPortable), "cudaMallocHost")
  CUDA_CHECK(cudaMallocHost(&h_fbuf_xz, fbuf_size_xz, cudaHostAllocPortable), "cudaMallocHost")
}

CCudaEH3d::~CCudaEH3d()
{
  cudaSetDevice(cuda_device);
  cudaStreamDestroy(cuda_stream);
  //free(h_buf);
  cudaFreeHost(h_buf);
  cudaFreeHost(h_fbuf_xy);
  cudaFreeHost(h_fbuf_yz);
  cudaFreeHost(h_fbuf_xz);
  for(int i = FIELD_FIRST; i <= MAT_LAST; i++) cudaFree(&d0[i]);
  cudaFree(d_fbuf_xy);
  cudaFree(d_fbuf_yz);
  cudaFree(d_fbuf_xz);
  cudaDeviceReset();
  delete[] d0;
}

void CCudaEH3d::reset()
{  
  cudaError error_id;

  // initialise host buffer & copy to device(s)
  cudaSetDevice(cuda_device);
  for(size_t i = 0; i < sXYZ; i++) h_buf[i] = 0.0;
  for(int i = FIELD_FIRST; i <= FIELD_LAST; i++) {
    CUDA_CHECK(cudaMemcpy(d0[i], h_buf, buf_size, cudaMemcpyDefault), "cudaMemcpy") // host -> device
    //CUDA_CHECK(cudaMemcpyAsync(d0[i], h_buf, buf_size, cudaMemcpyHostToDevice, 0), "cudaMemcpy") // host -> device
  }
}
void CCudaEH3d::clear_h_fbuf()   ////////////////// TEMP
{
  cudaError error_id;
  cudaSetDevice(cuda_device);
  CUDA_CHECK(cudaMemset(h_fbuf_xy, 0, fbuf_size_xy), "cudaMemset")
  CUDA_CHECK(cudaMemset(h_fbuf_yz, 0, fbuf_size_yz), "cudaMemset")
  CUDA_CHECK(cudaMemset(h_fbuf_xz, 0, fbuf_size_xz), "cudaMemset")
}

void CCudaEH3d::device_sync()
{
  cudaSetDevice(cuda_device);
  cudaDeviceSynchronize();
}

void CCudaEH3d::stream_sync()
{
  cudaStreamSynchronize(cuda_stream);
}

void CCudaEH3d::set_material(int m)
{
  cudaError error_id;
  cudaSetDevice(cuda_device);
  CUDA_CHECK(cudaMemcpy(d0[m], h_buf, buf_size, cudaMemcpyDefault), "cudaMemcpy")
  //CUDA_CHECK(cudaMemcpyAsync(d0[m], h_buf, buf_size, cudaMemcpyHostToDevice, 0), "cudaMemcpy")
  //cudaDeviceSynchronize();
}

void CCudaEH3d::cuda_e_in()
{
  cudaError error_id;
  dim3 blocks, threads;
  cudaSetDevice(cuda_device);
//  CUDA_CHECK(cudaMemcpy(d_fbuf_xy, h_fbuf_xy, fbuf_size_xy, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(d_fbuf_yz, h_fbuf_yz, fbuf_size_yz, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(d_fbuf_xz, h_fbuf_xz, fbuf_size_xz, cudaMemcpyDefault), "cudaMemcpy")
  CUDA_CHECK(cudaMemcpyAsync(d_fbuf_xy, h_fbuf_xy, fbuf_size_xy, cudaMemcpyHostToDevice, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(d_fbuf_yz, h_fbuf_yz, fbuf_size_yz, cudaMemcpyHostToDevice, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(d_fbuf_xz, h_fbuf_xz, fbuf_size_xz, cudaMemcpyHostToDevice, cuda_stream), "cudaMemcpyAsync")
  threads.x = fsX; blocks.x = fsY;
  e_in_Kernel_xy<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xy, d0[ex], d0[ey], d0[ez], sX, sY, sZ, sXY); // <<<blocks,  threads, 0, cuda_stream>>>
  threads.x = fsY; blocks.x = fsZ;
  e_in_Kernel_yz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_yz, d0[ex], d0[ey], d0[ez], sX, sY, sZ, sXY);
  threads.x = fsX; blocks.x = fsZ;
  e_in_Kernel_xz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xz, d0[ex], d0[ey], d0[ez], sX, sY, sZ, sXY);
}
void CCudaEH3d::cuda_e_out()
{
  cudaError error_id;
  dim3 blocks, threads;
  cudaSetDevice(cuda_device);
  threads.x = fsX; blocks.x = fsY;
  e_out_Kernel_xy<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xy, d0[ex], d0[ey], d0[ez], sX, sY, sZ, sXY); // <<<blocks,  threads, 0, cuda_stream>>>
  threads.x = fsY; blocks.x = fsZ;
  e_out_Kernel_yz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_yz, d0[ex], d0[ey], d0[ez], sX, sY, sZ, sXY);
  threads.x = fsX; blocks.x = fsZ;
  e_out_Kernel_xz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xz, d0[ex], d0[ey], d0[ez], sX, sY, sZ, sXY);
//  CUDA_CHECK(cudaMemcpy(h_fbuf_xy, d_fbuf_xy, fbuf_size_xy, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(h_fbuf_yz, d_fbuf_yz, fbuf_size_yz, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(h_fbuf_xz, d_fbuf_xz, fbuf_size_xz, cudaMemcpyDefault), "cudaMemcpy")
  CUDA_CHECK(cudaMemcpyAsync(h_fbuf_xy, d_fbuf_xy, fbuf_size_xy, cudaMemcpyDeviceToHost, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(h_fbuf_yz, d_fbuf_yz, fbuf_size_yz, cudaMemcpyDeviceToHost, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(h_fbuf_xz, d_fbuf_xz, fbuf_size_xz, cudaMemcpyDeviceToHost, cuda_stream), "cudaMemcpyAsync")
}
void CCudaEH3d::cuda_h_in()
{
  cudaError error_id;
  dim3 blocks, threads;
  cudaSetDevice(cuda_device);
//  CUDA_CHECK(cudaMemcpy(d_fbuf_xy, h_fbuf_xy, fbuf_size_xy, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(d_fbuf_yz, h_fbuf_yz, fbuf_size_yz, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(d_fbuf_xz, h_fbuf_xz, fbuf_size_xz, cudaMemcpyDefault), "cudaMemcpy")
  CUDA_CHECK(cudaMemcpyAsync(d_fbuf_xy, h_fbuf_xy, fbuf_size_xy, cudaMemcpyHostToDevice, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(d_fbuf_yz, h_fbuf_yz, fbuf_size_yz, cudaMemcpyHostToDevice, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(d_fbuf_xz, h_fbuf_xz, fbuf_size_xz, cudaMemcpyHostToDevice, cuda_stream), "cudaMemcpyAsync")
  threads.x = fsX; blocks.x = fsY;
  h_in_Kernel_xy<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xy, d0[hx], d0[hy], d0[hz], sX, sY, sZ, sXY); // <<<blocks,  threads, 0, cuda_stream>>>
  threads.x = fsY; blocks.x = fsZ;
  h_in_Kernel_yz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_yz, d0[hx], d0[hy], d0[hz], sX, sY, sZ, sXY);
  threads.x = fsX; blocks.x = fsZ;
  h_in_Kernel_xz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xz, d0[hx], d0[hy], d0[hz], sX, sY, sZ, sXY);
}
void CCudaEH3d::cuda_h_out()
{
  cudaError error_id;
  dim3 blocks, threads;
  cudaSetDevice(cuda_device);
  threads.x = fsX; blocks.x = fsY;
  h_out_Kernel_xy<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xy, d0[hx], d0[hy], d0[hz], sX, sY, sZ, sXY);
  threads.x = fsY; blocks.x = fsZ;
  h_out_Kernel_yz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_yz, d0[hx], d0[hy], d0[hz], sX, sY, sZ, sXY);
  threads.x = fsX; blocks.x = fsZ;
  h_out_Kernel_xz<<<blocks,  threads, 0, cuda_stream>>>(d_fbuf_xz, d0[hx], d0[hy], d0[hz], sX, sY, sZ, sXY);
//  CUDA_CHECK(cudaMemcpy(h_fbuf_xy, d_fbuf_xy, fbuf_size_xy, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(h_fbuf_yz, d_fbuf_yz, fbuf_size_yz, cudaMemcpyDefault), "cudaMemcpy")
//  CUDA_CHECK(cudaMemcpy(h_fbuf_xz, d_fbuf_xz, fbuf_size_xz, cudaMemcpyDefault), "cudaMemcpy")
  CUDA_CHECK(cudaMemcpyAsync(h_fbuf_xy, d_fbuf_xy, fbuf_size_xy, cudaMemcpyDeviceToHost, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(h_fbuf_yz, d_fbuf_yz, fbuf_size_yz, cudaMemcpyDeviceToHost, cuda_stream), "cudaMemcpyAsync")
  CUDA_CHECK(cudaMemcpyAsync(h_fbuf_xz, d_fbuf_xz, fbuf_size_xz, cudaMemcpyDeviceToHost, cuda_stream), "cudaMemcpyAsync")
}

void CCudaEH3d::update_e()
{
  const dim3 threads(cuda_tx,  cuda_ty, cuda_tz);
  const dim3 blocks(cuda_bx, cuda_by, cuda_bz);

  cudaSetDevice(cuda_device);
  update_e_Kernel<<<blocks, threads, 0, cuda_stream>>>
	(d0[ex], d0[ey], d0[ez], d0[hx], d0[hy], d0[hz],
	 d0[cexe], d0[ceye], d0[ceze], d0[cexh], d0[ceyh], d0[cezh], sX, sXY);
//  cudaMemcpy(d1_cell, d0_cell, buf_size * sizeof(yee_cell), cudaMemcpyDefault); // device -> device
}

void CCudaEH3d::update_h()
{
  const dim3 threads(cuda_tx,  cuda_ty, cuda_tz);
  const dim3 blocks(cuda_bx, cuda_by, cuda_bz);

  cudaSetDevice(cuda_device);
  update_h_Kernel<<<blocks, threads, 0, cuda_stream>>>
	(d0[hx], d0[hy], d0[hz], d0[ex], d0[ey], d0[ez],
	 d0[chxh], d0[chyh], d0[chzh], d0[chxe], d0[chye], d0[chze], sX, sXY);
}

void CCudaEH3d::get_eh(int m) // retrieve field data from devices
{
  cudaError error_id;

  cudaSetDevice(cuda_device);
  CUDA_CHECK(cudaMemcpy(h_buf, d0[m], buf_size, cudaMemcpyDefault), "cudaMemcpy")

  //CUDA_CHECK(cudaMemcpyAsync(h_buf, d0[m], buf_size, cudaMemcpyDeviceToHost, 0), "cudaMemcpy")
  //cudaDeviceSynchronize();
}

void CCudaEH3d::checkDeviceInfo()
{
  //int driver, runtime;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cuda_device);
  cout << node << "             device/model: " << cuda_device << "/" << prop.name << endl;
  //cout << "              capability: " << prop.major << "." << prop.minor << endl;
  //cudaDriverGetVersion(&driver);
  //cout << "                  driver: " << driver/1000 << "." << driver%100 << endl;
  //cudaRuntimeGetVersion(&runtime);
  //cout << "                 runtime: " << runtime/1000 << "." << runtime%100 << endl;
  //cout << "        pci bus/location: " << prop.pciBusID << "/" << prop.pciDeviceID << endl;
  //cout << "           global memory: " << prop.totalGlobalMem << endl;
  //cout << "     shared memory/block: " << prop.sharedMemPerBlock << endl;
  //cout << "  integrated host memory: " << (prop.integrated ? "Yes" : "No") << endl;
  //cout << "         map host memory: " << (prop.canMapHostMemory ? "Yes" : "No") << endl;
  //cout << "      unified addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << endl;
  //cout << "    multiprocessor count: " << prop.multiProcessorCount << endl;
  //cout << "        clock rate (kHz): " << prop.clockRate << endl;
}
