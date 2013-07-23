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

#ifndef SOURCE_H
#define SOURCE_H

#include <stdlib.h>

#include "defs.h"

/*
  signal source arguments:
   1 additive
   2 scaling factor
   3 destination pointer
   4 time step
   5 other... (source specific)
*/

void sourceGaussian(bool additive, F_TYPE scaling, F_TYPE *eh, size_t time_step, F_TYPE dts, F_TYPE nwtss);
void sourceSine(bool additive, F_TYPE scaling, F_TYPE *eh, size_t time_step, F_TYPE omega);
void sourceRicker(bool additive, F_TYPE scaling, F_TYPE *eh, size_t time_step, F_TYPE wts);

#endif // SOURCE_H
