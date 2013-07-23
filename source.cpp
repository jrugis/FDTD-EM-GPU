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

#include <math.h>

#include "defs.h"
#include "source.h"

void sourceGaussian(bool a, F_TYPE s, F_TYPE *eh, size_t ts, F_TYPE dts, F_TYPE nwtss)
{
  F_TYPE temp = (a ? *eh : 0.0); // additive?
  *eh = temp + s * exp( pow(ts - dts, 2) / nwtss);
}

void sourceSine(bool a, F_TYPE s, F_TYPE *eh, size_t ts, F_TYPE omega)
{
  F_TYPE temp = (a ? *eh : 0.0); // additive?
  *eh = temp + s * sin(omega * ts);
}

void sourceRicker(bool a, F_TYPE s, F_TYPE *eh, size_t ts, F_TYPE wts)
{
  F_TYPE temp = (a ? *eh : 0.0); // additive?
  F_TYPE arg = pow(2* PI * (ts / wts - 1.0), 2);
  *eh = temp + s * (1.0 - 2.0 * arg) * exp(-arg);
}
