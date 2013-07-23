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

#include "utils.h"

CTimer::CTimer() 
{
  set_start_time();
}

CTimer::~CTimer()
{
}

void CTimer::set_start_time(){
    clock_gettime(CLOCK_REALTIME, &start_time);
  }

double CTimer::get_elapsed_time(){
    timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    double result = time_diff(start_time, current_time);
    start_time = current_time;
    return result;
  }

double CTimer::time_diff(timespec start, timespec end)
{
  timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
	temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  } else {
  	temp.tv_sec = end.tv_sec - start.tv_sec;
  	temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return temp.tv_sec + temp.tv_nsec / 1000000000.0;
}

