/*
 * spf_executors_uninc.h: This file is part of the SPF project.
 *
 * SPF: the Sparse Polyhedral Format
 *
 * Copyright (C) 2023 UDC and CSU
 *
 * This program can be redistributed and/or modified under the terms
 * of the license specified in the LICENSE.txt file at the root of the
 * project.
 *
 * Contact: Gabriel Rodriguez Alvarez <gabriel.rodriguez@udc.es>
 *          Louis-Noel Pouchet <pouchet@colostate.edu>
 *
 */

#ifndef _SPF_EXECUTORS_UNINC_H
# define _SPF_EXECUTORS_UNINC_H

#include <spf_structure.h>

extern
int
spf_executors_spf_matrix_dense_vector_product_uninc
(s_spf_structure_t* restrict spf_matrix,
 void* restrict x,
 void* restrict y,
 int ncols,
 int nrows,
 int data_is_float);

extern
int
spf_executors_spf_matrix_dense_matrix_product_uninc
  (const s_spf_structure_t* restrict spf_matrix,
   const void* restrict B,
   void* restrict C,
   const int ni,
   const int nj,
   const int nk,
   const int data_is_float,
   const int TJ);


#endif // !_SPF_EXECUTORS_UNINC_H
