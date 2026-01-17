/*
 * spf_executors_uninc.c: This file is part of the SPF project.
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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <assert.h>

#include <sys/stat.h>
#include <fcntl.h>


#include <spf_structure.h>
#include <spf_executors.h>
#include <spf_executors_uninc.h>


int
spf_executors_spf_matrix_dense_vector_product_uninc
  (s_spf_structure_t* restrict spf_matrix,
   void* restrict x,
   void* restrict y,
   int ncols,
   int nrows,
   int data_is_float)
{
  if (!spf_matrix || !x || !y)
    return -1;

  int bypass_csr = 0;
  int bypass_coo = 0;

  if (spf_matrix->matrix_description.inc_nnz !=
      spf_matrix->matrix_description.nnz)
    {
      int pos_data = spf_matrix->matrix_description.inc_nnz;
      if (spf_matrix->unincorporated_points.uninc_format == 0)
	{
	  if (!bypass_csr)
	    {
	      if (data_is_float)
		{
		  // CSR.
		  #pragma omp parallel for
		  for (int i = 0; i < nrows; ++i)
		    {
		      for (int j = spf_matrix->unincorporated_points.rowptr[i];
			   j < spf_matrix->unincorporated_points.rowptr[i + 1];
			   ++j)
			{
			  ((float*)y)[i] +=
			    ((float*)spf_matrix->data)[pos_data++] *
			    ((float*)x)[spf_matrix->unincorporated_points
					.colidx[j]];
			}
		    }
		}
	      else
		{
		  // CSR.
		  #pragma omp parallel for
		  for (int i = 0; i < nrows; ++i)
		    {
		      for (int j = spf_matrix->unincorporated_points.rowptr[i];
			   j < spf_matrix->unincorporated_points.rowptr[i + 1];
			   ++j)
			{
			  ((double*)y)[i] +=
			    ((double*)spf_matrix->data)[pos_data++] *
			    ((double*)x)[spf_matrix->unincorporated_points
					 .colidx[j]];
			}
		    }
		}
	    }
	}
      else if (spf_matrix->unincorporated_points.uninc_format == 2)
	{
	  if (! bypass_coo)
	    {
	      if (data_is_float)
		{
		  // COO.
		  int uninc_nnz =
		    spf_matrix->matrix_description.nnz -
		    spf_matrix->matrix_description.inc_nnz;
#pragma omp parallel for
		  for (int i = 0; i < uninc_nnz; ++i)
		    {
		      long long idx_y = spf_matrix->unincorporated_points.rowptr[i];
		      long long idx_x = spf_matrix->unincorporated_points.colidx[i];
		      ((float*)y)[idx_y] +=
			((float*)spf_matrix->data)[pos_data++] *
			((float*)x)[idx_x];
		    }
		}
	      else
		{
		  // COO.
		  int uninc_nnz =
		    spf_matrix->matrix_description.nnz -
		    spf_matrix->matrix_description.inc_nnz;
#pragma omp parallel for
		  for (int i = 0; i < uninc_nnz; ++i)
		    {
		      long long idx_y = spf_matrix->unincorporated_points.rowptr[i];
		      long long idx_x = spf_matrix->unincorporated_points.colidx[i];
		      ((double*)y)[idx_y] +=
			((double*)spf_matrix->data)[pos_data++] *
			((double*)x)[idx_x];
		    }
		}
	    }
	}
    }

  return 0;
}

#define min(a,b) ((a) < (b) ? (a) : (b))

#define STM(type,j) ((type*)C)[((long long)i) * nj + (j)] += A_val * ((type*)B)[((long long)colidx[k]) * nj + (j)];


#define loop_nest_csr_spmm(type)					\
      for (int Tj = 0; Tj < nj; Tj += TJ)				\
	{								\
	  int pos_data_cur = pos_data;					\
	  for (int k = 0; k < k_sz; ++k)				\
	    {								\
	      type A_val = ((type*)spf_matrix->data)[pos_data_cur++];	\
/*            GRA: removed apparent performance bug                      */\		    
/*	      if (TJ > 7 && Tj + TJ <= nj)				\*/\
/*		{							\*/\
/*		  int j;						\*/\
/*		  int loop_ub = Tj + TJ - 8;				\*/\
/*		  for (j = Tj; j < loop_ub; j += 8)			\*/\
/*		    {							\*/\
/*		      STM(type,j);					\*/\
/*		      STM(type,j+1);					\*/\
/*		      STM(type,j+2);					\*/\
/*		      STM(type,j+3);					\*/\
/*		      STM(type,j+4);					\*/\
/*		      STM(type,j+5);					\*/\
/*		      STM(type,j+6);					\*/\
/*		      STM(type,j+7);					\*/\
/*		    }							\*/\
/*		  for (; j < Tj + TJ; ++j)				\*/\
/*		    {							\*/\
/*		      STM(type,j);					\*/\
/*		    }							\*/\
/*		}							\*/\
/*	      else							\*/\
		{							\
		  for (int j = Tj; j < nj; ++j)				\
		    STM(type,j);					\
		}							\
	    }								\
	}


#define computation_csr_spmm(type)					\
      int k_lb = spf_matrix->unincorporated_points.rowptr[i];		\
      int k_ub = spf_matrix->unincorporated_points.rowptr[i + 1];	\
      int TJ = TJpar;							\
      int k_sz = k_ub - k_lb;						\
      if (k_sz == 0)							\
	continue;							\
      int colidx[k_sz];							\
      for (int k = spf_matrix->unincorporated_points.rowptr[i];		\
	   k < spf_matrix->unincorporated_points.rowptr[i + 1];		\
	   ++k)								\
	colidx[k - spf_matrix->unincorporated_points.rowptr[i]] =	\
	  spf_matrix->unincorporated_points.colidx[k];			\
      loop_nest_csr_spmm(type);



static
int
spf_executors_spf_matrix_dense_matrix_product_uninc_csr
  (const s_spf_structure_t* restrict spf_matrix,
   const void* restrict B,
   void* restrict C,
   const int ni,
   const int nj,
   const int nk,
   const int data_is_float,
   const int TJpar)
{
  int pos_data = spf_matrix->matrix_description.inc_nnz;

  if (data_is_float)
    {
      #pragma omp parallel for						
      for (int i = 0; i < ni; ++i)						
	{									
	  computation_csr_spmm(float);
        }
    }
  else
    {
      #pragma omp parallel for						
      for (int i = 0; i < ni; ++i)						
	{									
	  computation_csr_spmm(double);
        }
    }

  return 0;
}



int
spf_executors_spf_matrix_dense_matrix_product_uninc
  (const s_spf_structure_t* restrict spf_matrix,
   const void* restrict B,
   void* restrict C,
   const int ni,
   const int nj,
   const int nk,
   const int data_is_float,
   const int TJ)
{
  if (!spf_matrix || !B || !C)
    return -1;

  int bypass_csr = 0;
  int bypass_coo = 0;

  if (spf_matrix->matrix_description.inc_nnz ==
      spf_matrix->matrix_description.nnz)
    return 0;

  if (spf_matrix->unincorporated_points.uninc_format == 0)
    {
      if (!bypass_csr)
	{
	  return spf_executors_spf_matrix_dense_matrix_product_uninc_csr
	    (spf_matrix, B, C, ni, nj, nk, data_is_float, TJ);
	}
    }
  else if (spf_matrix->unincorporated_points.uninc_format == 2)
    {
      if (! bypass_coo)
	{
	  int pos_data = spf_matrix->matrix_description.inc_nnz;

	  // COO.
	  int uninc_nnz =
	    spf_matrix->matrix_description.nnz -
	    spf_matrix->matrix_description.inc_nnz;
	  if (data_is_float)
	    {
      #pragma omp parallel for						
	      for (int i = 0; i < uninc_nnz; ++i)
		{
		  long long idx_i = spf_matrix->unincorporated_points.rowptr[i];
		  long long idx_j = spf_matrix->unincorporated_points.colidx[i];
		  float A_val = ((float*)spf_matrix->data)[pos_data++];
		  for (int j = 0; j < nj; ++j)
		    ((float*)C)[idx_i * nj + j] += A_val *
		      ((float*)B)[idx_j * nj + j];
		}
	    }
	  else
	    {
      #pragma omp parallel for						
	      for (int i = 0; i < uninc_nnz; ++i)
		{
		  long long idx_i = spf_matrix->unincorporated_points.rowptr[i];
		  long long idx_j = spf_matrix->unincorporated_points.colidx[i];
		  double A_val = ((double*)spf_matrix->data)[pos_data++];
		  for (int j = 0; j < nj; ++j)
		    ((double*)C)[idx_i * nj + j] += A_val *
		      ((double*)B)[idx_j * nj + j];
		}
	    }
	}
    }

  return 0;
}
