/*
 * spf_executors.c: This file is part of the SPF project.
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

#ifdef GEN_EXECUTOR_SPMV_V2
#include <generated_shapes.h>
#endif


#define shape_to_reg_1x1(shape_ptr, shape_reg) \
  s_codelet_1x1d_t shape_reg = *((s_codelet_1x1d_t*)shape_ptr);
#define shape_to_reg_2x1(shape_ptr, shape_reg) \
  s_codelet_2x1d_t shape_reg = *((s_codelet_2x1d_t*)shape_ptr);
#define shape_to_reg_3x1(shape_ptr, shape_reg) \
  s_codelet_3x1d_t shape_reg = *((s_codelet_3x1d_t*)shape_ptr);
#define shape_to_reg_1x2(shape_ptr, shape_reg) \
  s_codelet_1x2d_t shape_reg = *((s_codelet_1x2d_t*)shape_ptr);
#define shape_to_reg_2x2(shape_ptr, shape_reg) \
  s_codelet_2x2d_t shape_reg = *((s_codelet_2x2d_t*)shape_ptr);
#define shape_to_reg_3x2(shape_ptr, shape_reg) \
  s_codelet_3x2d_t shape_reg = *((s_codelet_3x2d_t*)shape_ptr);
#define shape_to_reg_1x3(shape_ptr, shape_reg) \
  s_codelet_1x3d_t shape_reg = *((s_codelet_1x3d_t*)shape_ptr);
#define shape_to_reg_2x3(shape_ptr, shape_reg) \
  s_codelet_2x3d_t shape_reg = *((s_codelet_2x3d_t*)shape_ptr);
#define shape_to_reg_3x3(shape_ptr, shape_reg) \
  s_codelet_3x3d_t shape_reg = *((s_codelet_3x3d_t*)shape_ptr);

static
inline
void run_shape_o1d_double(s_spf_structure_t* restrict spf_matrix,
			  s_origin_1d_t orig,
			  double* restrict x,
			  double* restrict y)
{
  printf ("this case makes no sense: 1d origins? we don't compact sparse vectors!\n");
  exit (144);
}

static
inline
void run_shape_o1d_float(s_spf_structure_t* restrict spf_matrix,
			 s_origin_1d_t orig,
			 float* restrict x,
			 float* restrict y)
{
  printf ("this case makes no sense: 1d origins? we don't compact sparse vectors!\n");
  exit (144);
}




#define loop_body_s1_gen(nbiters)			\
  for (int i = 0; i < nbiters; i += 1)			\
    {							\
      /* Compute. */					\
      y[idx_y] += data_vector[a_data_pos++] * x[idx_x];	\
      idx_y += lattice_0;				\
      idx_x += lattice_1;				\
    }

#define loop_body_s1_1x0(nbiters)				\
  for (int i = 0; i < nbiters; i += 1)				\
    {								\
      /* Compute. */						\
      y[idx_y++] += data_vector[a_data_pos++] * x[idx_x];	\
    }

#define loop_body_s1_0x1(nbiters)				\
  for (int i = 0; i < nbiters; i += 1)				\
    {								\
      /* Compute. */						\
      y[idx_y] += data_vector[a_data_pos++] * x[idx_x++];	\
    }

#define loop_body_s1_1x1(nbiters)                                      \
  for (int i = 0; i < nbiters; i += 1)                                 \
    {                                                                  \
      /* Compute. */                                                   \
      y[idx_y++] += data_vector[a_data_pos++] * x[idx_x++];            \
    }


#define loop_nbiter_case(nbiters)			\
      case nbiters:					\
         {						\
           if (lattice_0 == 1 && lattice_1 == 0)	\
  	     { loop_body_s1_1x0(nbiters); }		\
           else if (lattice_0 == 0 && lattice_1 == 1)	\
  	     { loop_body_s1_0x1(nbiters); }		\
           else if (lattice_0 == 1 && lattice_1 == 1)	\
  	     { loop_body_s1_1x1(nbiters); }		\
           else						\
  	     { loop_body_s1_gen(nbiters); }		\
	   break;					\
         }


#define loop_body_s1_1x0_mm(nbiters,datatype)		\
  for (int i = 0; i < nbiters; i += 1)			\
    {							\
      /* Compute. */					\
      datatype A_val = data_vector[a_data_pos++];	\
      for (int j = Tj; j < TjUB; ++j)			\
	((datatype*)C)[idx_y * nj + j] +=		\
	  A_val * ((datatype*)B)[idx_x * nj + j];	\
      idx_y++;						\
    }

#define loop_body_s1_0x1_mm(nbiters,datatype)		\
  for (int i = 0; i < nbiters; i += 1)			\
    {							\
      /* Compute. */					\
      datatype A_val = data_vector[a_data_pos++];	\
      for (int j = Tj; j < TjUB; ++j)			\
	((datatype*)C)[idx_y * nj + j] +=		\
	  A_val * ((datatype*)B)[idx_x * nj + j];	\
      idx_x++;						\
    }

#define loop_body_s1_1x1_mm(nbiters, datatype)		\
  for (int i = 0; i < nbiters; i += 1)			\
    {							\
      /* Compute. */					\
      datatype A_val = data_vector[a_data_pos++];	\
      for (int j = Tj; j < TjUB; ++j)			\
	((datatype*)C)[idx_y * nj + j] +=		\
	  A_val * ((datatype*)B)[idx_x * nj + j];	\
      idx_y++;						\
      idx_x++;						\
    }




#define key_loop_case_s1_0x1(key, nbiters)	\
	case key:				\
	  {					\
	    loop_body_s1_0x1(nbiters);		\
	    return;				\
	  }

#define key_loop_case_s1_1x0(key, nbiters)	\
	case key:				\
	  {					\
	    loop_body_s1_1x0(nbiters);		\
	    return;				\
	  }

#define key_loop_case_s1_1x1(key, nbiters)	\
	case key:				\
	  {					\
	    loop_body_s1_1x1(nbiters);		\
	    return;				\
	  }

#define key_loop_case_s1_0x1_mm(datatype, key, nbiters)	\
	case key:					\
	  {						\
	    loop_body_s1_0x1_mm(nbiters, datatype);	\
	    return;					\
	  }

#define key_loop_case_s1_1x0_mm(datatype, key, nbiters)	\
	case key:					\
	  {						\
	    loop_body_s1_1x0_mm(nbiters, datatype);	\
	    return;					\
	  }

#define key_loop_case_s1_1x1_mm(datatype, key, nbiters)	\
	case key:					\
	  {						\
	    loop_body_s1_1x1_mm(nbiters, datatype);	\
	    return;					\
	  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------- Original Gen spmv executor ---------------------------------//
#ifdef GEN_EXECUTOR_SPMV_ORIGINAL
#define fundecl_run_shape_o2d_multitype(datatypename)                     \
static                                                                    \
inline                                                                    \
void run_shape_o2d_##datatypename(s_spf_structure_t* restrict spf_matrix, \
                                  s_origin_2d_t orig,                     \
                                  datatypename* restrict x,               \
                                  datatypename* restrict y)               \
{                                                                         \
    int a_data_pos = orig.dataptr;                                        \
    datatypename* const data_vector = spf_matrix->data;                   \
                                                                          \
    long long idx_y = orig.coordinates[0];                                \
    long long idx_x = orig.coordinates[1];                                \
                                                                          \
    switch (spf_matrix->shape_dictionnary.shapes_key[orig.shape_id])      \
    {                                                                     \
        key_loop_case_s1_0x1(1, 2);                                       \
        key_loop_case_s1_0x1(2, 3);                                       \
        key_loop_case_s1_0x1(3, 4);                                       \
        key_loop_case_s1_0x1(4, 5);                                       \
        key_loop_case_s1_0x1(5, 6);                                       \
        key_loop_case_s1_0x1(6, 7);                                       \
        key_loop_case_s1_0x1(7, 8);                                       \
        key_loop_case_s1_0x1(8, 16);                                      \
        key_loop_case_s1_0x1(9, 32);                                      \
        key_loop_case_s1_0x1(10, 64);                                     \
        key_loop_case_s1_1x0(11, 2);                                      \
        key_loop_case_s1_1x0(12, 3);                                      \
        key_loop_case_s1_1x0(13, 4);                                      \
        key_loop_case_s1_1x0(14, 5);                                      \
        key_loop_case_s1_1x0(15, 6);                                      \
        key_loop_case_s1_1x0(16, 7);                                      \
        key_loop_case_s1_1x0(17, 8);                                      \
        key_loop_case_s1_1x0(18, 16);                                     \
        key_loop_case_s1_1x0(19, 32);                                     \
        key_loop_case_s1_1x0(20, 64);                                     \
        key_loop_case_s1_1x1(21, 2);					                  \
        key_loop_case_s1_1x1(22, 3);					                  \
        key_loop_case_s1_1x1(23, 4);					                  \
        key_loop_case_s1_1x1(24, 5);					                  \
        key_loop_case_s1_1x1(25, 6);					                  \
        key_loop_case_s1_1x1(26, 7);					                  \
        key_loop_case_s1_1x1(27, 8);					                  \
        key_loop_case_s1_1x1(28, 16);					                  \
        key_loop_case_s1_1x1(29, 32);					                  \
        key_loop_case_s1_1x1(30, 64);					                  \
    default:                                                              \
    {                                                                     \
        void* shape = spf_matrix->shape_dictionnary.shapes[orig.shape_id];\
        s_codelet_1x1d_t* as_1d = (s_codelet_1x1d_t*)shape;               \
        int dim_p = as_1d->dim_p;                                         \
                                                                          \
        switch (dim_p) {                                                  \
        case 1:                                                           \
        {                                                                 \
            shape_to_reg_2x1(shape, shape_reg);                           \
            unsigned int loop_i_lb = shape_reg.start_vertex[0];           \
            unsigned int loop_i_ub = shape_reg.end_vertex[0];             \
            unsigned int loop_i_stride = shape_reg.stride[0];             \
                                                                          \
            int lattice_0 = shape_reg.lattice[0][0];                      \
            int lattice_1 = shape_reg.lattice[0][1];                      \
                                                                          \
            int offset_idx_y = lattice_0 * loop_i_stride;                 \
            int offset_idx_x = lattice_1 * loop_i_stride;                 \
                                                                          \
            if (loop_i_lb)                                                \
            {                                                             \
                idx_y += lattice_0 * loop_i_lb;                           \
                idx_x += lattice_1 * loop_i_lb;                           \
            }                                                             \
                                                                          \
            if (loop_i_lb == 0 && loop_i_ub > 0) {                        \
                switch (loop_i_stride) {                                  \
                case 1:                                                   \
                {                                                         \
                  for (int i = 0; i <= loop_i_ub; i += 1)                 \
                  {                                                       \
                      /* Compute. */                                      \
                      y[idx_y] += data_vector[a_data_pos++] * x[idx_x];   \
                      idx_y += offset_idx_y;                              \
                      idx_x += offset_idx_x;                              \
                  }                                                       \
                  return;                                                 \
                }                                                         \
                case 2:                                                   \
                {                                                         \
                    for (int i = 0; i <= loop_i_ub; i += 2)               \
                    {                                                     \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                        idx_y += offset_idx_y;                            \
                        idx_x += offset_idx_x;                            \
                    }                                                     \
                    return;                                               \
                }                                                         \
                default:                                                  \
                {                                                         \
                    for (int i = 0; i <= loop_i_ub; i += loop_i_stride)   \
                    {                                                     \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                        idx_y += offset_idx_y;                            \
                        idx_x += offset_idx_x;                            \
                    }                                                     \
                    return;                                               \
                }                                                         \
                }                                                         \
            } else                                                        \
            {                                                             \
                for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
                {                                                         \
                    /* Compute. */                                        \
                    y[idx_y] += data_vector[a_data_pos++] * x[idx_x];     \
                    idx_y += offset_idx_y;                                \
                    idx_x += offset_idx_x;                                \
                }                                                         \
            }                                                             \
            return;                                                       \
        }                                                                 \
        case 2:                                                           \
        {                                                                 \
            shape_to_reg_2x2(shape, shape_reg);                           \
            unsigned int loop_i_lb = shape_reg.start_vertex[0];           \
            unsigned int loop_i_ub = shape_reg.end_vertex[0];             \
            unsigned int loop_i_stride = shape_reg.stride[0];             \
                                                                          \
            unsigned int loop_j_lb = shape_reg.start_vertex[1];           \
            unsigned int loop_j_ub = shape_reg.end_vertex[1];             \
            unsigned int loop_j_stride = shape_reg.stride[1];             \
                                                                          \
            int lattice_0_0 = shape_reg.lattice[0][0];                    \
            int lattice_0_1 = shape_reg.lattice[0][1];                    \
            int lattice_1_0 = shape_reg.lattice[1][0];                    \
            int lattice_1_1 = shape_reg.lattice[1][1];                    \
                                                                          \
            int offset_idx_y = lattice_1_0 * loop_j_stride;               \
            int offset_idx_x = lattice_1_1 * loop_j_stride;               \
                                                                          \
            if (loop_i_ub > loop_i_lb && loop_j_ub > loop_j_lb &&         \
                loop_i_stride > 0 && loop_j_stride > 0)                   \
            {                                                             \
                if (loop_j_stride == 1 && loop_j_ub > 1 && loop_j_lb == 0)\
                for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
                {                                                         \
                    idx_y = orig.coordinates[0] + i * lattice_0_0;        \
                    idx_x = orig.coordinates[1] + i * lattice_0_1;        \
                    for (int j = 0; j <= loop_j_ub; j += 1)               \
                    {                                                     \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                        idx_y += offset_idx_y;                            \
                        idx_x += offset_idx_x;                            \
                    }                                                     \
                }                                                         \
                else                                                      \
                {                                                         \
                    for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
                    {                                                     \
                        idx_y = orig.coordinates[0] + i * lattice_0_0     \
                            + lattice_1_0 * loop_j_lb;                    \
                        idx_x = orig.coordinates[1] + i * lattice_0_1     \
                            + lattice_1_1 * loop_j_lb;                    \
                        for (int j = loop_j_lb; j <= loop_j_ub; j += loop_j_stride) \
                        {                                                 \
                            /* Compute. */                                \
                            y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                            idx_y += offset_idx_y;                        \
                            idx_x += offset_idx_x;                        \
                        }                                                 \
                    }                                                     \
                }                                                         \
            }                                                             \
            return;                                                       \
        }                                                                 \
        case 3:                                                           \
        {                                                                 \
            shape_to_reg_2x3(shape, shape_reg);                           \
            for (int i = shape_reg.start_vertex[0]; i <= shape_reg.end_vertex[0]; \
                i += shape_reg.stride[0])                                 \
            {                                                             \
                for (int j = shape_reg.start_vertex[1]; j <= shape_reg.end_vertex[1]; \
                    j += shape_reg.stride[1])                             \
                {                                                         \
                    for (int k = shape_reg.start_vertex[2];               \
                        k <= shape_reg.end_vertex[2];                     \
                        k += shape_reg.stride[2])                         \
                    {                                                     \
                        long long idx_y = orig.coordinates[0]             \
                            + shape_reg.lattice[0][0] * i                 \
                            + shape_reg.lattice[1][0] * j                 \
                            + shape_reg.lattice[2][0] * k;                \
                        long long idx_x = orig.coordinates[1]             \
                            + shape_reg.lattice[0][1] * i                 \
                            + shape_reg.lattice[1][1] * j                 \
                            + shape_reg.lattice[2][1] * k;                \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                    }                                                     \
                }                                                         \
            }                                                             \
            return;                                                       \
        }                                                                 \
        default:                                                          \
            exit (142);                                                   \
        }                                                                 \
    }                                                                     \
    }                                                                     \
}
#endif



//------------------------ Gen ex-spmv v1 -------------------------------//
// Aggregate the consecutive co-ordiantes with in the generated shapes of spf
#ifdef GEN_EXECUTOR_SPMV_V1
#define fundecl_run_shape_o2d_multitype(datatypename)                     \
static                                                                    \
inline                                                                    \
void run_shape_o2d_##datatypename(s_spf_structure_t* restrict spf_matrix, \
                                  s_origin_2d_t orig,                     \
                                  datatypename* restrict x,               \
                                  datatypename* restrict y)               \
{                                                                         \
    int a_data_pos = orig.dataptr;                                        \
    datatypename* const data_vector = spf_matrix->data;                   \
                                                                          \
    long long idx_y = orig.coordinates[0];                                \
    long long idx_x = orig.coordinates[1];                                \
                                                                          \
    switch (spf_matrix->shape_dictionnary.shapes_key[orig.shape_id])      \
    {                                                                     \
        key_loop_case_s1_0x1(1, 2);                                       \
        key_loop_case_s1_0x1(2, 3);                                       \
        key_loop_case_s1_0x1(3, 4);                                       \
        key_loop_case_s1_0x1(4, 5);                                       \
        key_loop_case_s1_0x1(5, 6);                                       \
        key_loop_case_s1_0x1(6, 7);                                       \
        key_loop_case_s1_0x1(7, 8);                                       \
        key_loop_case_s1_0x1(8, 16);                                      \
        key_loop_case_s1_0x1(9, 32);                                      \
        key_loop_case_s1_0x1(10, 64);                                     \
        key_loop_case_s1_1x0(11, 2);                                      \
        key_loop_case_s1_1x0(12, 3);                                      \
        key_loop_case_s1_1x0(13, 4);                                      \
        key_loop_case_s1_1x0(14, 5);                                      \
        key_loop_case_s1_1x0(15, 6);                                      \
        key_loop_case_s1_1x0(16, 7);                                      \
        key_loop_case_s1_1x0(17, 8);                                      \
        key_loop_case_s1_1x0(18, 16);                                     \
        key_loop_case_s1_1x0(19, 32);                                     \
        key_loop_case_s1_1x0(20, 64);                                     \
        key_loop_case_s1_1x1(21, 2);					                            \
        key_loop_case_s1_1x1(22, 3);					                            \
        key_loop_case_s1_1x1(23, 4);					                            \
        key_loop_case_s1_1x1(24, 5);					                            \
        key_loop_case_s1_1x1(25, 6);					                            \
        key_loop_case_s1_1x1(26, 7);					                            \
        key_loop_case_s1_1x1(27, 8);					                            \
        key_loop_case_s1_1x1(28, 16);					                            \
        key_loop_case_s1_1x1(29, 32);					                            \
        key_loop_case_s1_1x1(30, 64);					                            \
    default:                                                              \
    {                                                                     \
        void* shape = spf_matrix->shape_dictionnary.shapes[orig.shape_id];\
        s_codelet_1x1d_t* as_1d = (s_codelet_1x1d_t*)shape;               \
        int dim_p = as_1d->dim_p;                                         \
                                                                          \
        switch (dim_p) {                                                  \
        case 1:                                                           \
        {                                                                 \
            shape_to_reg_2x1(shape, shape_reg);                           \
            unsigned int loop_i_lb = shape_reg.start_vertex[0];           \
            unsigned int loop_i_ub = shape_reg.end_vertex[0];             \
            unsigned int loop_i_stride = shape_reg.stride[0];             \
                                                                          \
            int lattice_0 = shape_reg.lattice[0][0];                      \
            int lattice_1 = shape_reg.lattice[0][1];                      \
                                                                          \
            int offset_idx_y = lattice_0 * loop_i_stride;                 \
            int offset_idx_x = lattice_1 * loop_i_stride;                 \
                                                                          \
            if (loop_i_lb)                                                \
            {                                                             \
                idx_y += lattice_0 * loop_i_lb;                           \
                idx_x += lattice_1 * loop_i_lb;                           \
            }                                                             \
                                                                          \
            if (loop_i_lb == 0 && loop_i_ub > 0) {                        \
                switch (loop_i_stride) {                                  \
                case 1:                                                   \
                {                                                         \
                  if (lattice_0 == 1 && lattice_1 == 1){                  \
                    for (int i = 0; i <= loop_i_ub; ++i)                  \
                    {                                                     \
                      /* Compute. */                                      \
                      y[idx_y++] += data_vector[a_data_pos++] * x[idx_x++]; \
                    }                                                     \
                    return;                                               \
                  }                                                       \
                  else if (lattice_0 == 0 && lattice_1 == 1){             \
                    for (int i = 0; i <= loop_i_ub; ++i)                  \
                    {                                                     \
                      /* Compute. */                                      \
                      y[idx_y] += data_vector[a_data_pos++] * x[idx_x++]; \
                    }                                                     \
                    return;                                               \
                  }                                                       \
                  else if (lattice_0 == 1 && lattice_1 == 0){             \
                    for (int i = 0; i <= loop_i_ub; ++i)                  \
                    {                                                     \
                      /* Compute. */                                      \
                      y[idx_y++] += data_vector[a_data_pos++] * x[idx_x]; \
                    }                                                     \
                    return;                                               \
                  }                                                       \
																		  \
                  for (int i = 0; i <= loop_i_ub; i += 1)                 \
                  {                                                       \
                      /* Compute. */                                      \
                      y[idx_y] += data_vector[a_data_pos++] * x[idx_x];   \
                      idx_y += offset_idx_y;                              \
                      idx_x += offset_idx_x;                              \
                  }                                                       \
                  return;                                                 \
                }                                                         \
                case 2:                                                   \
                {                                                         \
                    for (int i = 0; i <= loop_i_ub; i += 2)               \
                    {                                                     \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                        idx_y += offset_idx_y;                            \
                        idx_x += offset_idx_x;                            \
                    }                                                     \
                    return;                                               \
                }                                                         \
                default:                                                  \
                {                                                         \
                    for (int i = 0; i <= loop_i_ub; i += loop_i_stride)   \
                    {                                                     \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                        idx_y += offset_idx_y;                            \
                        idx_x += offset_idx_x;                            \
                    }                                                     \
                    return;                                               \
                }                                                         \
                }                                                         \
            } else                                                        \
            {                                                             \
                for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
                {                                                         \
                    /* Compute. */                                        \
                    y[idx_y] += data_vector[a_data_pos++] * x[idx_x];     \
                    idx_y += offset_idx_y;                                \
                    idx_x += offset_idx_x;                                \
                }                                                         \
            }                                                             \
            return;                                                       \
        }                                                                 \
        case 2:                                                           \
        {                                                                 \
            shape_to_reg_2x2(shape, shape_reg);                           \
            unsigned int loop_i_lb = shape_reg.start_vertex[0];           \
            unsigned int loop_i_ub = shape_reg.end_vertex[0];             \
            unsigned int loop_i_stride = shape_reg.stride[0];             \
                                                                          \
            unsigned int loop_j_lb = shape_reg.start_vertex[1];           \
            unsigned int loop_j_ub = shape_reg.end_vertex[1];             \
            unsigned int loop_j_stride = shape_reg.stride[1];             \
                                                                          \
            int lattice_0_0 = shape_reg.lattice[0][0];                    \
            int lattice_0_1 = shape_reg.lattice[0][1];                    \
            int lattice_1_0 = shape_reg.lattice[1][0];                    \
            int lattice_1_1 = shape_reg.lattice[1][1];                    \
                                                                          \
            int offset_idx_y = lattice_1_0 * loop_j_stride;               \
            int offset_idx_x = lattice_1_1 * loop_j_stride;               \
                                                                          \
            if (loop_i_ub > loop_i_lb && loop_j_ub > loop_j_lb &&         \
                loop_i_stride > 0 && loop_j_stride > 0)                   \
            {                                                             \
                if (loop_j_stride == 1 && loop_j_ub > 1 && loop_j_lb == 0)\
                for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
                {                                                         \
                    idx_y = orig.coordinates[0] + i * lattice_0_0;        \
                    idx_x = orig.coordinates[1] + i * lattice_0_1;        \
                    for (int j = 0; j <= loop_j_ub; j += 1)               \
                    {                                                     \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                        idx_y += offset_idx_y;                            \
                        idx_x += offset_idx_x;                            \
                    }                                                     \
                }                                                         \
                else                                                      \
                {                                                         \
                    for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
                    {                                                     \
                        idx_y = orig.coordinates[0] + i * lattice_0_0     \
                            + lattice_1_0 * loop_j_lb;                    \
                        idx_x = orig.coordinates[1] + i * lattice_0_1     \
                            + lattice_1_1 * loop_j_lb;                    \
                        for (int j = loop_j_lb; j <= loop_j_ub; j += loop_j_stride) \
                        {                                                 \
                            /* Compute. */                                \
                            y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                            idx_y += offset_idx_y;                        \
                            idx_x += offset_idx_x;                        \
                        }                                                 \
                    }                                                     \
                }                                                         \
            }                                                             \
            return;                                                       \
        }                                                                 \
        case 3:                                                           \
        {                                                                 \
            shape_to_reg_2x3(shape, shape_reg);                           \
            for (int i = shape_reg.start_vertex[0]; i <= shape_reg.end_vertex[0]; \
                i += shape_reg.stride[0])                                 \
            {                                                             \
                for (int j = shape_reg.start_vertex[1]; j <= shape_reg.end_vertex[1]; \
                    j += shape_reg.stride[1])                             \
                {                                                         \
                    for (int k = shape_reg.start_vertex[2];               \
                        k <= shape_reg.end_vertex[2];                     \
                        k += shape_reg.stride[2])                         \
                    {                                                     \
                        long long idx_y = orig.coordinates[0]             \
                            + shape_reg.lattice[0][0] * i                 \
                            + shape_reg.lattice[1][0] * j                 \
                            + shape_reg.lattice[2][0] * k;                \
                        long long idx_x = orig.coordinates[1]             \
                            + shape_reg.lattice[0][1] * i                 \
                            + shape_reg.lattice[1][1] * j                 \
                            + shape_reg.lattice[2][1] * k;                \
                        /* Compute. */                                    \
                        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
                    }                                                     \
                }                                                         \
            }                                                             \
            return;                                                       \
        }                                                                 \
        default:                                                          \
            exit (142);                                                   \
        }                                                                 \
    }                                                                     \
    }                                                                     \
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define fundecl_run_shape_o2d_matmult_multitype(datatypename)		\
static									\
inline									\
void run_shape_o2d_mm_##datatypename(s_spf_structure_t* restrict spf_matrix, \
			  s_origin_2d_t orig,				\
			  datatypename* restrict B,			\
			  datatypename* restrict C,			\
			  int ni, int nj, int nk, int Tj, int TjUB)	\
{									\
  int a_data_pos = orig.dataptr;					\
  datatypename* const data_vector = spf_matrix->data;			\
									\
  long long idx_y = orig.coordinates[0];					\
  long long idx_x = orig.coordinates[1];					\
									\
  switch (spf_matrix->shape_dictionnary.shapes_key[orig.shape_id])	\
    {									\
      key_loop_case_s1_0x1_mm(datatypename, 1, 2);			\
      key_loop_case_s1_0x1_mm(datatypename, 2, 3);			\
      key_loop_case_s1_0x1_mm(datatypename, 3, 4);			\
      key_loop_case_s1_0x1_mm(datatypename, 4, 5);			\
      key_loop_case_s1_0x1_mm(datatypename, 5, 6);			\
      key_loop_case_s1_0x1_mm(datatypename, 6, 7);			\
      key_loop_case_s1_0x1_mm(datatypename, 7, 8);			\
      key_loop_case_s1_1x0_mm(datatypename, 8, 2);			\
      key_loop_case_s1_1x0_mm(datatypename, 9, 3);			\
      key_loop_case_s1_1x0_mm(datatypename, 10, 4);			\
      key_loop_case_s1_1x0_mm(datatypename, 11, 5);			\
      key_loop_case_s1_1x0_mm(datatypename, 12, 6);			\
      key_loop_case_s1_1x0_mm(datatypename, 13, 7);			\
      key_loop_case_s1_1x0_mm(datatypename, 14, 8);			\
      key_loop_case_s1_1x1_mm(datatypename, 15, 2);			\
      key_loop_case_s1_1x1_mm(datatypename, 16, 3);			\
      key_loop_case_s1_1x1_mm(datatypename, 17, 4);			\
      key_loop_case_s1_1x1_mm(datatypename, 18, 5);			\
      key_loop_case_s1_1x1_mm(datatypename, 19, 6);			\
      key_loop_case_s1_1x1_mm(datatypename, 20, 7);			\
      key_loop_case_s1_1x1_mm(datatypename, 21, 8);			\
    default:								\
      {									\
	void* shape = spf_matrix->shape_dictionnary.shapes[orig.shape_id]; \
	s_codelet_1x1d_t* as_1d = (s_codelet_1x1d_t*)shape;		\
	int dim_p = as_1d->dim_p;					\
									\
	switch (dim_p) {						\
	case 1:								\
	  {								\
	    shape_to_reg_2x1(shape, shape_reg);				\
	    unsigned int loop_i_lb = shape_reg.start_vertex[0];		\
	    unsigned int loop_i_ub = shape_reg.end_vertex[0];		\
	    unsigned int loop_i_stride = shape_reg.stride[0];		\
									\
	    int lattice_0 = shape_reg.lattice[0][0];			\
	    int lattice_1 = shape_reg.lattice[0][1];			\
									\
	    int offset_idx_y = lattice_0 * loop_i_stride;		\
	    int offset_idx_x = lattice_1 * loop_i_stride;		\
									\
	    if (loop_i_lb)						\
	      {								\
		idx_y += lattice_0 * loop_i_lb;				\
		idx_x += lattice_1 * loop_i_lb;				\
	      }								\
									\
	    if (loop_i_lb == 0 && loop_i_ub > 0) {			\
	      switch (loop_i_stride) {					\
              default:							\
		{							\
	      for (int i = 0; i <= loop_i_ub; i += loop_i_stride)	\
	       {							\
		 datatypename A_val = data_vector[a_data_pos++];	\
		 for (int j = Tj; j < TjUB; ++j)			\
		   ((datatypename*)C)[idx_y * nj + j] += A_val * ((datatypename*)B)[idx_x * nj + j]; \
		idx_y += offset_idx_y;					\
		idx_x += offset_idx_x;					\
	       }							\
	      return;							\
		}							\
		} }							\
	     else							\
	      for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
	        {							\
		/* Compute. */						\
		 datatypename A_val = data_vector[a_data_pos++];	\
		 for (int j = Tj; j < TjUB; ++j)			\
		   ((datatypename*)C)[idx_y * nj + j] += A_val * ((datatypename*)B)[idx_x * nj + j]; \
		idx_y += offset_idx_y;					\
		idx_x += offset_idx_x;					\
	        }							\
	    return;							\
	  }								\
	case 2:								\
	  {								\
	    shape_to_reg_2x2(shape, shape_reg);				\
	    unsigned int loop_i_lb = shape_reg.start_vertex[0];		\
	    unsigned int loop_i_ub = shape_reg.end_vertex[0];		\
	    unsigned int loop_i_stride = shape_reg.stride[0];		\
									\
	    unsigned int loop_j_lb = shape_reg.start_vertex[1];		\
	    unsigned int loop_j_ub = shape_reg.end_vertex[1];		\
	    unsigned int loop_j_stride = shape_reg.stride[1];		\
									\
	    int lattice_0_0 = shape_reg.lattice[0][0];			\
	    int lattice_0_1 = shape_reg.lattice[0][1];			\
	    int lattice_1_0 = shape_reg.lattice[1][0];			\
	    int lattice_1_1 = shape_reg.lattice[1][1];			\
									\
	    int offset_idx_y = lattice_1_0 * loop_j_stride;		\
	    int offset_idx_x = lattice_1_1 * loop_j_stride;		\
									\
	    if (loop_i_ub > loop_i_lb && loop_j_ub > loop_j_lb &&	\
		loop_i_stride > 0 && loop_j_stride > 0)			\
	      {								\
		if (loop_j_stride == 1 && loop_j_ub > 1 && loop_j_lb == 0) \
		  for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride) \
		    {							\
		      idx_y = orig.coordinates[0] + i * lattice_0_0;	\
		      idx_x = orig.coordinates[1] + i * lattice_0_1;	\
		      for (int j = 0; j <= loop_j_ub; j += 1)		\
			{						\
			  /* Compute. */				\
			  datatypename A_val = data_vector[a_data_pos++]; \
			  for (int k = Tj; k < TjUB; ++k)		\
			    ((datatypename*)C)[idx_y * nj + k] += A_val * ((datatypename*)B)[idx_x * nj + k]; \
			  idx_y += offset_idx_y;			\
			  idx_x += offset_idx_x;			\
			}						\
		    }							\
		else							\
		  {							\
		    for (int i = loop_i_lb; i <= loop_i_ub; i += loop_i_stride)	\
		      {							\
			idx_y = orig.coordinates[0] + i * lattice_0_0	\
			  + lattice_1_0 * loop_j_lb;			\
			idx_x = orig.coordinates[1] + i * lattice_0_1	\
			  + lattice_1_1 * loop_j_lb;			\
			for (int j = loop_j_lb; j <= loop_j_ub; j += loop_j_stride) \
			  {						\
			  /* Compute. */				\
			    datatypename A_val = data_vector[a_data_pos++]; \
			    for (int k = Tj; k < TjUB; ++k)		\
			    ((datatypename*)C)[idx_y * nj + k] += A_val * ((datatypename*)B)[idx_x * nj + k]; \
			    idx_y += offset_idx_y;			\
			    idx_x += offset_idx_x;			\
			  }						\
		      }							\
		  }							\
	      }								\
	    return;							\
	  }								\
	case 3:								\
	  {								\
	    shape_to_reg_2x3(shape, shape_reg);				\
	    fprintf (stderr, "3D shapes for MM Case not implemented!\n"); \
	    exit (198);							\
	    return;							\
	  }								\
	default:							\
	  exit (142);							\
	}								\
      }									\
    }									\
}



fundecl_run_shape_o2d_multitype(double)

fundecl_run_shape_o2d_multitype(float)


fundecl_run_shape_o2d_matmult_multitype(double)

fundecl_run_shape_o2d_matmult_multitype(float)


int
spf_executors_spf_matrix_dense_vector_product(s_spf_structure_t* restrict spf_matrix,
					      void* restrict x,
					      void* restrict y,
					      int ncols,
					      int nrows,
					      int data_is_float)
{
  if (!spf_matrix || !x || !y)
    return -1;

  int bypass_codelets = 0;

  // 1. Iterate on all origins.
  int nb_origins = spf_matrix->origins_list.norigins;
  int dim_o = spf_matrix->matrix_description.dim_o;

  if (!bypass_codelets)
    {
      if (dim_o == 1)
	{
	  if (data_is_float)
	    {
	      #pragma omp parallel for
	      for (int i = 0; i < nb_origins; ++i)
		{
		  s_origin_1d_t orig =
		    ((s_origin_1d_t*)spf_matrix->origins_list.origins)[i];
		  run_shape_o1d_float(spf_matrix, orig, x, y);
		}
	    }
	  else
	    {
	      #pragma omp parallel for
	      for (int i = 0; i < nb_origins; ++i)
		{
		  s_origin_1d_t orig =
		    ((s_origin_1d_t*)spf_matrix->origins_list.origins)[i];
		  run_shape_o1d_double(spf_matrix, orig, x, y);
		}

	    }
	}
      else if (dim_o == 2)
	{
	  if (data_is_float)
	    {
	      #pragma omp parallel for
	      for (int i = 0; i < nb_origins; ++i)
		{
		  s_origin_2d_t orig =
		    ((s_origin_2d_t*)spf_matrix->origins_list.origins)[i];
		  run_shape_o2d_float(spf_matrix, orig, x, y);
		}
	    }
	  else
	    {
	      #pragma omp parallel for
	      for (int i = 0; i < nb_origins; ++i)
		{
		  s_origin_2d_t orig =
		    ((s_origin_2d_t*)spf_matrix->origins_list.origins)[i];
		  run_shape_o2d_double(spf_matrix, orig, x, y);
		}
	    }
	}
      else if (dim_o == 3)
	{
	  /* for (int i = 0; i < nb_origins; ++i) */
	  /* 	{ */
	  /* 	  s_origin_3d_t orig = */
	  /* 	    ((s_origin_3d_t*)spf_matrix->origins_list.origins)[i]; */
	  /* 	  run_shape_o3d(spf_matrix, orig, x, y); */
	  /* 	} */
	  printf ("unsupported\n");
	  exit (155);
	}
    }

  return spf_executors_spf_matrix_dense_vector_product_uninc
    (spf_matrix, x, y, ncols, nrows, data_is_float);
}



int
spf_executors_spf_matrix_dense_matrix_product(s_spf_structure_t* restrict spf_matrix,
					      void* restrict B, // matrix
					      void* restrict C, // matrix
					      int ni,
					      int nj,
					      int nk,
					      int data_is_float,
					      int TJ)
{
  if (!spf_matrix || !C || !B)
    return -1;

  int bypass_codelets = 0;

  // 1. Iterate on all origins.
  int nb_origins = spf_matrix->origins_list.norigins;
  int dim_o = spf_matrix->matrix_description.dim_o;

  if (!bypass_codelets)
    {
      if (dim_o == 1)
      	{
      	  printf ("unsupported\n");
      	  exit (156);
      	}
      else
	if (dim_o == 2)
	{
	  if (data_is_float)
	    {
	      #pragma omp parallel for
	      for (int Tj = 0; Tj < nj; Tj += TJ)
		{
		  int loop_ub = Tj + TJ < nj ? Tj + TJ : nj;
		  for (int i = 0; i < nb_origins; ++i)
		    {
		      s_origin_2d_t orig =
			((s_origin_2d_t*)spf_matrix->origins_list.origins)[i];
		      run_shape_o2d_mm_float(spf_matrix, orig, B, C,
					     ni, nj, nk, Tj, loop_ub);
		    }
		}
	    }
	  else
	    {
	      #pragma omp parallel for
	      for (int Tj = 0; Tj < nj; Tj += TJ)
		{
		  int loop_ub = Tj + TJ < nj ? Tj + TJ : nj;
		  for (int i = 0; i < nb_origins; ++i)
		    {
		      s_origin_2d_t orig =
			((s_origin_2d_t*)spf_matrix->origins_list.origins)[i];
		      run_shape_o2d_mm_double(spf_matrix, orig, B, C,
					      ni, nj, nk, Tj, loop_ub);
		    }
		}
	    }
	}
      else if (dim_o == 3)
      	{
      	  printf ("unsupported\n");
      	  exit (155);
      	}
    }
  return spf_executors_spf_matrix_dense_matrix_product_uninc
    (spf_matrix, B, C, ni, nj, nk, data_is_float, TJ);
}
