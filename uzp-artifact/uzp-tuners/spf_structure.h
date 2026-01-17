/*
 * spf_structure.h: This file is part of the SPF project.
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

#ifndef _SPF_STRUCTURE_H
# define _SPF_STRUCTURE_H

# include <stdio.h>

struct __attribute__((__packed__)) s_codelet_1x1d {
  int dim_p;
  int start_vertex[1];
  int end_vertex[1];
  int stride[1];
  int lattice[1][1];
};
typedef struct s_codelet_1x1d s_codelet_1x1d_t;

struct __attribute__((__packed__)) s_codelet_2x1d {
  int dim_p;
  int start_vertex[1];
  int end_vertex[1];
  int stride[1];
  int lattice[2][1];
};
typedef struct s_codelet_2x1d s_codelet_2x1d_t;

struct __attribute__((__packed__)) s_codelet_3x1d {
  int dim_p;
  int start_vertex[1];
  int end_vertex[1];
  int stride[1];
  int lattice[3][1];
};
typedef struct s_codelet_3x1d s_codelet_3x1d_t;

struct __attribute__((__packed__)) s_codelet_1x2d {
  int dim_p;
  int start_vertex[2];
  int end_vertex[2];
  int stride[2];
  int lattice[1][2];
};
typedef struct s_codelet_1x2d s_codelet_1x2d_t;

struct __attribute__((__packed__)) s_codelet_2x2d {
  int dim_p;
  int start_vertex[2];
  int end_vertex[2];
  int stride[2];
  int lattice[2][2];
};
typedef struct s_codelet_2x2d s_codelet_2x2d_t;

struct __attribute__((__packed__)) s_codelet_3x2d {
  int dim_p;
  int start_vertex[2];
  int end_vertex[2];
  int stride[2];
  int lattice[3][2];
};
typedef struct s_codelet_3x2d s_codelet_3x2d_t;

struct __attribute__((__packed__)) s_codelet_1x3d {
  int dim_p;
  int start_vertex[3];
  int end_vertex[3];
  int stride[3];
  int lattice[1][3];
};
typedef struct s_codelet_1x3d s_codelet_1x3d_t;

struct __attribute__((__packed__)) s_codelet_2x3d {
  int dim_p;
  int start_vertex[3];
  int end_vertex[3];
  int stride[3];
  int lattice[2][3];
};
typedef struct s_codelet_2x3d s_codelet_2x3d_t;

struct __attribute__((__packed__)) s_codelet_3x3d {
  int dim_p;
  int start_vertex[3];
  int end_vertex[3];
  int stride[3];
  int lattice[3][3];
};
typedef struct s_codelet_3x3d s_codelet_3x3d_t;


struct __attribute__((__packed__)) s_matrix_desc {
  int nnz;
  int inc_nnz;
  int intnrow;
  int ncols;
  short dim_o;
  int nbase_shapes;
  int nhierch_shapes;
  int data_start;
  short dim_p;
};
typedef struct s_matrix_desc s_matrix_desc_t;

// shapes[id] = malloc(shape_n_d)
struct s_shape_dictionnary {
  int		nbase_shapes;
  int*		nshapes_dim_p; // array of size dim_p, malloc'ed
  void**	shapes;
  int*		shapes_nb_origins;
  int*		shapes_key;
};
typedef struct s_shape_dictionnary s_shape_dictionnary_t;

struct __attribute__((__packed__)) s_origin_1d {
  short		shape_id;
  int		coordinates[1];
  int		dataptr;
};
typedef struct s_origin_1d s_origin_1d_t;

struct __attribute__((__packed__)) s_origin_2d {
  short		shape_id;
  int		coordinates[2];
  int		dataptr;
};
typedef struct s_origin_2d s_origin_2d_t;

struct __attribute__((__packed__)) s_origin_3d {
  short		shape_id;
  int		coordinates[3];
  int		dataptr;
};
typedef struct s_origin_3d s_origin_3d_t;


struct s_origin_list {
  short		dim_o;
  int		norigins;
  void*		origins;
};
typedef struct s_origin_list s_origin_list_t;


struct s_unincorporated_points_csr {
  unsigned char		uninc_format;
  long long*		rowptr; // nrow + 1
  long long*		colidx; // uninc_nnz
};
typedef struct s_unincorporated_points_csr s_unincorporated_points_csr_t;

struct s_spf_structure {
  s_matrix_desc_t		matrix_description;
  s_shape_dictionnary_t		shape_dictionnary;
  s_origin_list_t		origins_list;
  s_unincorporated_points_csr_t	unincorporated_points;
  void*				data;
  int				data_is_float;
};
typedef struct s_spf_structure s_spf_structure_t;

extern
s_spf_structure_t* spf_matrix_read_from_file(char* spfpath);

extern
int spf_matrix_write_to_file(s_spf_structure_t* spf_matrix, char* spfpath);

extern
int spf_matrix_print(s_spf_structure_t* spf_matrix);

extern
int spf_matrix_print_structure(FILE* f, s_spf_structure_t* spf_matrix);

extern
int spf_matrix_print_summary(FILE* f, s_spf_structure_t* spf_matrix);

extern
int spf_matrix_shapes_key_set(s_spf_structure_t* spf_matrix);

extern
int spf_matrix_convert_data_to_float (s_spf_structure_t* spf_mat);



#endif // !_SPF_STRUCTURE_H
