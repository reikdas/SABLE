/*
 * spf_structure.c: This file is part of the SPF project.
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
#include <polybench.h>


#define print_shape(shape_reg, dim_p, dim_o, output)		\
  {								\
    fprintf (output, "\t\tStart vertex: [ " );			\
    for( int i = 0; i < dim_p; ++i ) {				\
      fprintf (output, "%d ", (shape_reg).start_vertex[i] );	\
    }								\
    fprintf (output, "]\n" );					\
    fprintf (output, "\t\tEnd vertex: [ " );			\
    for( int i = 0; i < dim_p; ++i ) {				\
      fprintf (output, "%d ", (shape_reg).end_vertex[i] );	\
    }								\
    fprintf (output, "]\n" );					\
    fprintf (output, "\t\tStride: [ " );			\
    for( int i = 0; i < dim_p; ++i ) {				\
      fprintf (output, "%d ", (shape_reg).stride[i] );		\
    }								\
    fprintf (output, "]\n" );					\
    fprintf (output, "\t\tLattice:\n" );			\
    for( int i = 0; i < dim_o; ++i ) {				\
      fprintf (output, "\t\t\t[ " );				\
      for( int j = 0; j < dim_p; ++j ) {			\
	fprintf (output, "%d ", (shape_reg).lattice[i][j] );	\
      }								\
      fprintf (output, "]\n" );					\
    }								\
}

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


s_spf_structure_t* spf_matrix_read_from_file(char* spfpath)
{
    int debug = 0;

    // Open file
    if( debug ) {
	    printf( "Reading file: %s\n", spfpath );
    }
    int fd = open( spfpath, O_RDONLY );
    if (fd == -1)
      return NULL;

    s_spf_structure_t* spf_matrix = malloc (sizeof(s_spf_structure_t));
    spf_matrix->data_is_float = 0;

    // Read NNZ, incorporated points, shapes in dictionary, matrix
    // shape (4 x integer)
    read (fd, &(spf_matrix->matrix_description), sizeof(s_matrix_desc_t));

    int dim_p = spf_matrix->matrix_description.dim_p;
    int dim_o = spf_matrix->matrix_description.dim_o;
    int nnz = spf_matrix->matrix_description.nnz;
    int inc_nnz = spf_matrix->matrix_description.inc_nnz;
    int intnrow = spf_matrix->matrix_description.intnrow;
    int ncols = spf_matrix->matrix_description.ncols;
    int nbase_shapes = spf_matrix->matrix_description.nbase_shapes;
    int nhierch_shapes = spf_matrix->matrix_description.nhierch_shapes;
    int data_start = spf_matrix->matrix_description.data_start;

    if( debug ) {
        printf( "Reading matrix:\n" );
        printf( "\tNNZ       = %d\n", nnz );
        printf( "\tINC       = %d\n", inc_nnz );
        printf( "\tdims      = %d x %d\n", intnrow, ncols );
        printf( "\tdim_o     = %d\n", dim_o );
        printf( "\tBase sh   = %d\n", nbase_shapes );
        printf( "\tHierch sh = %d\n", nhierch_shapes );
        printf( "\tData ptr  = %d\n", data_start );
        printf( "\tMax dim_p = %d\n", dim_p );
    }

    spf_matrix->shape_dictionnary.nshapes_dim_p = malloc (dim_p * sizeof(int));
    int* nshapes_dim_p = spf_matrix->shape_dictionnary.nshapes_dim_p;
    read( fd, spf_matrix->shape_dictionnary.nshapes_dim_p,
	  dim_p * sizeof(int) );
    if( debug ) {
        printf( "\tShapes/dim= [ " );
        for( int i = 0; i < dim_p; ++i ) {
            printf( "%d ", nshapes_dim_p[i] );
        }
        printf( "]\n" );
    }

    // Read shapes
    spf_matrix->shape_dictionnary.nbase_shapes = nbase_shapes;
    spf_matrix->shape_dictionnary.shapes =
      malloc (nbase_shapes * sizeof(void*));
    spf_matrix->shape_dictionnary.shapes_nb_origins =
      malloc (nbase_shapes * sizeof(int));
    spf_matrix->shape_dictionnary.shapes_key =
      malloc (nbase_shapes * sizeof(int));
    if( debug ) printf( "\nReading basic shapes:\n" );
    for( int i = 0; i < nbase_shapes; ++i ) {
        short shape_id, shape_encoding, dim_p;

        read( fd, &shape_id, sizeof(short) );
        read( fd, &shape_encoding, sizeof(short) );
        read( fd, &dim_p, sizeof(short) );
        if( debug ) {
            printf( "\tShape #%d\n", i );
            printf( "\t\tEncoding type: %d\n", shape_encoding );
            printf( "\t\tdim(p)       : %d\n", dim_p );
        }
	if (shape_encoding != 0)
	  {
	    printf ("other formats for shapes not supported yet!\n");
	    exit (51);
	  }
	int shape_bytes_size = 0;
	if (dim_o == 1)
	  {
	    switch (dim_p)
	      {
	      case 1:
		shape_bytes_size = sizeof(s_codelet_1x1d_t);
		break;
	      case 2:
		shape_bytes_size = sizeof(s_codelet_1x2d_t);
		break;
	      case 3:
		shape_bytes_size = sizeof(s_codelet_1x3d_t);
		break;
	      default:
		exit (48);
	      }
	  }
	else if (dim_o == 2)
	  {
	    switch (dim_p)
	      {
	      case 1:
		shape_bytes_size = sizeof(s_codelet_2x1d_t);
		break;
	      case 2:
		shape_bytes_size = sizeof(s_codelet_2x2d_t);
		break;
	      case 3:
		shape_bytes_size = sizeof(s_codelet_2x3d_t);
		break;
	      default:
		exit (48);
	      }
	  }
	else if (dim_o == 3)
	  {
	    switch (dim_p)
	      {
	      case 1:
		shape_bytes_size = sizeof(s_codelet_3x1d_t);
		break;
	      case 2:
		shape_bytes_size = sizeof(s_codelet_3x2d_t);
		break;
	      case 3:
		shape_bytes_size = sizeof(s_codelet_3x3d_t);
		break;
	      default:
		exit (48);
	      }
	  }
	else
	  exit (47);

	// Was: s/int/short below, with 'short dim_p' in the structure.
	int* shape_ptr_short = malloc (shape_bytes_size);
	shape_ptr_short++;
	shape_bytes_size -= sizeof(int);
	read( fd, shape_ptr_short, shape_bytes_size);
	shape_ptr_short--;
	shape_ptr_short[0] = dim_p;
	spf_matrix->shape_dictionnary.shapes[i] = shape_ptr_short;
	spf_matrix->shape_dictionnary.shapes_nb_origins[i] = 0;
	spf_matrix->shape_dictionnary.shapes_key[i] = 0;
    }

    // Read AST locations for each shape
    int norigins;
    read( fd, &norigins, sizeof(int) );

    spf_matrix->origins_list.dim_o = dim_o;
    spf_matrix->origins_list.norigins = norigins;
    int origin_byte_size = 0;
    switch (dim_o)
      {
      case 1:
	origin_byte_size = sizeof(s_origin_1d_t);
	break;
      case 2:
	origin_byte_size = sizeof(s_origin_2d_t);
	break;
      case 3:
	origin_byte_size = sizeof(s_origin_3d_t);
	break;
      default:
	exit (52);
      }
    spf_matrix->origins_list.origins = malloc (norigins * origin_byte_size);
    void* origin_buffer = malloc (origin_byte_size);

    if( debug ) printf( "\nReading %d shape origins:\n", norigins );
    for (int i = 0; i < norigins; ++i) {
      read (fd, origin_buffer, origin_byte_size);
      int shape_id = ((s_origin_1d_t*)origin_buffer)->shape_id;
      // Error in shape id.
      if (shape_id < 0)
	exit (189);
      switch (dim_o) {
      case 1:
	{
	  ((s_origin_1d_t*)spf_matrix->origins_list.origins)[i] =
	    *((s_origin_1d_t*)origin_buffer);
	  spf_matrix->shape_dictionnary.shapes_nb_origins
	    [((s_origin_1d_t*)spf_matrix->origins_list.origins)[i].shape_id]++;
	  break;
	}
      case 2:
	{
	  ((s_origin_2d_t*)spf_matrix->origins_list.origins)[i] =
	    *((s_origin_2d_t*)origin_buffer);
	  spf_matrix->shape_dictionnary.shapes_nb_origins
	    [((s_origin_2d_t*)spf_matrix->origins_list.origins)[i].shape_id]++;
	  break;
	}
      case 3:
	{
	  ((s_origin_3d_t*)spf_matrix->origins_list.origins)[i] =
	    *((s_origin_3d_t*)origin_buffer);
	  spf_matrix->shape_dictionnary.shapes_nb_origins
	    [((s_origin_3d_t*)spf_matrix->origins_list.origins)[i].shape_id]++;
	  break;
	}
      default:
	exit (53);
      }
    }
    free (origin_buffer);

    // Unincorporated metadata
    // Read unincorporated format: CSR, COO
    read(fd, &(spf_matrix->unincorporated_points.uninc_format),
	       sizeof(unsigned char) );
    int uninc_format = spf_matrix->unincorporated_points.uninc_format;

    if( debug ) printf( "\nUnincorporated metadata:\n" );
    if( debug ) printf( "\tUnincorporated format: %d (0:CSR, 1: CSC, 2:COO)\n", uninc_format );


    // Read CSR/COO data
    int uninc_nnz = nnz - inc_nnz;
    if( uninc_nnz == 0 ) {
      spf_matrix->unincorporated_points.rowptr = NULL;
      spf_matrix->unincorporated_points.colidx = NULL;
    } else if(uninc_format == 0 ) {
        int * tmprow = (int*)malloc(sizeof(int)*((intnrow)+1));
        int * tmpcol = (int*)malloc(sizeof(int)*(uninc_nnz));
        long long *rowptr = (long long *)malloc( sizeof(long long)*((intnrow)+1) );
        long long *colidx = (long long *)malloc( sizeof(long long)*(uninc_nnz) );
        read( fd, tmprow, ((intnrow)+1)*sizeof(int) );
        read( fd, tmpcol, (uninc_nnz)*sizeof(int) );
        for( int i = 0; i < (intnrow)+1; ++i ) {
            rowptr[i] = tmprow[i];
        }
        for( int i = 0; i < uninc_nnz; ++i ) { // Convert from int to long long
            colidx[i] = tmpcol[i];
        }
        if( debug > 4) {
            printf( "\tRowptr: [ " );
            for( int i = 0; i < (intnrow)+1; ++i ) {
                printf( "%d, ", tmprow[i] );
            }
            printf( "]\n" );
            printf( "\tColptr: [ " );
            for( int i = 0; i < uninc_nnz; ++i ) {
                printf( "%d, ", tmpcol[i] );
            }
            printf( "]\n" );
        }
    	rowptr[intnrow] = uninc_nnz;
        free(tmprow);
        free(tmpcol);
	spf_matrix->unincorporated_points.rowptr = rowptr;
	spf_matrix->unincorporated_points.colidx = colidx;
    } else if( uninc_format == 2 ) {
        int * tmprow = (int*)malloc(sizeof(int)*(uninc_nnz));
        int * tmpcol = (int*)malloc(sizeof(int)*(uninc_nnz));
        long long *rowptr = (long long *)malloc( sizeof(long long)*(uninc_nnz) );
        long long *colidx = (long long *)malloc( sizeof(long long)*(uninc_nnz) );
        read( fd, tmprow, (uninc_nnz)*sizeof(int) );
        read( fd, tmpcol, (uninc_nnz)*sizeof(int) );
        for( int i = 0; i < uninc_nnz; ++i ) { // Convert from int to long long
            rowptr[i] = tmprow[i];
            colidx[i] = tmpcol[i];
        }

        if( debug > 4) {
            printf( "\tLocations of unincorporated points: [ " );
            for( int i = 0; i < uninc_nnz; ++i ) {
                printf( "( %d, %d ), ", tmprow[i], tmpcol[i] );
            }
            printf( "]\n" );
        }
        free(tmprow);
        free(tmpcol);
	spf_matrix->unincorporated_points.rowptr = rowptr;
	spf_matrix->unincorporated_points.colidx = colidx;
    }

    // Allocate data and read
    double* A = (double *)polybench_alloc_data (nnz, sizeof(double));
    int bytes_read = read( fd, A, nnz*sizeof(double) );

    if( debug > 5) {
        printf( "\nData section: " );
        for( int i = 0; i < nnz; ++i ) {
            printf( "%lf ", A[i] );
        }
        printf( "\n" );
        printf( "\nFile ended at offset: %d\n", lseek( fd, 0, SEEK_CUR ) );
        printf( "\nTotal number of doubles read: %lf\n", ((float)bytes_read) / sizeof(double) );
    }
    close(fd);

    spf_matrix->data = A;

    // Post-processing.
    spf_matrix_shapes_key_set (spf_matrix);

    return spf_matrix;
}


int spf_matrix_write_to_file(s_spf_structure_t* spf_matrix, char* spfpath)
{
    int debug = 0;

    // Open file for writing
    if (debug) {
        printf("Writing to file: %s\n", spfpath);
    }

    int fd = open(spfpath, O_WRONLY | O_CREAT, 0644);
    if (fd == -1) {
        return -1; // Error opening file
    }

    // Write matrix description (NNZ, incorporated points, shapes, etc.)
    if (debug) {
        printf("Writing matrix description:\n");
        printf("\tNNZ       = %d\n", spf_matrix->matrix_description.nnz);
        printf("\tINC       = %d\n", spf_matrix->matrix_description.inc_nnz);
        printf("\tdims      = %d x %d\n", spf_matrix->matrix_description.intnrow, spf_matrix->matrix_description.ncols);
        printf("\tdim_o     = %d\n", spf_matrix->matrix_description.dim_o);
        printf("\tBase sh   = %d\n", spf_matrix->matrix_description.nbase_shapes);
        printf("\tHierch sh = %d\n", spf_matrix->matrix_description.nhierch_shapes);
        printf("\tData ptr  = %d\n", spf_matrix->matrix_description.data_start);
        printf("\tMax dim_p = %d\n", spf_matrix->matrix_description.dim_p);
    }

    // Write the matrix description (size = sizeof(s_matrix_desc_t))
    if (write(fd, &(spf_matrix->matrix_description), sizeof(s_matrix_desc_t)) == -1) {
        close(fd);
        return -1; // Error writing matrix description
    }


    // Write shape dictionary: nshapes_dim_p
    int dim_p = spf_matrix->matrix_description.dim_p;
    int* nshapes_dim_p = spf_matrix->shape_dictionnary.nshapes_dim_p;

    if (debug > 1) {
        printf("Writing shapes per dimension:\n\tShapes/dim= [ ");
        for (int i = 0; i < dim_p; ++i) {
            printf("%d ", nshapes_dim_p[i]);
        }
        printf("]\n");
    }

    // Write the shapes per dimension data (size = dim_p * sizeof(int))
    if (write(fd, nshapes_dim_p, dim_p * sizeof(int)) == -1) {
        close(fd);
        return -1; // Error writing nshapes_dim_p
    }
    

    // Writing shapes
    int nbase_shapes = spf_matrix->shape_dictionnary.nbase_shapes;
    if (debug) {
        printf("\nWriting basic shapes:\n");
    }


    for (int i = 0; i < nbase_shapes; ++i) {
        short shape_id, shape_encoding, dim_p;

        // Assuming shape encoding and other metadata are already stored in shape structure
        // We extract them from the structure
        int* shape_ptr_short = (int*)spf_matrix->shape_dictionnary.shapes[i];
        dim_p = shape_ptr_short[0]; // First element of the shape structure holds dim_p

        // Here we need to have store the shape_id and shape_encoding in the data structure
        shape_id = (short)i; // shape_id is the index (can replace this in future)
        shape_encoding = 0;  // all shapes have encoding 0 as per read logic

        // Write shape ID, encoding, and dim_p
        if (write(fd, &shape_id, sizeof(short)) == -1 ||
            write(fd, &shape_encoding, sizeof(short)) == -1 ||
            write(fd, &dim_p, sizeof(short)) == -1) {
            close(fd);
            return -1; // Error writing shape data
        }
        
        if (debug) {
            printf("\tShape #%d\n", i);
            printf("\t\tEncoding type: %d\n", shape_encoding);
            printf("\t\tdim(p)       : %d\n", dim_p);
        }

        // Calculate shape byte size based on dim_o and dim_p
        int shape_bytes_size = 0;
        int dim_o = spf_matrix->matrix_description.dim_o;

        if (dim_o == 1) {
            switch (dim_p) {
                case 1:
                    shape_bytes_size = sizeof(s_codelet_1x1d_t);
                    break;
                case 2:
                    shape_bytes_size = sizeof(s_codelet_1x2d_t);
                    break;
                case 3:
                    shape_bytes_size = sizeof(s_codelet_1x3d_t);
                    break;
                default:
                    exit(48);
            }
        } else if (dim_o == 2) {
            switch (dim_p) {
                case 1:
                    shape_bytes_size = sizeof(s_codelet_2x1d_t);
                    break;
                case 2:
                    shape_bytes_size = sizeof(s_codelet_2x2d_t);
                    break;
                case 3:
                    shape_bytes_size = sizeof(s_codelet_2x3d_t);
                    break;
                default:
                    exit(48);
            }
        } else if (dim_o == 3) {
            switch (dim_p) {
                case 1:
                    shape_bytes_size = sizeof(s_codelet_3x1d_t);
                    break;
                case 2:
                    shape_bytes_size = sizeof(s_codelet_3x2d_t);
                    break;
                case 3:
                    shape_bytes_size = sizeof(s_codelet_3x3d_t);
                    break;
                default:
                    exit(48);
            }
        } else {
            exit(47);
        }

        // Write shape data (excluding the first int, which is dim_p)
        int* shape_ptr = (int*)spf_matrix->shape_dictionnary.shapes[i];
        shape_ptr++;  // Skip the first element (dim_p)
        shape_bytes_size -= sizeof(int);  // Exclude the first int from the size

        if (write(fd, shape_ptr, shape_bytes_size) == -1) {
            close(fd);
            return -1; // Error writing shape data
        }

        shape_ptr--;  // Reset pointer (not necessary, but for consistency)
    }


    // Write the number of origins
    int norigins = spf_matrix->origins_list.norigins;
    if (write(fd, &norigins, sizeof(int)) == -1) {
        return -1; // Error writing number of origins
    }

    int dim_o = spf_matrix->origins_list.dim_o;
    int origin_byte_size = 0;
    // Determine origin byte size based on dim_o
    switch (dim_o) {
        case 1:
            origin_byte_size = sizeof(s_origin_1d_t);
            break;
        case 2:
            origin_byte_size = sizeof(s_origin_2d_t);
            break;
        case 3:
            origin_byte_size = sizeof(s_origin_3d_t);
            break;
        default:
            exit(52); // Unsupported dimension
    }


    void* origins = spf_matrix->origins_list.origins;
    if (debug) {
        printf("\nWriting %d shape origins and %d size:\n", norigins, origin_byte_size);
    }

    // Write the origins for each shape based on the dimensionality (dim_o)
    for (int i = 0; i < norigins; ++i) {
        if (write(fd, origins + i * origin_byte_size, origin_byte_size) == -1) {
            return -1; // Error writing 1D origin
        }
    }


    // Write unincorporated format: CSR, CSC, COO
    unsigned char uninc_format = spf_matrix->unincorporated_points.uninc_format;
    if (write(fd, &uninc_format, sizeof(unsigned char)) == -1) {
        return -1; // Error writing unincorporated format
    }

    // Write CSR/COO data
    int intnrow = spf_matrix->matrix_description.intnrow;
    int nnz = spf_matrix->matrix_description.nnz;
    int inc_nnz = spf_matrix->matrix_description.inc_nnz;
    int uninc_nnz = nnz - inc_nnz;

    if (uninc_nnz == 0) {
        spf_matrix->unincorporated_points.rowptr = NULL;
        spf_matrix->unincorporated_points.colidx = NULL;
    }
    // Handle CSR format (uninc_format == 0)
    else if (uninc_format == 0) {
        if (debug) {
            printf("\tWriting CSR format:\n");
        }

        // Write rowptr and colidx
        long long* rowptr = spf_matrix->unincorporated_points.rowptr;
        long long* colidx = spf_matrix->unincorporated_points.colidx;

        // Convert back to int from double for writing
        int* tmprow = (int*)malloc(sizeof(int) * (intnrow + 1));
        int* tmpcol = (int*)malloc(sizeof(int) * uninc_nnz);

        for (int i = 0; i < (intnrow + 1); ++i) {
            tmprow[i] = (int)rowptr[i];
        }
        for (int i = 0; i < uninc_nnz; ++i) {
            tmpcol[i] = (int)colidx[i];
        }

        // Write the rowptr and colidx to the file
        if (write(fd, tmprow, (intnrow + 1) * sizeof(int)) == -1 || write(fd, tmpcol, uninc_nnz * sizeof(int)) == -1) {
            free(tmprow);
            free(tmpcol);
            return -1;  // Error writing rowptr/colidx
        }

        if (debug > 5) {
            printf("\tRowptr: [ ");
            for (int i = 0; i < (intnrow + 1); ++i) {
                printf("%d, ", tmprow[i]);
            }
            printf("]\n\tColptr: [ ");
            for (int i = 0; i < uninc_nnz; ++i) {
                printf("%d, ", tmpcol[i]);
            }
            printf("]\n");
        }

        free(tmprow);
        free(tmpcol);
    }
    // Handle COO format (uninc_format == 2)
    else if (uninc_format == 2) {
        if (debug) {
            printf("\tWriting COO format:\n");
        }

        // Convert back to int from double for writing
        long long* rowptr = spf_matrix->unincorporated_points.rowptr;
        long long* colidx = spf_matrix->unincorporated_points.colidx;
        int* tmprow = (int*)malloc(sizeof(int) * uninc_nnz);
        int* tmpcol = (int*)malloc(sizeof(int) * uninc_nnz);

        for (int i = 0; i < uninc_nnz; ++i) {
            tmprow[i] = (int)rowptr[i];
            tmpcol[i] = (int)colidx[i];
        }

        // Write the rowptr and colidx to the file
        if (write(fd, tmprow, uninc_nnz * sizeof(int)) == -1 || write(fd, tmpcol, uninc_nnz * sizeof(int)) == -1) {
            free(tmprow);
            free(tmpcol);
            return -1;  // Error writing rowptr/colidx
        }

        if (debug > 5) {
            printf("\tLocations of unincorporated points: [ ");
            for (int i = 0; i < uninc_nnz; ++i) {
                printf("( %d, %d ), ", tmprow[i], tmpcol[i]);
            }
            printf("]\n");
        }

        free(tmprow);
        free(tmpcol);
    }


    // Allocate the data pointer (already done in spf_matrix)
    double* A = spf_matrix->data;

    // Write data to the file
    int bytes_written = write(fd, A, nnz * sizeof(double));
    if (bytes_written == -1) {
        return -1; // Error writing data
    }

    close(fd);
    
    if (debug) {
        printf("Matrix written successfully to %s\n", spfpath);
    }

    return 0; // Success
}


#define print_origin(origin_reg, dim_o)				\
{								\
  printf( "\tOrigin for shape #%d: [", (origin_reg).shape_id );	\
  for( int j = 0; j < dim_o; ++j ) {				\
    printf( "%d ", (origin_reg).coordinates[j] );		\
  }								\
  printf( "]\n" );						\
  printf( "\tData ptr       : %d\n", (origin_reg).dataptr );	\
}


int spf_matrix_print_summary(FILE* f, s_spf_structure_t* spf_matrix)
{
  if (! spf_matrix)
    return -1;
  int dim_p = spf_matrix->matrix_description.dim_p;
  int dim_o = spf_matrix->matrix_description.dim_o;
  int nnz = spf_matrix->matrix_description.nnz;
  int inc_nnz = spf_matrix->matrix_description.inc_nnz;
  int intnrow = spf_matrix->matrix_description.intnrow;
  int ncols = spf_matrix->matrix_description.ncols;
  int nbase_shapes = spf_matrix->matrix_description.nbase_shapes;
  int nhierch_shapes = spf_matrix->matrix_description.nhierch_shapes;
  int data_start = spf_matrix->matrix_description.data_start;
  int* nshapes_dim_p = spf_matrix->shape_dictionnary.nshapes_dim_p;

  // Print the core information:
  fprintf (f, "SPF matrix stats:\n" );
  fprintf (f, "\tNNZ       = %d\n", nnz );
  fprintf (f, "\tINC       = %d\n", inc_nnz );
  if (spf_matrix->unincorporated_points.uninc_format == 0)
    fprintf (f, "\tformat    = CSR (%.2f%%)\n", 100*(nnz - inc_nnz)/(float)nnz);
  else
    fprintf (f, "\tformat    = COO (%.2f%%)\n", 100*(nnz - inc_nnz)/(float)nnz);
  fprintf (f, "\tdims      = %d x %d\n", intnrow, ncols );
  fprintf (f, "\tdim_o     = %d\n", dim_o );
  fprintf (f, "\tBase sh   = %d\n", nbase_shapes );
  fprintf (f, "\tHierch sh = %d\n", nhierch_shapes );
  fprintf (f, "\tMax dim_p = %d\n", dim_p );
  if (spf_matrix->data_is_float)
    fprintf (f, "\tData type = float\n");
  else
    fprintf (f, "\tData type = double\n");

  return 0;
}

int spf_matrix_print_structure(FILE* f, s_spf_structure_t* spf_matrix)
{
  if (! spf_matrix)
    return -1;
  int dim_p = spf_matrix->matrix_description.dim_p;
  int dim_o = spf_matrix->matrix_description.dim_o;
  int nnz = spf_matrix->matrix_description.nnz;
  int inc_nnz = spf_matrix->matrix_description.inc_nnz;
  int intnrow = spf_matrix->matrix_description.intnrow;
  int ncols = spf_matrix->matrix_description.ncols;
  int nbase_shapes = spf_matrix->matrix_description.nbase_shapes;
  int nhierch_shapes = spf_matrix->matrix_description.nhierch_shapes;
  int data_start = spf_matrix->matrix_description.data_start;
  int* nshapes_dim_p = spf_matrix->shape_dictionnary.nshapes_dim_p;

  // Print the core information:
  fprintf (f, "Reading matrix:\n" );
  fprintf (f, "\tNNZ       = %d\n", nnz );
  fprintf (f, "\tINC       = %d\n", inc_nnz );
  fprintf (f, "\tdims      = %d x %d\n", intnrow, ncols );
  fprintf (f, "\tdim_o     = %d\n", dim_o );
  fprintf (f, "\tBase sh   = %d\n", nbase_shapes );
  fprintf (f, "\tHierch sh = %d\n", nhierch_shapes );
  fprintf (f, "\tData ptr  = %d\n", data_start );
  fprintf (f, "\tMax dim_p = %d\n", dim_p );

  fprintf (f, "\tShapes/dim= [ " );
  for (int i = 0; i < dim_p; ++i)
    fprintf (f, "%d ", nshapes_dim_p[i] );
  fprintf (f, "]\n" );

  // Print the dictionnary of shapes:
  fprintf (f, "\nReading basic shapes:\n" );
  for (int i = 0; i < nbase_shapes; ++i) {
    void* shape = spf_matrix->shape_dictionnary.shapes[i];
    s_codelet_1x1d_t* as_1d = (s_codelet_1x1d_t*)shape;
    short dim_p = as_1d->dim_p;

    fprintf (f, "\tShape #%d\n", i );
    fprintf (f, "\t\tEncoding type		: %d\n", 0 );
    fprintf (f, "\t\tdim(p)			: %d\n", dim_p );
    fprintf (f, "\t\tNumber of occurences	: %d\n",
	     spf_matrix->shape_dictionnary.shapes_nb_origins[i]);

    if (dim_o == 1)
      {
	switch (dim_p) {
	case 1:
	  {
	    shape_to_reg_1x1(shape, shape_reg);
	    print_shape(shape_reg, dim_p, dim_o, f);
	    break;
	  }
	case 2:
	  {
	    shape_to_reg_1x2(shape, shape_reg);
	    print_shape(shape_reg, dim_p, dim_o, f);
	    break;
	  }
	default:
	  exit (42);
	}
      }
    else if (dim_o == 2)
      {
	switch (dim_p) {
	case 1:
	  {
	    shape_to_reg_2x1(shape, shape_reg);
	    print_shape(shape_reg, dim_p, dim_o, f);
	    break;
	  }
	case 2:
	  {
	    shape_to_reg_2x2(shape, shape_reg);
	    print_shape(shape_reg, dim_p, dim_o, f);
	    break;
	  }
	default:
	  exit (43);
	}
      }
    else if (dim_o == 3)
      {
	switch (dim_p) {
	case 1:
	  {
	    shape_to_reg_3x1(shape, shape_reg);
	    print_shape(shape_reg, dim_p, dim_o, f);
	    break;
	  }
	case 2:
	  {
	    shape_to_reg_3x2(shape, shape_reg);
	    print_shape(shape_reg, dim_p, dim_o, f);
	    break;
	  }
	default:
	  exit (44);
	}
      }
    else
      exit (45);
  }

  return 0;
}


int spf_matrix_print(s_spf_structure_t* spf_matrix)
{
  if (! spf_matrix)
    return -1;
  int dim_p = spf_matrix->matrix_description.dim_p;
  int dim_o = spf_matrix->matrix_description.dim_o;
  int nnz = spf_matrix->matrix_description.nnz;
  int inc_nnz = spf_matrix->matrix_description.inc_nnz;
  int intnrow = spf_matrix->matrix_description.intnrow;
  int ncols = spf_matrix->matrix_description.ncols;
  int nbase_shapes = spf_matrix->matrix_description.nbase_shapes;
  int nhierch_shapes = spf_matrix->matrix_description.nhierch_shapes;
  int data_start = spf_matrix->matrix_description.data_start;
  int* nshapes_dim_p = spf_matrix->shape_dictionnary.nshapes_dim_p;

  spf_matrix_print_structure (stdout, spf_matrix);

  // Read AST locations for each shape
  int norigins = spf_matrix->origins_list.norigins;
  printf( "\nReading %d shape origins:\n", norigins );
  switch (dim_o) {
  case 1:
    {
      for (int i = 0; i < norigins; ++i)
	print_origin(((s_origin_1d_t*)spf_matrix->origins_list.origins)[i], dim_o);
      break;
    }
  case 2:
    {
      for (int i = 0; i < norigins; ++i)
	print_origin(((s_origin_2d_t*)spf_matrix->origins_list.origins)[i], dim_o);
      break;
    }
  case 3:
    {
      for (int i = 0; i < norigins; ++i)
	print_origin(((s_origin_2d_t*)spf_matrix->origins_list.origins)[i], dim_o);
      break;
    }
  default:
    exit (46);
  }

  // Read CSR/COO data
  int uninc_nnz = nnz - inc_nnz;
  if (uninc_nnz == 0)
    printf (" => No CSR data leftover points\n");
  else if (spf_matrix->unincorporated_points.uninc_format == 0)
    {
      printf ("Printing unincorporated points coordinates in CSR\n");
      printf( "\tRowptr: [ " );
      for( int i = 0; i < (intnrow)+1; ++i ) {
	printf( "%lld, ", spf_matrix->unincorporated_points.rowptr[i] );
      }
      printf( "]\n" );
      printf( "\tColptr: [ " );
      for( int i = 0; i < uninc_nnz; ++i ) {
	printf( "%lld, ", spf_matrix->unincorporated_points.colidx[i] );
      }
      printf( "]\n" );
      printf ("Nb nnz in CSR: %lld\n",
	      spf_matrix->unincorporated_points.rowptr[intnrow]);
    }
  else
    {
      printf ("other sparse formats not supported yet\n");
      //exit (47);
    }

    // Allocate data and read
    printf( "\nData section: " );
    if (spf_matrix->data_is_float)
      {
	for( int i = 0; i < nnz; ++i )
	  printf( "%f ", ((float*)spf_matrix->data)[i] );
      }
    else
      {
	for( int i = 0; i < nnz; ++i )
	  printf( "%lf ", ((double*)spf_matrix->data)[i] );
      }
    printf( "\n" );

    return 0;
}


#define _key_code_compute_oneval_2x1_0x1(nbiters, keyvalue)	\
if (dim_p == 1 && dim_o == 2 &&					\
    shape_reg.stride[0] == 1 &&					\
    shape_reg.end_vertex[0] == (nbiters) &&			\
    shape_reg.lattice[0][0] == 0 &&				\
    shape_reg.lattice[1][0] == 1)				\
  key_value = keyvalue;

#define _key_code_compute_oneval_2x1_1x0(nbiters, keyvalue)	\
if (dim_p == 1 && dim_o == 2 &&					\
    shape_reg.stride[0] == 1 &&					\
    shape_reg.end_vertex[0] == (nbiters) &&			\
    shape_reg.lattice[0][0] == 1 &&				\
    shape_reg.lattice[1][0] == 0)				\
  key_value = keyvalue;

#define _key_code_compute_oneval_2x1_1x1(nbiters, keyvalue)	\
if (dim_p == 1 && dim_o == 2 &&					\
    shape_reg.stride[0] == 1 &&					\
    shape_reg.end_vertex[0] == (nbiters) &&			\
    shape_reg.lattice[0][0] == 1 &&				\
    shape_reg.lattice[1][0] == 1)				\
  key_value = keyvalue;

#define key_code_compute_2x1				\
int key_value = 0;					\
_key_code_compute_oneval_2x1_0x1(1, 1)			\
else _key_code_compute_oneval_2x1_0x1(2, 2)		\
else _key_code_compute_oneval_2x1_0x1(3, 3)		\
else _key_code_compute_oneval_2x1_0x1(4, 4)		\
else _key_code_compute_oneval_2x1_0x1(5, 5)		\
else _key_code_compute_oneval_2x1_0x1(6, 6)		\
else _key_code_compute_oneval_2x1_0x1(7, 7)		\
else _key_code_compute_oneval_2x1_0x1(15, 8)		\
else _key_code_compute_oneval_2x1_0x1(31, 9)		\
else _key_code_compute_oneval_2x1_0x1(63, 10)		\
else _key_code_compute_oneval_2x1_1x0(1, 11)		\
else _key_code_compute_oneval_2x1_1x0(2, 12)		\
else _key_code_compute_oneval_2x1_1x0(3, 13)		\
else _key_code_compute_oneval_2x1_1x0(4, 14)		\
else _key_code_compute_oneval_2x1_1x0(5, 15)		\
else _key_code_compute_oneval_2x1_1x0(6, 16)		\
else _key_code_compute_oneval_2x1_1x0(7, 17)		\
else _key_code_compute_oneval_2x1_1x0(15, 18)		\
else _key_code_compute_oneval_2x1_1x0(31, 19)		\
else _key_code_compute_oneval_2x1_1x0(63, 20)		\
else _key_code_compute_oneval_2x1_1x1(1, 21)		\
else _key_code_compute_oneval_2x1_1x1(2, 22)		\
else _key_code_compute_oneval_2x1_1x1(3, 23)		\
else _key_code_compute_oneval_2x1_1x1(4, 24)		\
else _key_code_compute_oneval_2x1_1x1(5, 25)		\
else _key_code_compute_oneval_2x1_1x1(6, 26)		\
else _key_code_compute_oneval_2x1_1x1(7, 27)		\
else _key_code_compute_oneval_2x1_1x1(15, 28)		\
else _key_code_compute_oneval_2x1_1x1(31, 29)		\
else _key_code_compute_oneval_2x1_1x1(63, 30)		\
spf_matrix->shape_dictionnary.shapes_key[i] = key_value;


int spf_matrix_shapes_key_set(s_spf_structure_t* spf_matrix)
{
  int dim_o = spf_matrix->matrix_description.dim_o;
  int nbase_shapes = spf_matrix->matrix_description.nbase_shapes;

  for (int i = 0; i < nbase_shapes; ++i) {
    void* shape = spf_matrix->shape_dictionnary.shapes[i];
    s_codelet_1x1d_t* as_1d = (s_codelet_1x1d_t*)shape;
    short dim_p = as_1d->dim_p;

    if (dim_o == 1)
      {
	switch (dim_p) {
	case 1:
	  {
	    shape_to_reg_1x1(shape, shape_reg);
	    break;
	  }
	case 2:
	  {
	    shape_to_reg_1x2(shape, shape_reg);
	    break;
	  }
	default:
	  exit (42);
	}
      }
    else if (dim_o == 2)
      {
	switch (dim_p) {
	case 1:
	  {
	    shape_to_reg_2x1(shape, shape_reg);
	    key_code_compute_2x1;
	    break;
	  }
	case 2:
	  {
	    shape_to_reg_2x2(shape, shape_reg);
	    break;
	  }
	default:
	  exit (43);
	}
      }
    else if (dim_o == 3)
      {
	switch (dim_p) {
	case 1:
	  {
	    shape_to_reg_3x1(shape, shape_reg);
	    break;
	  }
	case 2:
	  {
	    shape_to_reg_3x2(shape, shape_reg);
	    break;
	  }
	default:
	  exit (44);
	}
      }
    else
      exit (45);
  }

}


int spf_matrix_convert_data_to_float (s_spf_structure_t* spf_mat)
{
  if (!spf_mat)
    return -1;

  int nnz = spf_mat->matrix_description.nnz;
  if (!nnz)
    {
      spf_mat->data_is_float = 1;
      return 0;
    }

  if (! spf_mat->data_is_float)
    {
      // Allocate data
      float* A = (float *)polybench_alloc_data (nnz, sizeof(float));
      for (int i = 0; i < nnz; ++i)
	A[i] = (float)((double*)spf_mat->data)[i];
      free (spf_mat->data);
      spf_mat->data = A;
      spf_mat->data_is_float = 1;
    }

  return 0;
}
