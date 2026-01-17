/*
 * spf_matvect.c: This file is part of the SPF project.
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
#include <time.h>

#include <spf_structure.h>
#include <spf_executors.h>

#include <polybench.h>

#ifndef NB_REPEATS_EXPERIMENT_COLD_CACHE
# define NB_REPEATS_EXPERIMENT_COLD_CACHE 1
#endif

// Tolerance for considering two values "similar"
#define TOLERANCE 0.00001
#define PRECISION 6

static void usage(char** argv)
{
  fprintf (stderr, "Usage: %s [-float,-stats] <spf_matrix.spfdata>\n", argv[0]);
  exit (1);
}


int ensure_base_folder_exists(const char *folder_path) {
    struct stat st = {0};
    if (stat(folder_path, &st) == -1) {
        if (mkdir(folder_path, 0700) != 0) {
            perror("Error creating directory");
            return 1;
        } else {
            printf("Directory created: %s\n", folder_path);
        }
    } else {
        printf("Directory already exists: %s\n", folder_path);
    }
    return 0;
}

void compare_y_values(const char *file_path, void *current_y, int size, int data_is_float) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    size_t elements_read;
    if (data_is_float) {
        float *original_y_float = (float *)malloc(size * sizeof(float));
        float *current_y_float = (float*)current_y;
        if (original_y_float == NULL) {
            perror("Memory allocation failed"); fclose(file); return;
        }

        elements_read = fread(original_y_float, sizeof(float), size, file);
        if (elements_read != size) {
            printf("Error: Expected to read %d elements, but read %zu elements.\n", size, elements_read);
            free(original_y_float); fclose(file); return;
        }

        for (int i = 0; i < size; i++) {
            float net_diff;
            if (current_y_float[i] == 0) {
                net_diff = original_y_float[i] == 0 ? 0.0f : 100.0f;
            } else {
                net_diff = fabsf(original_y_float[i] - current_y_float[i]);
            }
            printf("y[%d]: Original = %.*f, Read = %.*f, Difference = %.*f\n", 
                   i, PRECISION, current_y_float[i], PRECISION, original_y_float[i], PRECISION, net_diff);
        }
        free(original_y_float);
    } else {
        double *original_y_double = (double *)malloc(size * sizeof(double));
        double *current_y_double = (double*)current_y;
        if (original_y_double == NULL) {
            perror("Memory allocation failed"); fclose(file); return;
        }

        elements_read = fread(original_y_double, sizeof(double), size, file);
        if (elements_read != size) {
            printf("Error: Expected to read %d elements, but read %zu elements.\n", size, elements_read);
            free(original_y_double); fclose(file); return;
        }

        for (int i = 0; i < size; i++) {
            double net_diff;
            if (current_y_double[i] == 0) {
                net_diff = original_y_double[i] == 0 ? 0.0 : 100.0;
            } else {
                net_diff = fabs(original_y_double[i] - current_y_double[i]);
            }
            printf("y[%d]: Original = %.*lf, Read = %.*lf, Difference = %.*lf\n", 
                   i, PRECISION, current_y_double[i], PRECISION, original_y_double[i], PRECISION, net_diff);
        }
        free(original_y_double);
    }
    fclose(file);
}


float round_to_precision_float(float value, int precision) {
    float factor = pow(10.0, precision);
    return round(value * factor) / factor;
}

double round_to_precision_double(double value, int precision) {
    double factor = pow(10.0, precision);
    return round(value * factor) / factor;
}


// Function to read binary file and compare y values for similarity, supporting both float and double
void compare_y_values_similarity(const char *file_path, void* current_y, int size, int data_is_float) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    size_t elements_read;
    int matching_elements = 0;

    if (data_is_float) {
        float *read_y_float = (float *)malloc(size * sizeof(float));
        float *current_y_float = (float*)current_y;
        if (read_y_float == NULL) {
            perror("Memory allocation failed"); fclose(file); return;
        }

        elements_read = fread(read_y_float, sizeof(float), size, file);
        if (elements_read != size) {
            printf("Error: Expected to read %d elements, but read %zu elements.\n", size, elements_read);
            free(read_y_float); fclose(file); return;
        }
        fclose(file);

        // Compare rounded float values with tolerance
        for (int i = 0; i < size; i++) {
            float rounded_original = round_to_precision_float(current_y_float[i], PRECISION);
            float rounded_read = round_to_precision_float(read_y_float[i], PRECISION);
            float difference = fabsf(rounded_original - rounded_read);
            if (difference <= TOLERANCE) {
                matching_elements++;
            }
        }
        free(read_y_float);
    } else {
        double *read_y = (double *)malloc(size * sizeof(double));
        double *current_y_double = (double*)current_y;
        if (read_y == NULL) {
            perror("Memory allocation failed"); fclose(file); return;
        }

        // Read data as doubles
        elements_read = fread(read_y, sizeof(double), size, file);
        if (elements_read != size) {
            printf("Error: Expected to read %d elements, but read %zu elements.\n", size, elements_read);
            free(read_y); fclose(file); return;
        }
        fclose(file);

        // Compare rounded double values with tolerance
        for (int i = 0; i < size; i++) {
            double rounded_original = round_to_precision_double(current_y_double[i], PRECISION);
            double rounded_read = round_to_precision_double(read_y[i], PRECISION);
            double difference = fabs(rounded_original - rounded_read);
            if (difference <= TOLERANCE) {
                matching_elements++;
            }
        }
        free(read_y);
    }
    printf("Similarity: %.2f%%\n", ((double)matching_elements / size) * 100.0);
}



void initialize_data(int nrows, int ncols, int data_is_float, void* x, void* y) {
    if (!data_is_float) {
        for (int i = 0; i < nrows; ++i) {
            ((double*)y)[i] = 0.0;
        }
        for (int i = 0; i < ncols; ++i) {
            ((double*)x)[i] = (double)(i % ncols) / ncols;
        }
    } else {
        for (int i = 0; i < nrows; ++i) {
            ((float*)y)[i] = 0.0f;
        }
        for (int i = 0; i < ncols; ++i) {
            ((float*)x)[i] = (float)(i % ncols) / ncols;
        }
    }
}

int polybench_ntasks=1;
int main(int argc, char** argv)
{
  struct timespec start, end;
  if( argc < 2 ) {
      printf( "Usage: spmv_spf <SPF file>\n" );
      exit(0);
  }

  int debug = 0;

  int data_is_float = 0;
  int print_matrix_summary = debug;
  int print_matrix_structure = 0 || (debug > 1);

  char* rbpath = NULL;
  for (int i = 1; i < argc; ++i)
    {
      if (! strcmp (argv[i], "-float"))
	data_is_float = 1;
      else if (! strcmp (argv[i], "-stats"))
	print_matrix_summary = 1;
      else if (! strcmp (argv[i], "-structure"))
	print_matrix_structure = 1;
      else
	{
	  if (rbpath == NULL)
	    rbpath = argv[i];
	  else
	    {
	      fprintf (stderr, "[ERROR] Unsupported argument %s\n",
		       argv[i]);
	      usage (argv);
	    }
	}
    }
  
  if (! rbpath)
    usage (argv);

  // Load the matrix.
  s_spf_structure_t* spf_mat = spf_matrix_read_from_file (rbpath);
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  char binary_file_name[200];
  const char *base_folder = "./original_y_values/";
    // Find the last '/' in the path
  char *filename = strrchr(rbpath, '/');
  if (filename != NULL) {
      filename++;
      char *dot = strchr(filename, '.');
      if (dot != NULL) {
          *dot = '\0';
      }
//      printf("Extracted name: %s\n", filename);
  } else {
      printf("No '/' found in the path!\n");
  }
//  strcpy(binary_file_name, base_folder);
//  strcat(binary_file_name, filename);
//  strcat(binary_file_name, ".bin");
//  printf("%s\n", binary_file_name);
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  if (! spf_mat)
    exit (1);

  // Convert data to float, if asked.
  if (data_is_float)
    spf_matrix_convert_data_to_float (spf_mat);

  if (print_matrix_summary)
    spf_matrix_print_summary (stdout, spf_mat);
  if (print_matrix_structure)
    spf_matrix_print_structure (stdout, spf_mat);
  if (debug > 10)
    spf_matrix_print (spf_mat);

  // Set the total number of flops.
#ifdef POLYBENCH_GFLOPS
  polybench_program_total_flops = spf_mat->matrix_description.nnz * 2L * REPS;
#endif

  // Allocate and initialize x, y to fake values.
  int ncols = spf_mat->matrix_description.ncols;
  int nrows = spf_mat->matrix_description.intnrow;
  void* x;
  void* y;
  if (! data_is_float)
  {
      x = (double*) polybench_alloc_data (ncols, sizeof(double));
      y = (double*) polybench_alloc_data (nrows, sizeof(double));
  }
  else
  {
      x = (float*) polybench_alloc_data (ncols, sizeof(float));
      y = (float*) polybench_alloc_data (nrows, sizeof(float));
  }


  // Repeat the spmv computation, and monitor it with polybench,
  // clearing cache between each repetition.
  for (int repeat = 0; repeat < NB_REPEATS_EXPERIMENT_COLD_CACHE; ++repeat)
  {
      initialize_data(nrows, ncols, data_is_float, x, y);
      polybench_start_instruments;
      for( int t = 0; t < REPS; ++t )
      spf_executors_spf_matrix_dense_vector_product (spf_mat, x, y,
						     ncols, nrows,
						     data_is_float);
      polybench_stop_instruments;
      polybench_print_instruments;

      #ifdef COMPARE_OUTPUT
          compare_y_values_similarity(binary_file_name, y, nrows, data_is_float);
      #endif


      #ifdef COMPARE_INDIVIDUAL_OUTPUT_VALUE
          compare_y_values(binary_file_name, y, nrows, data_is_float);
      #endif


      #ifdef SAVE_OUTPUT
        ensure_base_folder_exists(base_folder);
        FILE *file = fopen(binary_file_name, "wb");
        if (file == NULL) {
            perror("Error opening file");
            return 1;
        }
        size_t elements_written;
        if(data_is_float){
            elements_written = fwrite(y, sizeof(float), nrows, file);
        }
        else{
            elements_written = fwrite(y, sizeof(double), nrows, file);
        }
        
        if (elements_written != nrows) {
            perror("Error writing to file");
            return 1;
        }
        fclose(file);
      #endif
  }

  // Dump the output vector.
  if (debug > 10)
    {
      if (data_is_float)
	{
	  for (int i = 0; i < nrows; i++)
	    printf ("%f ", ((float*)y)[i]);
	}
      else
	{
	  for (int i = 0; i < nrows; i++)
	    printf ("%f ", ((double*)y)[i]);
	}
      printf ("\n");
    }

  return 0;
}
