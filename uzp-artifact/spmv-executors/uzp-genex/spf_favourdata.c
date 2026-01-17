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

#include <polybench.h>
#include <time.h>
#include <omp.h>

#define DEBUG 0
#define FIND_CONSECUTIVE_ORIGINS 1


typedef struct {
    int x;             // x-coordinate
    int y;             // y-coordinate
    int* dataptr;       // data pointer
    int data_count;     // number of data points
} Coordinate;


typedef struct {
    int shape_id;
    Coordinate* coordinates;
    int count;
    int lattice_y; // row
    int lattice_x; // column
    int start_vertex;
    int end_vertex;
    int capacity; // the number of origin co-ordinates present in the given shape
} ShapeOrigins;


typedef struct {
    void **shapes;
    int count;
    int capacity;
} ShapeMap;


void swap_and_update_prime(void** shapes, int* start_index, int j) {
    // printf("latticey: %d, latticex: %d\n", ((ShapeOrigins*)shapes[j])->lattice_y, ((ShapeOrigins*)shapes[j])->lattice_x);
    void* temp = shapes[*start_index];
    shapes[*start_index] = shapes[j];
    shapes[j] = temp;
    (*start_index)++;
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


// Function to compare two Coordinate structs lexicographically based on y, then x
int compare_coordinates(const void* a, const void* b) {
    const Coordinate* coord_a = (const Coordinate*)a;
    const Coordinate* coord_b = (const Coordinate*)b;

    // First compare based on y-coordinate
    if (coord_a->y < coord_b->y) return -1;
    if (coord_a->y > coord_b->y) return 1;

    // If y-coordinates are the same, compare based on x-coordinate
    if (coord_a->x < coord_b->x) return -1;
    if (coord_a->x > coord_b->x) return 1;

    // If both y and x are the same, they are equal
    return 0;
}

int compare_coordinates_xsort(const void* a, const void* b) {
    const Coordinate* coord_a = (const Coordinate*)a;
    const Coordinate* coord_b = (const Coordinate*)b;

    const int block_size = 128;
    int block_a = coord_a->x / block_size;
    int block_b = coord_b->x / block_size;
    if (block_a < block_b) return -1;
    if (block_a > block_b) return 1;
    return 0;
}


// Function to create a ShapeOrigins entry
void create_shape(ShapeMap* map, int shape_id, int lattice_y, int lattice_x, int start_vertex, int end_vertex, int num_shape_occurrences) {
    // create a new entry
    ShapeOrigins* new_shape = (ShapeOrigins*)malloc(sizeof(ShapeOrigins));
    if (new_shape == NULL) {
      fprintf(stderr, "Failed to allocate memory for ShapeOrigins.\n");
      exit(EXIT_FAILURE);
    }

    new_shape->shape_id = shape_id;
    new_shape->count = 0;
    new_shape->capacity = num_shape_occurrences;
    new_shape->lattice_y = lattice_y;
    new_shape->lattice_x = lattice_x;
    new_shape->start_vertex = start_vertex;
    new_shape->end_vertex = end_vertex;
    new_shape->coordinates = (Coordinate*)malloc(sizeof(Coordinate) * num_shape_occurrences);
    map->shapes[map->count] = (void*)new_shape;
    map->count++;
}


// Function to find a ShapeOrigins entry
// Finds an existing shape_id entry if not found return NULL
ShapeOrigins* find_shape(ShapeMap* map, int shape_id) {
    for (int i = 0; i < map->count; ++i) {
        // Cast the void* to ShapeOrigins* to access the shape_id
        ShapeOrigins* shape = (ShapeOrigins*)map->shapes[i];
        if (shape->shape_id == shape_id) {
            return shape;  // Return the found shape
        }
    }
    return NULL;  // Return NULL if no matching shape is found
}


// Function to add coordinates to a ShapeOrigins entry
// Adds a new set of coordinates to the ShapeOrigins.
void add_coordinates(ShapeOrigins* shape, int y, int x, int data_ptr) {
    if(shape->count >= shape->capacity){
      printf("Writing more then expected origin co-ordinated in the given shape, something went wrong!!! \n");
      exit(1);
    }
    // Note: dim_o is assumed to be 2, if its not 2 then the following function need to be modified...
    // Add the new coordinate to the end of the array
    // printf("1co-ordinates: %d %d\n", y, x);
    shape->coordinates[shape->count].y = y;
    shape->coordinates[shape->count].x = x;
    shape->coordinates[shape->count].data_count = 1;

    // Allocate memory for dataptr (int pointer) and assign value
    shape->coordinates[shape->count].dataptr = (int*)malloc(sizeof(int));
    if (shape->coordinates[shape->count].dataptr == NULL) {
        fprintf(stderr, "Memory allocation for dataptr failed!\n");
        exit(1);  // Exit if memory allocation fails
    }
    *(shape->coordinates[shape->count].dataptr) = data_ptr;
    shape->count++;
}


// Function to sort the coordinates in lexicographical order based on y, then x
void sort_coordinates(ShapeOrigins* shape) {
    qsort(shape->coordinates, shape->count, sizeof(Coordinate), compare_coordinates);
}


int find_shape_by_coordinates(ShapeOrigins* shape, int current_idx, int next_row, int next_col) {
    for (int j = current_idx; j < shape->count; ++j) {
        if (shape->coordinates[j].y == next_row && shape->coordinates[j].x == next_col) {
            return j;
        }
    }
    return -1;
}



int binary_search_row(ShapeOrigins* shape, int next_row) {
    int left = 0;
    int right = shape->count - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (shape->coordinates[mid].y == next_row) {
            // We found the row, but there might be multiple coordinates with the same row
            // We want to find the first occurrence of this row.
            while (mid > 0 && shape->coordinates[mid - 1].y == next_row) {
                mid--;
            }
            return mid;  // Return the index of the first occurrence of the row, this is the start point to start search for x (column).
        } else if (shape->coordinates[mid].y < next_row) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;  // Row not found
}



int binary_search_column(ShapeOrigins* shape, int start_idx, int next_col, int row) {
    int left = start_idx;

    // Find the last index of the row
    int right = start_idx;
    while (right < shape->count && shape->coordinates[right].y == row) {
        right++;
    }
    right--;  // now right points to the last index of the row

    // Binary search within the column range
    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (shape->coordinates[mid].x == next_col) {
            return mid;  // Column found
        } else if (shape->coordinates[mid].x < next_col) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;  // Column not found
}



int find_coordinate(ShapeOrigins* shape, int next_row, int next_col) {
    // Step 1: Search for the row
    int row_start_idx = binary_search_row(shape, next_row);

    if (row_start_idx == -1) {
        return -1;  // Row not found
    }

    // Step 2: Search for the column within the row
    return binary_search_column(shape, row_start_idx, next_col, next_row);
}

void shuffle_coordinates(Coordinate* coordinates, int count) {
  for (int i = count - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    Coordinate temp = coordinates[i];
    coordinates[i] = coordinates[j];
    coordinates[j] = temp;
  }
}


void sort_origins_to_favour_locality(ShapeMap* new_map, ShapeOrigins* shape, int* hashmap_to_merge) {
  if (shape == NULL || shape->coordinates == NULL || shape->count <= 0) {
    return;
  }

  // Shuffle the coordinates to randomize their order within the shape
  shuffle_coordinates(shape->coordinates, shape->count);

  if (new_map->count >= new_map->capacity) {
    printf("the new shape holding capacity exceeded\n\n");
    exit(1);
  }

  // Add the shape to the new map
  new_map->count++;
  shape->shape_id = new_map->count - 1;
  new_map->shapes[new_map->count - 1] = (void*)shape;
  shape = NULL;
}

// void sort_origins_to_favour_locality(ShapeMap* new_map, ShapeOrigins* shape, int* hashmap_to_merge){
//   // printf("shape id: %d\n", shape->shape_id);
//   if (shape == NULL || shape->coordinates == NULL || shape->count <= 0) {
//     return;
//   }
//   qsort(shape->coordinates, shape->count, sizeof(Coordinate), compare_coordinates_xsort);
//   // for(int itr = 0; itr < shape->count; itr++){
//   //   printf("%d %d\n", shape->coordinates[itr].y, shape->coordinates[itr].x);
//   // }

//   if(new_map->count >= new_map->capacity){
//     printf("the new shape holding capacity exceeded\n\n");
//     exit(1);
//   }
//   new_map->count++;
//   shape->shape_id = new_map->count - 1;
//   new_map->shapes[new_map->count - 1] = (void*)shape;
//   shape = NULL;
// }


void merge_shapes(ShapeMap* new_map, ShapeOrigins* shape, int* hashmap_to_merge) {
    int flag_merge_not_happened = 1;
    // Iterate over hashmap_to_merge in reverse order
    for (int j = shape->count - 1; j >= 0; --j)
    {
        if (hashmap_to_merge[j] != -1) {
            flag_merge_not_happened = 0;
            int merge_idx = hashmap_to_merge[j];
            // Merge the coordinates by collecting the data pointers
            if(((shape->coordinates[j].data_count + shape->coordinates[merge_idx].data_count)*(shape->end_vertex + 1)) < 128)
            {
              int new_data_count = shape->coordinates[j].data_count + shape->coordinates[merge_idx].data_count;
              shape->coordinates[j].dataptr = (int*)realloc(shape->coordinates[j].dataptr, new_data_count * sizeof(int));
              if (shape->coordinates[j].dataptr == NULL) {
                  perror("Failed to allocate memory for data pointers");
                  exit(EXIT_FAILURE);
              }
              // Copy the data pointers from the merged coordinate
              for (int k = 0; k < shape->coordinates[merge_idx].data_count; ++k) {
                  shape->coordinates[j].dataptr[shape->coordinates[j].data_count + k] = shape->coordinates[merge_idx].dataptr[k];
              }
              shape->coordinates[j].data_count = new_data_count;

              free(shape->coordinates[merge_idx].dataptr); // Free the merged data pointers
              shape->coordinates[merge_idx].dataptr = NULL; // Mark as merged
              shape->coordinates[merge_idx].data_count = 0; // Mark as merged
            }
        }
    }

    if(flag_merge_not_happened){
      if(new_map->count >= new_map->capacity){
        printf("the new shape holding capacity exceeded\n\n");
        exit(1);
      }
      new_map->count++;
      shape->shape_id = new_map->count - 1;
      new_map->shapes[new_map->count - 1] = (void*)shape;
      shape = NULL;
    }
    else{
      // Process the hashmap_to_merge to create or add to shapes in new_map
      for (int j = 0; j < shape->count; ++j) {
          if (shape->coordinates[j].data_count != 0) {
              int flag = 1;
              int current_shape_end_vertex = (shape->end_vertex + 1)*shape->coordinates[j].data_count - 1;
              for (int i = 0; i < new_map->count; i++) {
                  ShapeOrigins* existing_shape = (ShapeOrigins*)new_map->shapes[i];
                  if (existing_shape->end_vertex == current_shape_end_vertex &&
                      existing_shape->lattice_y == shape->lattice_y &&
                      existing_shape->lattice_x == shape->lattice_x) {
                      flag = 0;
                      existing_shape->count++;
                      existing_shape->coordinates = (Coordinate*)realloc(existing_shape->coordinates, existing_shape->count * sizeof(Coordinate));
                      if (existing_shape->coordinates == NULL) {
                          perror("Failed 1 to allocate memory for coordinates");
                          exit(EXIT_FAILURE);
                      }
                      existing_shape->coordinates[existing_shape->count - 1].data_count = shape->coordinates[j].data_count;
                      existing_shape->coordinates[existing_shape->count - 1].y = shape->coordinates[j].y;
                      existing_shape->coordinates[existing_shape->count - 1].x = shape->coordinates[j].x;
                      existing_shape->coordinates[existing_shape->count - 1].dataptr = (int*)shape->coordinates[j].dataptr;
                      shape->coordinates[j].dataptr = NULL;
                      shape->coordinates[j].data_count = 0;
                      break;
                  }
              }

              if (flag) {
                  if(new_map->count >= new_map->capacity){
                    printf("the new shape holding capacity exceeded\n\n");
                    exit(1);
                  }
                  new_map->count++;
                  // Allocate memory for the new ShapeOrigins
                  ShapeOrigins* new_shape_ptr = (ShapeOrigins*)malloc(sizeof(ShapeOrigins));
                  if (new_shape_ptr == NULL) {
                      perror("Failed to allocate memory for new ShapeOrigins");
                      exit(EXIT_FAILURE);
                  }
                  new_shape_ptr->shape_id = new_map->count - 1;
                  new_shape_ptr->lattice_y = shape->lattice_y;
                  new_shape_ptr->lattice_x = shape->lattice_x;
                  new_shape_ptr->count = 1;
                  new_shape_ptr->capacity = 1;
                  new_shape_ptr->coordinates = (Coordinate*)malloc(sizeof(Coordinate)); // Initialize with the first coordinate

                  if (new_shape_ptr->coordinates == NULL) {
                      perror("Failed 3 to allocate memory for coordinates");
                      exit(EXIT_FAILURE);
                  }

                  new_shape_ptr->coordinates[0].data_count = shape->coordinates[j].data_count;
                  new_shape_ptr->coordinates[0].y = shape->coordinates[j].y;
                  new_shape_ptr->coordinates[0].x = shape->coordinates[j].x;
                  new_shape_ptr->coordinates[0].dataptr = (int*)shape->coordinates[j].dataptr;
                  shape->coordinates[j].dataptr = NULL;
                  shape->coordinates[j].data_count = 0;

                  new_shape_ptr->start_vertex = shape->start_vertex;
                  new_shape_ptr->end_vertex = current_shape_end_vertex;
                  new_map->shapes[new_map->count - 1] = (void*)new_shape_ptr;
              }
          }
      }
      free(shape->coordinates);
      shape->coordinates = NULL;
      free(shape);
      shape = NULL;
    }// else end
}


s_spf_structure_t* aggregrate_the_origins(s_spf_structure_t* spf_matrix)
{
  int data_is_float = 0;
  if (!spf_matrix) {
    if (DEBUG) {
      printf("Error: spf_matrix is NULL\n");
    }
    return (s_spf_structure_t*)-1;  // Assuming you want to return an error pointer here
  }

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

  ShapeMap origin_map;
  origin_map.count = 0;
  origin_map.shapes = (void **)malloc(nbase_shapes * sizeof(void *));  // Allocate memory once for all shapes
  if (origin_map.shapes == NULL) {
    fprintf(stderr, "Memory allocation failed for origin_map.shapes\n");
    exit(EXIT_FAILURE);  // Handle allocation failure
  }

  // Initialize all pointers to NULL for safety
  for (int i = 0; i < nbase_shapes; i++) {
    origin_map.shapes[i] = NULL;
  }

  if (DEBUG) {
    printf("Initializing ShapeMap and beginning shape aggregation\n");
  }

  for (int i = 0; i < nbase_shapes; ++i) {
    void* shape1 = spf_matrix->shape_dictionnary.shapes[i];
    s_codelet_1x1d_t* as_1d = (s_codelet_1x1d_t*)shape1;
    short dim_p = as_1d->dim_p;
    int num_shape_occurrences = spf_matrix->shape_dictionnary.shapes_nb_origins[i];
    shape_to_reg_2x1(shape1, shape_reg);
    int shape_id = i;

    create_shape(&origin_map, shape_id, shape_reg.lattice[0][0], shape_reg.lattice[1][0],
                          shape_reg.start_vertex[0], shape_reg.end_vertex[0], num_shape_occurrences);
    if (DEBUG) {
      printf("Processed shape ID %d: lattice = [%d, %d], occurrences = %d\n", shape_id, shape_reg.lattice[0][0], 
             shape_reg.lattice[1][0], num_shape_occurrences);
    }
  }

  int norigins = spf_matrix->origins_list.norigins;
  if (DEBUG) {
    printf("Number of origins: %d\n", norigins);
  }

  ShapeOrigins* shape = NULL;
  for (int i = 0; i < norigins; ++i) {
    s_origin_2d_t origin_reg = ((s_origin_2d_t*)spf_matrix->origins_list.origins)[i];

    int shape_id = origin_reg.shape_id;
    if(shape == NULL || shape->shape_id != shape_id){
        // Get the shape
        shape = find_shape(&origin_map, shape_id);
        if (shape == NULL) {
            fprintf(stderr, "Shape with ID %d not found!\n", shape_id);
            exit(1); // Handle the error as appropriate
        }
    }

    // Add coordinates to the shape entry
    add_coordinates(shape, origin_reg.coordinates[0], origin_reg.coordinates[1], origin_reg.dataptr);

    if (DEBUG) {
      printf("Added coordinates for shape ID %d: [%d, %d]\n", shape_id, origin_reg.coordinates[0], origin_reg.coordinates[1]);
    }
  }
  
  // the data is already lexicographically sorted
  //   // Sort all coordinates for each shape_id
  // for (int i = 0; i < origin_map.count; ++i) {
  //     sort_coordinates(&origin_map.shapes[i]);
  // }

  ShapeMap new_map;
  new_map.count = 0;
  new_map.shapes = (void **)malloc(2000 * sizeof(void *));  // Allocate memory once for all shapes
  new_map.capacity = 2000;

  if (new_map.shapes == NULL) {
    fprintf(stderr, "Memory allocation failed for new_map.shapes\n");
    exit(EXIT_FAILURE);
  }

  // Initialize all pointers to NULL for safety
  for (int i = 0; i < 2000; i++) {
    new_map.shapes[i] = NULL;
  }

  for (int i = 0; i < origin_map.count; ++i) {
    ShapeOrigins* shape = (ShapeOrigins *)origin_map.shapes[i];

    // Initialize hashmap_to_merge with -1 (indicating no match initially)
    int *hashmap_to_merge = (int*)malloc(shape->count * sizeof(int));
    if (hashmap_to_merge == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 0;
    }
    memset(hashmap_to_merge, -1, shape->count * sizeof(int));

    // omp_set_num_threads(4);
    // #pragma omp parallel for
    if(FIND_CONSECUTIVE_ORIGINS){
      #pragma omp parallel for shared(hashmap_to_merge, shape)
      for (int j = 0; j < shape->count; ++j) {
        if (DEBUG) {
          #pragma omp critical
          {
            printf("current: %d %d ", shape->coordinates[j].y, shape->coordinates[j].x);
          }
        }

        // Predict the next coordinate based on the lattice and end_vertex
        int next_coordinate_row = shape->coordinates[j].y + shape->lattice_y * (shape->end_vertex + 1);
        int next_coordinate_column = shape->coordinates[j].x + shape->lattice_x * (shape->end_vertex + 1);

        if (DEBUG) {
          #pragma omp critical
          {
            printf("next: %d %d\n", next_coordinate_row, next_coordinate_column);
          }
        }

        // Find the shape index using the predicted next coordinates
        // int index = find_shape_by_coordinates(shape, j, next_coordinate_row, next_coordinate_column);
        // Perform the two-step binary search for the next coordinate
        int index = find_coordinate(shape, next_coordinate_row, next_coordinate_column);

        // If a matching shape is found, update the hashmap_to_merge array
        if (index != -1) {
          hashmap_to_merge[j] = index;
        }
      }
    }

    if (DEBUG) {
      // Debug output for the hashmap to merge
      printf("----------------------------------------------------------------------------------\n");
      printf("hashmap_to_merge: ");
      for (int j = 0; j < shape->count; ++j) {
        printf("%d ", hashmap_to_merge[j]);
      }
      printf("\n");
    }

    if(FIND_CONSECUTIVE_ORIGINS){
      // Merge the shapes using the generated hashmap
      merge_shapes(&new_map, shape, hashmap_to_merge);
    }
    else
    {
      sort_origins_to_favour_locality(&new_map, shape, hashmap_to_merge);
    }

    free(hashmap_to_merge);
  }

  // sort the shapes in the ascending order:
  for(int i = 0; i < new_map.count; i++){
      ShapeOrigins* prime_shape = (ShapeOrigins*)new_map.shapes[i];
    for(int j = i + 1; j < new_map.count; j++){
      ShapeOrigins* current_shape = (ShapeOrigins*)new_map.shapes[j];
      if(prime_shape->end_vertex < current_shape->end_vertex){
        void * temp = new_map.shapes[i];
        new_map.shapes[i] = new_map.shapes[j];
        new_map.shapes[j] = temp;
        prime_shape = (ShapeOrigins*)new_map.shapes[i];
      }
    }
    prime_shape->shape_id = i;
  }




/*
  int start_index = 0;
  for(int i = 0; i < 6; i++) {
      ShapeOrigins* prime_shape = (ShapeOrigins*)new_map.shapes[start_index];
      // Break if at the last shape
      if(start_index >= new_map.count) {
          break;
      }

      for(int j = start_index; j < new_map.count; j++) {
          if(start_index >= new_map.count) {
              break;
          }

          ShapeOrigins* current_shape = (ShapeOrigins*)new_map.shapes[j];
          int lattice_y = current_shape->lattice_y;
          int lattice_x = current_shape->lattice_x;
          int stride_value;

          // Determine stride_value based on lattice values
          if(lattice_y == 0 && lattice_x != 0) {
              stride_value = abs(lattice_x);
          }
          else if(lattice_y != 0 && lattice_x == 0) {
              stride_value = abs(lattice_y);
          }
          else {
              stride_value = abs(lattice_y) > abs(lattice_x) ? abs(lattice_y) : abs(lattice_x);
          } 

          // Check conditions and swap shapes based on type and stride
          if(lattice_y == 0 && lattice_x != 0 && stride_value == 1 && i == 2) {
              swap_and_update_prime(new_map.shapes, &start_index, j); // Horizontal shape
          }
          else if(lattice_y != 0 && lattice_x == 0 && stride_value == 1 && i == 1) {
              swap_and_update_prime(new_map.shapes, &start_index, j); // Vertical shape
          }
          else if(lattice_y != 0 && lattice_x != 0 && stride_value == 1 && i == 0) {
              swap_and_update_prime(new_map.shapes, &start_index, j); // Diagonal shape
          }
          else if(lattice_y == 0 && lattice_x != 0 && stride_value != 1 && i == 5) {
              swap_and_update_prime(new_map.shapes, &start_index, j); // Horizontal with stride shape
          }
          else if(lattice_y != 0 && lattice_x == 0 && stride_value != 1 && i == 4) {
              swap_and_update_prime(new_map.shapes, &start_index, j); // Vertical with stride shape
          }
          else if(lattice_y != 0 && lattice_x != 0 && stride_value != 1 && i == 3) {
              swap_and_update_prime(new_map.shapes, &start_index, j); // Diagonal with stride shape
          }
      }
  }
*/

  // for(int i = 0; i < new_map.count; i++){
  //     ShapeOrigins* prime_shape = (ShapeOrigins*)new_map.shapes[i];
  //     prime_shape->shape_id = i;
  //     new_map.shapes[i] = (void*)prime_shape;
  //     // printf("shape id and values: %d %d %d\n", prime_shape->shape_id, prime_shape->end_vertex, prime_shape->count);
  // }
  

  // for(int i = 0; i < new_map.count; i++){
  //    ShapeOrigins* prime_shape = (ShapeOrigins*)new_map.shapes[i];
  //    printf("shape id and values: %d %d %d\n", prime_shape->shape_id, prime_shape->end_vertex, prime_shape->count);
  // }

   
  // update the shape dictionary of the spf matrix
  int old_nbase_shapes = spf_matrix->matrix_description.nbase_shapes;
  spf_matrix->matrix_description.nbase_shapes = new_map.count;
  spf_matrix->shape_dictionnary.nbase_shapes = new_map.count;
  nbase_shapes = new_map.count;
  
  int shape_bytes_size = sizeof(s_codelet_2x1d_t);
  int origin_byte_size = sizeof(s_origin_2d_t);

  if (spf_matrix->shape_dictionnary.shapes_nb_origins != NULL) {
    free(spf_matrix->shape_dictionnary.shapes_nb_origins);
  }
  spf_matrix->shape_dictionnary.shapes_nb_origins = (int*)malloc(nbase_shapes * sizeof(int));

  if (spf_matrix->shape_dictionnary.shapes_key != NULL) {
    free(spf_matrix->shape_dictionnary.shapes_key);
  }
  spf_matrix->shape_dictionnary.shapes_key = (int*)malloc(nbase_shapes * sizeof(int));


  for(int i = 0; i < old_nbase_shapes; i++) {
    free(spf_matrix->shape_dictionnary.shapes[i]);
  }

  spf_matrix->shape_dictionnary.shapes = malloc (nbase_shapes * sizeof(void*));

  norigins = 0;
  for(int i = 0; i < new_map.count; i++){
    ShapeOrigins *shape = (ShapeOrigins *)new_map.shapes[i];
    norigins += shape->count;
  }

  // printf("old and new norigins: %d %d\n", spf_matrix->origins_list.norigins, norigins);
  spf_matrix->origins_list.norigins = norigins;

  if(spf_matrix->origins_list.origins != NULL) {
    free(spf_matrix->origins_list.origins);
    spf_matrix->origins_list.origins = NULL;
  }
  spf_matrix->origins_list.origins = malloc (norigins * origin_byte_size);
  int global_data_ptr_offset = 0;
  int origin_index = 0;
  int hi = 0;

  double* new_data = (double *)polybench_alloc_data(nnz, sizeof(double));
  double* const data_vector = spf_matrix->data;

  for(int i = 0; i < new_map.count; i++) {
    ShapeOrigins* shape = (ShapeOrigins*)new_map.shapes[i];
    int shape_id = i;
    short shape_encoding = 0;
    short dim_p = 1;
      
    s_codelet_2x1d_t* shape_ptr = (s_codelet_2x1d_t*)malloc(shape_bytes_size);
    shape_ptr->dim_p = dim_p;
    shape_ptr->start_vertex[0] = shape->start_vertex;
    shape_ptr->end_vertex[0] = shape->end_vertex;
    shape_ptr->stride[0] = 1;
    shape_ptr->lattice[0][0] = shape->lattice_y;
    shape_ptr->lattice[1][0] = shape->lattice_x;

    spf_matrix->shape_dictionnary.shapes[i] = (int*)shape_ptr;
    spf_matrix->shape_dictionnary.shapes_nb_origins[i] = shape->count;
    spf_matrix->shape_dictionnary.shapes_key[i] = 0;

    for(int j = 0; j < shape->count; j++) {
      // printf("shape_id %d %d %d\n", shape_id, shape->coordinates[j].y, shape->coordinates[j].x);
      ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].shape_id = (short)shape_id;
      ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].coordinates[0] = shape->coordinates[j].y;
      ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].coordinates[1] = shape->coordinates[j].x;
      ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].dataptr = global_data_ptr_offset;
      origin_index++;

      int block_size = (shape->end_vertex + 1)/shape->coordinates[j].data_count;
      for(int k = 0; k < shape->coordinates[j].data_count; k++) {
        // printf("co-ordinates offsets: %d %d\n", k, *(shape->coordinates[j].dataptr + k));
        memcpy(&new_data[global_data_ptr_offset], &data_vector[*(shape->coordinates[j].dataptr + k)], block_size * sizeof(double));
        global_data_ptr_offset += block_size;
      }
    }
  }

  int uninc_nnz = nnz - inc_nnz;
  memcpy(&new_data[global_data_ptr_offset], &data_vector[global_data_ptr_offset], uninc_nnz * sizeof(double));

  spf_matrix->data = new_data;
  free(data_vector);

  for (int i = 0; i < new_map.count; ++i) {
    if (new_map.shapes[i]) {
      ShapeOrigins* shape = (ShapeOrigins*)new_map.shapes[i];
      if (shape) {

        // Free each coordinate's dataptr before freeing coordinates
        if (shape->coordinates) {
          for (int j = 0; j < shape->count; ++j) {
            if (shape->coordinates[j].dataptr != NULL) {
              free(shape->coordinates[j].dataptr);  // Free dataptr
              shape->coordinates[j].dataptr = NULL; // Set to NULL to avoid accidental reuse
            }
          }
          free(shape->coordinates);  // Free the coordinates array
          shape->coordinates = NULL; // Set to NULL to avoid dangling pointer
        }

        free(shape);  // Free the ShapeOrigins struct itself
        new_map.shapes[i] = NULL;  // Set to NULL to avoid dangling pointer
      }
    }
  }

  // free the complete maps
  free(new_map.shapes);
  free(origin_map.shapes);

  return spf_matrix;
}

static void usage(char** argv)
{
  fprintf (stderr, "Usage: %s [-float,-stats] <spf_matrix.spfdata>\n", argv[0]);
  exit (1);
}

long get_file_size(const char *file_path) {
    struct stat st;
    
    // Use stat to get file info
    if (stat(file_path, &st) == 0) {
        return st.st_size; // File size in bytes
    } else {
        perror("stat failed");
        return -1; // Error case
    }
}


int main(int argc, char** argv)
{
  if( argc < 2 ) {
    printf( "Usage: spmv_spf <SPF file>\n" );
    exit(0);
  }

  int debug = 0;
  int data_is_float = 0;
  int print_matrix_summary = debug;
  int print_matrix_structure = 0 || (debug > 1);

  char* rbpath = NULL;
  char* dst_folder = "results";
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


    long file_size = get_file_size(rbpath);
    if (file_size != -1) {
        printf("File size: %ld bytes\n", file_size);
    }

  // Load the matrix.
  s_spf_structure_t* spf_mat = spf_matrix_read_from_file(rbpath);
  spf_mat = aggregrate_the_origins(spf_mat);

  // Write the aggregated matrix to a file
  char* file_name = strrchr(rbpath, '/');
  // Combine destination folder and file name into the final path
  char dst_path[1024]; // in future Adjust size according to the required size
  snprintf(dst_path, sizeof(dst_path), "%s%s", dst_folder, file_name);
  // write the spf_mat into the file
  spf_matrix_write_to_file(spf_mat, dst_path);

  // now deallocate the memeory we have allocated for spf_mat
  free(spf_mat->shape_dictionnary.shapes_nb_origins);
  free(spf_mat->shape_dictionnary.shapes_key);
  for(int i = 0; i < spf_mat->shape_dictionnary.nbase_shapes; i++){
    free(spf_mat->shape_dictionnary.shapes[i]);
  }
  free(spf_mat->shape_dictionnary.shapes);
  free(spf_mat->origins_list.origins);
  free(spf_mat->data);
  free(spf_mat);

  return 0;
}
