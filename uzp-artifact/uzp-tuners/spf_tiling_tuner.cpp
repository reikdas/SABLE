#include <cstddef>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdio>

#include <cstdlib>
#include <unistd.h> // For read()
#include <cstdio>

#include <sys/stat.h>
#include <map>
#include <set>
#include <utility> // For std::pair
#include <map>
#include <set>
#include <tuple>
#include <cstdlib>  // For std::atoi

int TILE_SIZE = 2048;

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



// Define structs with packed attribute
struct __attribute__((packed)) s_codelet_1x1d {
    int dim_p;
    int start_vertex[1];
    int end_vertex[1];
    int stride[1];
    int lattice[1][1];
};
using s_codelet_1x1d_t = s_codelet_1x1d;

struct __attribute__((packed)) s_codelet_2x1d {
    int dim_p;
    int start_vertex[1];
    int end_vertex[1];
    int stride[1];
    int lattice[2][1];
};
using s_codelet_2x1d_t = s_codelet_2x1d;

struct __attribute__((packed)) s_codelet_3x1d {
    int dim_p;
    int start_vertex[1];
    int end_vertex[1];
    int stride[1];
    int lattice[3][1];
};
using s_codelet_3x1d_t = s_codelet_3x1d;

struct __attribute__((packed)) s_codelet_1x2d {
    int dim_p;
    int start_vertex[2];
    int end_vertex[2];
    int stride[2];
    int lattice[1][2];
};
using s_codelet_1x2d_t = s_codelet_1x2d;

struct __attribute__((packed)) s_codelet_2x2d {
    int dim_p;
    int start_vertex[2];
    int end_vertex[2];
    int stride[2];
    int lattice[2][2];
};
using s_codelet_2x2d_t = s_codelet_2x2d;

struct __attribute__((packed)) s_codelet_3x2d {
    int dim_p;
    int start_vertex[2];
    int end_vertex[2];
    int stride[2];
    int lattice[3][2];
};
using s_codelet_3x2d_t = s_codelet_3x2d;

struct __attribute__((packed)) s_codelet_1x3d {
    int dim_p;
    int start_vertex[3];
    int end_vertex[3];
    int stride[3];
    int lattice[1][3];
};
using s_codelet_1x3d_t = s_codelet_1x3d;

struct __attribute__((packed)) s_codelet_2x3d {
    int dim_p;
    int start_vertex[3];
    int end_vertex[3];
    int stride[3];
    int lattice[2][3];
};
using s_codelet_2x3d_t = s_codelet_2x3d;

struct __attribute__((packed)) s_codelet_3x3d {
    int dim_p;
    int start_vertex[3];
    int end_vertex[3];
    int stride[3];
    int lattice[3][3];
};
using s_codelet_3x3d_t = s_codelet_3x3d;

struct __attribute__((packed)) s_matrix_desc {
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
using s_matrix_desc_t = s_matrix_desc;

// Shapes dictionary
struct s_shape_dictionnary {
    int nbase_shapes;
    int* nshapes_dim_p; // Array of size dim_p, allocated dynamically
    void** shapes;
    int* shapes_nb_origins;
    int* shapes_key;
};
using s_shape_dictionnary_t = s_shape_dictionnary;

struct __attribute__((packed)) s_origin_1d {
    short shape_id;
    int coordinates[1];
    int dataptr;
};
using s_origin_1d_t = s_origin_1d;

struct __attribute__((packed)) s_origin_2d {
    short shape_id;
    int coordinates[2];
    int dataptr;
};
using s_origin_2d_t = s_origin_2d;

struct __attribute__((packed)) s_origin_3d {
    short shape_id;
    int coordinates[3];
    int dataptr;
};
using s_origin_3d_t = s_origin_3d;

struct s_origin_list {
    short dim_o;
    int norigins;
    void* origins;
};
using s_origin_list_t = s_origin_list;

struct s_unincorporated_points_csr {
    unsigned char uninc_format;
    long long* rowptr; // nrow + 1
    long long* colidx; // uninc_nnz
};
using s_unincorporated_points_csr_t = s_unincorporated_points_csr;

struct s_spf_structure {
    s_matrix_desc_t matrix_description;
    s_shape_dictionnary_t shape_dictionnary;
    s_origin_list_t origins_list;
    s_unincorporated_points_csr_t unincorporated_points;
    void* data;
    int data_is_float;
};
using s_spf_structure_t = s_spf_structure;


inline double* allocate_data(int size) {
    return new double[size];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
s_spf_structure_t* spf_matrix_read_from_file(const char* spfpath) {
    int debug = 0;

    // Open file
    if (debug) {
        std::cout << "Reading file: " << spfpath << std::endl;
    }

    int fd = open(spfpath, O_RDONLY);
    if (fd == -1) {
        return nullptr;
    }

    s_spf_structure_t* spf_matrix = static_cast<s_spf_structure_t*>(std::malloc(sizeof(s_spf_structure_t)));
    if (!spf_matrix) {
        close(fd);
        return nullptr;
    }

    spf_matrix->data_is_float = false;

    // Read matrix description
    read(fd, &(spf_matrix->matrix_description), sizeof(s_matrix_desc_t));

    int dim_p = spf_matrix->matrix_description.dim_p;
    int dim_o = spf_matrix->matrix_description.dim_o;
    int nnz = spf_matrix->matrix_description.nnz;
    int inc_nnz = spf_matrix->matrix_description.inc_nnz;
    int intnrow = spf_matrix->matrix_description.intnrow;
    int ncols = spf_matrix->matrix_description.ncols;
    int nbase_shapes = spf_matrix->matrix_description.nbase_shapes;
    int nhierch_shapes = spf_matrix->matrix_description.nhierch_shapes;
    int data_start = spf_matrix->matrix_description.data_start;

    if (debug) {
        std::cout << "Reading matrix:\n"
                  << "\tNNZ       = " << nnz << "\n"
                  << "\tINC       = " << inc_nnz << "\n"
                  << "\tdims      = " << intnrow << " x " << ncols << "\n"
                  << "\tdim_o     = " << dim_o << "\n"
                  << "\tBase sh   = " << nbase_shapes << "\n"
                  << "\tHierch sh = " << nhierch_shapes << "\n"
                  << "\tData ptr  = " << data_start << "\n"
                  << "\tMax dim_p = " << dim_p << std::endl;
    }

    // Allocate memory for nshapes_dim_p
    spf_matrix->shape_dictionnary.nshapes_dim_p = static_cast<int*>(std::malloc(dim_p * sizeof(int)));
    int* nshapes_dim_p = spf_matrix->shape_dictionnary.nshapes_dim_p;
    
    read(fd, spf_matrix->shape_dictionnary.nshapes_dim_p, dim_p * sizeof(int));

    if (debug) {
        std::cout << "\tShapes/dim= [ ";
        for (int i = 0; i < dim_p; ++i) {
            std::cout << nshapes_dim_p[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    // Read shapes
    spf_matrix->shape_dictionnary.nbase_shapes = nbase_shapes;
    spf_matrix->shape_dictionnary.shapes = static_cast<void**>(std::malloc(nbase_shapes * sizeof(void*)));
    spf_matrix->shape_dictionnary.shapes_nb_origins = static_cast<int*>(std::malloc(nbase_shapes * sizeof(int)));
    spf_matrix->shape_dictionnary.shapes_key = static_cast<int*>(std::malloc(nbase_shapes * sizeof(int)));

    if (debug) std::cout << "\nReading basic shapes:\n";

    for (int i = 0; i < nbase_shapes; ++i) {
        short shape_id, shape_encoding, shape_dim_p;

        read(fd, &shape_id, sizeof(short));
        read(fd, &shape_encoding, sizeof(short));
        read(fd, &shape_dim_p, sizeof(short));

        if (debug) {
            std::cout << "\tShape #" << i << std::endl;
            std::cout << "\t\tEncoding type: " << shape_encoding << std::endl;
            std::cout << "\t\tdim(p)       : " << shape_dim_p << std::endl;
        }

        if (shape_encoding != 0) {
            std::cerr << "Other formats for shapes not supported yet!" << std::endl;
            std::exit(51);
        }

        int shape_bytes_size = 0;

        if (dim_o == 1) {
            switch (shape_dim_p) {
                case 1: shape_bytes_size = sizeof(s_codelet_1x1d_t); break;
                case 2: shape_bytes_size = sizeof(s_codelet_1x2d_t); break;
                case 3: shape_bytes_size = sizeof(s_codelet_1x3d_t); break;
                default: std::exit(48);
            }
        } else if (dim_o == 2) {
            switch (shape_dim_p) {
                case 1: shape_bytes_size = sizeof(s_codelet_2x1d_t); break;
                case 2: shape_bytes_size = sizeof(s_codelet_2x2d_t); break;
                case 3: shape_bytes_size = sizeof(s_codelet_2x3d_t); break;
                default: std::exit(48);
            }
        } else if (dim_o == 3) {
            switch (shape_dim_p) {
                case 1: shape_bytes_size = sizeof(s_codelet_3x1d_t); break;
                case 2: shape_bytes_size = sizeof(s_codelet_3x2d_t); break;
                case 3: shape_bytes_size = sizeof(s_codelet_3x3d_t); break;
                default: std::exit(48);
            }
        } else {
            std::exit(47);
        }


//        printf("shape_bytes_size %d\n", shape_bytes_size);
        int* shape_ptr_short = (int*)(std::malloc(shape_bytes_size));
        if (!shape_ptr_short) {
            std::cerr << "Memory allocation failed for shape_ptr_short!" << std::endl;
            std::exit(1);
        }

        shape_ptr_short++;  // Move the pointer ahead
        shape_bytes_size -= sizeof(int);  // Adjust remaining size

        read(fd, shape_ptr_short, shape_bytes_size);

        shape_ptr_short--;  // Move pointer back
        shape_ptr_short[0] = dim_p;  // Store dim_p as short

        spf_matrix->shape_dictionnary.shapes[i] = shape_ptr_short;
        spf_matrix->shape_dictionnary.shapes_nb_origins[i] = 0;
        spf_matrix->shape_dictionnary.shapes_key[i] = 0;
    }

    int norigins;
    read(fd, &norigins, sizeof(int));

    spf_matrix->origins_list.dim_o = dim_o;
    spf_matrix->origins_list.norigins = norigins;
    int origin_byte_size = 0;

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
            std::exit(52);
    }

    spf_matrix->origins_list.origins = malloc(norigins * origin_byte_size);
    void* origin_buffer = malloc(origin_byte_size);

    if (debug) {
        std::printf("\nReading %d shape origins:\n", norigins);
    }

    for (int i = 0; i < norigins; ++i) {
        read(fd, origin_buffer, origin_byte_size);
        int shape_id = ((s_origin_1d_t*)origin_buffer)->shape_id;

        // Error in shape id.
        if (shape_id < 0) {
            std::exit(189);
        }

        switch (dim_o) {
            case 1:
                ((s_origin_1d_t*)spf_matrix->origins_list.origins)[i] =
                    *((s_origin_1d_t*)origin_buffer);
                spf_matrix->shape_dictionnary.shapes_nb_origins
                    [((s_origin_1d_t*)spf_matrix->origins_list.origins)[i].shape_id]++;
                break;

            case 2:
                ((s_origin_2d_t*)spf_matrix->origins_list.origins)[i] =
                    *((s_origin_2d_t*)origin_buffer);
                spf_matrix->shape_dictionnary.shapes_nb_origins
                    [((s_origin_2d_t*)spf_matrix->origins_list.origins)[i].shape_id]++;
                break;

            case 3:
                ((s_origin_3d_t*)spf_matrix->origins_list.origins)[i] =
                    *((s_origin_3d_t*)origin_buffer);
                spf_matrix->shape_dictionnary.shapes_nb_origins
                    [((s_origin_3d_t*)spf_matrix->origins_list.origins)[i].shape_id]++;
                break;

            default:
                std::exit(53);
        }
    }
    free(origin_buffer);


    // Read unincorporated format: CSR, COO
    read(fd, &(spf_matrix->unincorporated_points.uninc_format), sizeof(unsigned char));
    int uninc_format = spf_matrix->unincorporated_points.uninc_format;

    if (debug) {
        std::cout << "\nUnincorporated metadata:\n";
        std::cout << "\tUnincorporated format: " << uninc_format << " (0:CSR, 1: CSC, 2:COO)\n";
    }

    // Read CSR/COO data
    int uninc_nnz = nnz - inc_nnz;
    if (uninc_nnz == 0) {
        spf_matrix->unincorporated_points.rowptr = nullptr;
        spf_matrix->unincorporated_points.colidx = nullptr;
    } 
    else if (uninc_format == 0) 
    {
        int* tmprow = static_cast<int*>(malloc(sizeof(int) * (intnrow + 1)));
        int* tmpcol = static_cast<int*>(malloc(sizeof(int) * uninc_nnz));
        long long* rowptr = static_cast<long long*>(malloc(sizeof(long long) * (intnrow + 1)));
        long long* colidx = static_cast<long long*>(malloc(sizeof(long long) * uninc_nnz));

        read(fd, tmprow, (intnrow + 1) * sizeof(int));
        read(fd, tmpcol, uninc_nnz * sizeof(int));

        for (int i = 0; i < intnrow + 1; ++i) {
            rowptr[i] = tmprow[i];
        }
        for (int i = 0; i < uninc_nnz; ++i) { // Convert from int to long long
            colidx[i] = tmpcol[i];
        }

        if (debug > 4) {
            std::cout << "\tRowptr: [ ";
            for (int i = 0; i < intnrow + 1; ++i) {
                std::cout << tmprow[i] << ", ";
            }
            std::cout << "]\n";
            std::cout << "\tColptr: [ ";
            for (int i = 0; i < uninc_nnz; ++i) {
                std::cout << tmpcol[i] << ", ";
            }
            std::cout << "]\n";
        }

        rowptr[intnrow] = uninc_nnz;
        free(tmprow);
        free(tmpcol);
        spf_matrix->unincorporated_points.rowptr = rowptr;
        spf_matrix->unincorporated_points.colidx = colidx;
    }
    else if (uninc_format == 2)
    {
        int* tmprow = static_cast<int*>(malloc(sizeof(int) * uninc_nnz));
        int* tmpcol = static_cast<int*>(malloc(sizeof(int) * uninc_nnz));
        long long* rowptr = static_cast<long long*>(malloc(sizeof(long long) * uninc_nnz));
        long long* colidx = static_cast<long long*>(malloc(sizeof(long long) * uninc_nnz));

        read(fd, tmprow, uninc_nnz * sizeof(int));
        read(fd, tmpcol, uninc_nnz * sizeof(int));

        for (int i = 0; i < uninc_nnz; ++i) { // Convert from int to long long
            rowptr[i] = tmprow[i];
            colidx[i] = tmpcol[i];
        }

        if (debug > 4) {
            std::cout << "\tLocations of unincorporated points: [ ";
            for (int i = 0; i < uninc_nnz; ++i) {
                std::cout << "( " << tmprow[i] << ", " << tmpcol[i] << " ), ";
            }
            std::cout << "]\n";
        }

        free(tmprow);
        free(tmpcol);
        spf_matrix->unincorporated_points.rowptr = rowptr;
        spf_matrix->unincorporated_points.colidx = colidx;
    }

    // Allocate data and read
    double* A = allocate_data(nnz);
    int bytes_read = read(fd, A, nnz * sizeof(double));

    if (debug > 5) {
        std::cout << "\nData section: ";
        for (int i = 0; i < nnz; ++i) {
            std::cout << A[i] << " ";
        }
        std::cout << "\n";

        std::cout << "\nFile ended at offset: " << lseek(fd, 0, SEEK_CUR) << std::endl;
        std::cout << "\nTotal number of doubles read: " << static_cast<double>(bytes_read) / sizeof(double) << std::endl;
    }

    close(fd);

    spf_matrix->data = A;

    // // Post-processing
    // spf_matrix_shapes_key_set(spf_matrix);

    return spf_matrix;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
            return -1;
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
    double* A = static_cast<double*>(spf_matrix->data);
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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void usage(char** argv)
{
  fprintf (stderr, "Usage: %s [-float,-stats] [--tile <tile_size>] <spf_matrix.spfdata> <output_tuned.spfdata>\n", argv[0]);
  exit (1);
}



long get_file_size(const char* file_path) {
    struct stat st;
    
    // Use stat to get file info
    if (stat(file_path, &st) == 0) {
        return st.st_size; // File size in bytes
    } else {
        std::cerr << "stat failed: " << strerror(errno) << std::endl;
        return -1; // Error case
    }
}


std::string compute_key(int y, int x) {
    // Compute {y/128} _ {x/128}
    return std::to_string(y / TILE_SIZE) + "_" + std::to_string(x / TILE_SIZE);
}


void print_coordinate_map(const std::map<std::string, std::set<std::tuple<int, int, int>>>& coordinate_map) {
    for (const auto& entry : coordinate_map) {
        std::cout << "Key: " << entry.first << " -> Coordinates: ";
        
        for (const auto& coord : entry.second) {
            std::cout << "(" 
                      << std::get<0>(coord) << ", " 
                      << std::get<1>(coord) << ", " 
                      << std::get<2>(coord) << ") ";
        }
        
        std::cout << std::endl;
    }
}


s_spf_structure_t* tile_the_matrix(s_spf_structure_t* spf_matrix)
{
    int DEBUG = 1;
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
    int norigins = spf_matrix->origins_list.norigins;


    int hashMap[nbase_shapes];
    // Note: The function assume that the shape id is from 0 to nbase_shapes - 1
    for(int i = 0; i < nbase_shapes; i++){
        void* shape1 = spf_matrix->shape_dictionnary.shapes[i];
        s_codelet_1x1d_t* as_1d = (s_codelet_1x1d_t*)shape1;
        short dim_p = as_1d->dim_p;
        int num_shape_occurrences = spf_matrix->shape_dictionnary.shapes_nb_origins[i];
        shape_to_reg_2x1(shape1, shape_reg);
        hashMap[i] = (shape_reg.end_vertex[0] + 1);
    }
    // Define the map: Key -> Set of coordinate pairs
    std::map<std::string, std::set<std::tuple<int, int, int>>> coordinate_map;
  
    for(int i = 0; i < norigins; i++){
      s_origin_2d_t origin_reg = ((s_origin_2d_t*)spf_matrix->origins_list.origins)[i];
    //   printf("origin_reg.shape_id %d\n", (int)origin_reg.shape_id);
      std::string result = compute_key(origin_reg.coordinates[0], origin_reg.coordinates[1]);
      coordinate_map[result].insert({(int)origin_reg.shape_id, (int)origin_reg.coordinates[0], (int)origin_reg.coordinates[1]});
    }

    // print_coordinate_map(coordinate_map);

    int origin_index = 0;
    int global_data_ptr_offset = 0;
    for (const auto& [key, values] : coordinate_map) {
        // printf("key: %s\n", key.c_str());
        for (const auto& coord : values) {
            int shape_id = std::get<0>(coord);        // Extract shape_id
            int y = std::get<1>(coord);        // Extract y
            int x = std::get<2>(coord); // Extract x

            // if (origin_index < 50) {
            //     std::cout << "curr->shape_id " << shape_id << std::endl;
            // }

            // Insert values into the spf_matrix structure
            ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].shape_id = static_cast<short>(shape_id);
            ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].coordinates[0] = y;
            ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].coordinates[1] = x;
            ((s_origin_2d_t*)spf_matrix->origins_list.origins)[origin_index].dataptr = global_data_ptr_offset;

            // Increment the global data pointer offset
            global_data_ptr_offset += hashMap[shape_id];

            origin_index++;
        }
    }
    return spf_matrix;
}


int main(int argc, char** argv) {
    if (argc < 4) {
        usage(argv);
    }

    int debug = 0;
    int data_is_float = 0;
    int print_matrix_summary = debug;
    int print_matrix_structure = 0;

    char* rbpath = nullptr;
    char* dst_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-float") == 0) {
            data_is_float = 1;
        } else if (std::strcmp(argv[i], "-stats") == 0) {
            print_matrix_summary = 1;
        } else if (std::strcmp(argv[i], "-structure") == 0) {
            print_matrix_structure = 1;
        } else if (std::strcmp(argv[i], "--tile") == 0) {  
            // Parse TILE_SIZE argument
            if (i + 1 < argc) {
                TILE_SIZE = std::atoi(argv[i + 1]);
                if (TILE_SIZE <= 0) {
                    std::cerr << "[ERROR] Invalid tile size: " << argv[i + 1] << ". Must be a positive integer." << std::endl;
                    return 1;
                }
                i++;
            } else {
                std::cerr << "[ERROR] Missing value for --tile argument." << std::endl;
                return 1;
            }
        } else {
            if (rbpath == nullptr) {
                rbpath = argv[i];
	    } else if (dst_path == nullptr) {
		dst_path = argv[i];
            } else {
                std::cerr << "[ERROR] Unsupported argument: " << argv[i] << std::endl;
                usage(argv);
            }
        }
    }

    if (!rbpath) {
        usage(argv);
    }

    long file_size = get_file_size(rbpath);
    if (file_size != -1) {
        std::cout << "File size: " << file_size << " bytes" << std::endl;
    }

    s_spf_structure_t* spf_mat = spf_matrix_read_from_file(rbpath);
    spf_mat = tile_the_matrix(spf_mat);  // TILE_SIZE affects tiling logic

    char* file_name = strrchr(rbpath, '/');
    *strrchr(file_name, '.') = '\0';
//    snprintf(dst_path, sizeof(dst_path), "%s/%s_tile%d.uzp", dst_folder, file_name, TILE_SIZE);
    spf_matrix_write_to_file(spf_mat, dst_path);

    free(spf_mat->shape_dictionnary.shapes_nb_origins);
    free(spf_mat->shape_dictionnary.shapes_key);
    for (int i = 0; i < spf_mat->shape_dictionnary.nbase_shapes; i++) {
        free(spf_mat->shape_dictionnary.shapes[i]);
    }
    free(spf_mat->shape_dictionnary.shapes);
    free(spf_mat->origins_list.origins);
    free(spf_mat->data);
    free(spf_mat);

    return 0;
}
