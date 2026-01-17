import struct
import sys

class Codelet:
    def __init__(self) -> None:
        self.dim_p = None
        self.start_vertex = None
        self.end_vertex = None
        self.stride = None
        self.lattice = None
        self.num_occurrences = 0
        self.coordinates = []
        self.dataptr = []
    
    @property
    def print(self):
        return f"dim_p={self.dim_p}, start={self.start_vertex}, end={self.end_vertex}, stride={self.stride}, lattice={self.lattice}"

class Codelet1x1d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None,), end_vertex=(None,), stride=(None,), lattice=((None,),)):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 1:
            raise ValueError(f"start_vertex must be of size 1, got size {len(start_vertex)}")
        if len(end_vertex) != 1:
            raise ValueError(f"end_vertex must be of size 1, got size {len(end_vertex)}")
        if len(stride) != 1:
            raise ValueError(f"stride must be of size 1, got size {len(stride)}")
        if len(lattice) != 1 or len(lattice[0]) != 1:
            raise ValueError(f"lattice must be 1x1, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet2x1d(Codelet):
    def __init__(self, dim_p, start_vertex, end_vertex, stride, lattice):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 1:
            raise ValueError(f"start_vertex must be of size 1, got size {len(start_vertex)}")
        if len(end_vertex) != 1:
            raise ValueError(f"end_vertex must be of size 1, got size {len(end_vertex)}")
        if len(stride) != 1:
            raise ValueError(f"stride must be of size 1, got size {len(stride)}")
        if len(lattice) != 2 or len(lattice[0]) != 1 or len(lattice[1]) != 1:
            raise ValueError(f"lattice must be 2x1, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet3x1d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None,), end_vertex=(None,), stride=(None,), lattice=((None,), (None,), (None,))):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 1:
            raise ValueError(f"start_vertex must be of size 1, got size {len(start_vertex)}")
        if len(end_vertex) != 1:
            raise ValueError(f"end_vertex must be of size 1, got size {len(end_vertex)}")
        if len(stride) != 1:
            raise ValueError(f"stride must be of size 1, got size {len(stride)}")
        if len(lattice) != 3 or len(lattice[0]) != 1 or len(lattice[1]) != 1 or len(lattice[2]) != 1:
            raise ValueError(f"lattice must be 3x1, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet1x2d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None, None), end_vertex=(None, None), stride=(None, None), lattice=((None, None),)):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 2:
            raise ValueError(f"start_vertex must be of size 2, got size {len(start_vertex)}")
        if len(end_vertex) != 2:
            raise ValueError(f"end_vertex must be of size 2, got size {len(end_vertex)}")
        if len(stride) != 2:
            raise ValueError(f"stride must be of size 2, got size {len(stride)}")
        if len(lattice) != 1 or len(lattice[0]) != 2:
            raise ValueError(f"lattice must be 1x2, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet2x2d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None, None), end_vertex=(None, None), stride=(None, None), lattice=((None, None), (None, None))):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 2:
            raise ValueError(f"start_vertex must be of size 2, got size {len(start_vertex)}")
        if len(end_vertex) != 2:
            raise ValueError(f"end_vertex must be of size 2, got size {len(end_vertex)}")
        if len(stride) != 2:
            raise ValueError(f"stride must be of size 2, got size {len(stride)}")
        if len(lattice) != 2 or len(lattice[0]) != 2 or len(lattice[1]) != 2:
            raise ValueError(f"lattice must be 2x2, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet3x2d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None, None), end_vertex=(None, None), stride=(None, None), lattice=((None, None), (None, None), (None, None))):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 2:
            raise ValueError(f"start_vertex must be of size 2, got size {len(start_vertex)}")
        if len(end_vertex) != 2:
            raise ValueError(f"end_vertex must be of size 2, got size {len(end_vertex)}")
        if len(stride) != 2:
            raise ValueError(f"stride must be of size 2, got size {len(stride)}")
        if len(lattice) != 3 or len(lattice[0]) != 2 or len(lattice[1]) != 2 or len(lattice[2]) != 2:
            raise ValueError(f"lattice must be 3x2, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet1x3d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None, None, None), end_vertex=(None, None, None), stride=(None, None, None), lattice=((None, None, None),)):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 3:
            raise ValueError(f"start_vertex must be of size 3, got size {len(start_vertex)}")
        if len(end_vertex) != 3:
            raise ValueError(f"end_vertex must be of size 3, got size {len(end_vertex)}")
        if len(stride) != 3:
            raise ValueError(f"stride must be of size 3, got size {len(stride)}")
        if len(lattice) != 1 or len(lattice[0]) != 3:
            raise ValueError(f"lattice must be 1x3, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet2x3d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None, None, None), end_vertex=(None, None, None), stride=(None, None, None), lattice=((None, None, None), (None, None, None))):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 3:
            raise ValueError(f"start_vertex must be of size 3, got size {len(start_vertex)}")
        if len(end_vertex) != 3:
            raise ValueError(f"end_vertex must be of size 3, got size {len(end_vertex)}")
        if len(stride) != 3:
            raise ValueError(f"stride must be of size 3, got size {len(stride)}")
        if len(lattice) != 2 or len(lattice[0]) != 3 or len(lattice[1]) != 3:
            raise ValueError(f"lattice must be 2x3, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class Codelet3x3d(Codelet):
    def __init__(self, dim_p=0, start_vertex=(None, None, None), end_vertex=(None, None, None), stride=(None, None, None), lattice=((None, None, None), (None, None, None), (None, None, None))):
        super().__init__()
        # Validation checks
        if len(start_vertex) != 3:
            raise ValueError(f"start_vertex must be of size 3, got size {len(start_vertex)}")
        if len(end_vertex) != 3:
            raise ValueError(f"end_vertex must be of size 3, got size {len(end_vertex)}")
        if len(stride) != 3:
            raise ValueError(f"stride must be of size 3, got size {len(stride)}")
        if len(lattice) != 3 or len(lattice[0]) != 3 or len(lattice[1]) != 3 or len(lattice[2]) != 3:
            raise ValueError(f"lattice must be 3x3, got size {len(lattice)}x{len(lattice[0])}")

        self.dim_p = dim_p
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.stride = stride
        self.lattice = lattice

class SMatrixDesc:
    def __init__(self, nnz, inc_nnz, intnrow, ncols, dim_o, nbase_shapes, nhierch_shapes, data_start, dim_p):
        self.nnz = nnz
        self.inc_nnz = inc_nnz
        self.intnrow = intnrow
        self.ncols = ncols
        self.dim_o = dim_o
        self.nbase_shapes = nbase_shapes
        self.nhierch_shapes = nhierch_shapes
        self.data_start = data_start
        self.dim_p = dim_p

class SShapeDictionary:
    def __init__(self):
        self.nbase_shapes = 0
        self.nshapes_dim_p = []
        self.shapes = []
        self.shapes_nb_origins = []
        self.shapes_key = []

class SSpfStructure:
    def __init__(self):
        self.matrix_description = None
        self.nshapes_dim_p = None
        self.shape_dictionnary = SShapeDictionary()
        self.data_is_float = 0

def spf_matrix_read_from_file(spfpath):
    debug = False

    # Open file
    if debug:
        print(f"Reading file: {spfpath}")
    
    try:
        with open(spfpath, 'rb') as f:
            spf_matrix = SSpfStructure()

            # Read matrix description (equivalent to `s_matrix_desc_t` in C) iiiihiiih
            desc_size = struct.calcsize('<iiiihiiih')  # Format matches `s_matrix_desc_t`
            matrix_desc_data = f.read(desc_size)
            (nnz, inc_nnz, intnrow, ncols, dim_o, nbase_shapes, nhierch_shapes, data_start, dim_p) = struct.unpack('<iiiihiiih', matrix_desc_data)

            # Set matrix description
            spf_matrix.matrix_description = SMatrixDesc(nnz, inc_nnz, intnrow, ncols, dim_o, nbase_shapes, nhierch_shapes, data_start, dim_p)

            if debug:
                print(f"Reading matrix:")
                print(f"\tNNZ       = {nnz}")
                print(f"\tINC       = {inc_nnz}")
                print(f"\tDims      = {intnrow} x {ncols}")
                print(f"\tDim_o     = {dim_o}")
                print(f"\tBase shapes   = {nbase_shapes}")
                print(f"\tHierch shapes = {nhierch_shapes}")
                print(f"\tData ptr  = {data_start}")
                print(f"\tMax dim_p = {dim_p}")


            # Read nshapes_dim_p (array of dim_p integers)
            nshapes_dim_p_size = dim_p * struct.calcsize('<i')  # 'i' is a 4-byte int
            nshapes_dim_p_data = f.read(nshapes_dim_p_size)
            nshapes_dim_p = struct.unpack(f'<{dim_p}i', nshapes_dim_p_data)  # Unpack dim_p integers

            spf_matrix.nshapes_dim_p = nshapes_dim_p

            if debug:
                print("\tShapes/dim= [", end=" ")
                for i in range(dim_p):
                    print(nshapes_dim_p[i], end=" ")
                print("]")

            # Now read the shapes
            spf_matrix.shape_dictionnary.nbase_shapes = nbase_shapes
            spf_matrix.shape_dictionnary.shapes = [None] * nbase_shapes
            spf_matrix.shape_dictionnary.shapes_nb_origins = [0] * nbase_shapes
            spf_matrix.shape_dictionnary.shapes_key = [0] * nbase_shapes
            spf_matrix.shape_dictionnary.nshapes_dim_p = [0] * nbase_shapes

            if debug:
                print("\nReading basic shapes:")

            for i in range(nbase_shapes):
                # Read the shape id, encoding, and dim_p
                shape_id, shape_encoding, dim_p_local = struct.unpack('<hhh', f.read(6))

                if debug:
                    print(f"\tShape #{shape_id}")
                    print(f"\t\tEncoding type: {shape_encoding}")
                    print(f"\t\tdim(p)       : {dim_p_local}")

                # Check if encoding is not supported
                if shape_encoding != 0:
                    print("Other formats for shapes not supported yet!")
                    return None

                # Determine the shape size based on dim_o and dim_p_local
                shape_bytes_size = 0
                if dim_o == 1:
                    shape_bytes_size = {1: 16, 2: 32, 3: 48}.get(dim_p_local, None)
                elif dim_o == 2:
                    shape_bytes_size = {1: 20, 2: 40, 3: 60}.get(dim_p_local, None)
                elif dim_o == 3:
                    shape_bytes_size = {1: 24, 2: 48, 3: 72}.get(dim_p_local, None)

                if shape_bytes_size is None:
                    print("Unsupported shape size or dimensions!")
                    return None

                # Read shape data and store it in shape_dictionnary
                if dim_o == 1:
                    if dim_p_local == 1:
                        data = struct.unpack('<iiii', f.read(shape_bytes_size))
                        shape = Codelet1x1d(dim_p=dim_p_local, 
                                            start_vertex=(data[0],), 
                                            end_vertex=(data[1],), 
                                            stride=(data[2],), 
                                            lattice=((data[3],),))
                    elif dim_p_local == 2:
                        data = struct.unpack('<iiiiiiii', f.read(shape_bytes_size))
                        shape = Codelet1x2d(dim_p=dim_p_local, 
                                            start_vertex=(data[0], data[1]), 
                                            end_vertex=(data[2], data[3]), 
                                            stride=(data[4], data[5]), 
                                            lattice=((data[6], data[7]),))
                    elif dim_p_local == 3:
                        data = struct.unpack('<iiiiiiiiiiii', f.read(shape_bytes_size))
                        shape = Codelet1x3d(dim_p=dim_p_local, 
                                            start_vertex=(data[0], data[1], data[2]), 
                                            end_vertex=(data[3], data[4], data[5]), 
                                            stride=(data[6], data[7], data[8]), 
                                            lattice=((data[9], data[10], data[11]),))
                elif dim_o == 2:
                    if dim_p_local == 1:
                        data = struct.unpack('<iiiii', f.read(shape_bytes_size))
                        shape = Codelet2x1d(dim_p=dim_p_local, 
                                            start_vertex=[data[0]], 
                                            end_vertex=[data[1]], 
                                            stride=[data[2]], 
                                            lattice=[[data[3]], [data[4]]])
                    elif dim_p_local == 2:
                        data = struct.unpack('<iiiiiiiiii', f.read(shape_bytes_size))
                        shape = Codelet2x2d(dim_p=dim_p_local, 
                                            start_vertex=(data[0], data[1]), 
                                            end_vertex=(data[2], data[3]), 
                                            stride=(data[4], data[5]), 
                                            lattice=((data[6], data[7]), (data[8], data[9])))
                    elif dim_p_local == 3:
                        data = struct.unpack('<iiiiiiiiiiiiiii', f.read(shape_bytes_size))
                        shape = Codelet2x3d(dim_p=dim_p_local, 
                                            start_vertex=(data[0], data[1], data[2]), 
                                            end_vertex=(data[3], data[4], data[5]), 
                                            stride=(data[6], data[7], data[8]), 
                                            lattice=((data[9], data[10], data[11]), (data[12], data[13], data[14])))
                elif dim_o == 3:
                    if dim_p_local == 1:
                        data = struct.unpack('<iiiiii', f.read(shape_bytes_size))
                        shape = Codelet3x1d(dim_p=dim_p_local, 
                                            start_vertex=(data[0],), 
                                            end_vertex=(data[1],), 
                                            stride=(data[2],), 
                                            lattice=((data[3],), (data[4],), (data[5],)))
                    elif dim_p_local == 2:
                        data = struct.unpack('<iiiiiiiiiiii', f.read(shape_bytes_size))
                        shape = Codelet3x2d(dim_p=dim_p_local, 
                                            start_vertex=(data[0], data[1]), 
                                            end_vertex=(data[2], data[3]), 
                                            stride=(data[4], data[5]), 
                                            lattice=((data[6], data[7]), (data[8], data[9]), (data[10], data[11])))
                    elif dim_p_local == 3:
                        data = struct.unpack('<iiiiiiiiiiiiiiiiii', f.read(shape_bytes_size))
                        shape = Codelet3x3d(dim_p=dim_p_local, 
                                            start_vertex=(data[0], data[1], data[2]), 
                                            end_vertex=(data[3], data[4], data[5]), 
                                            stride=(data[6], data[7], data[8]), 
                                            lattice=((data[9], data[10], data[11]), (data[12], data[13], data[14]), (data[15], data[16], data[17])))
               
                # Store the shape pointer (here as list of bytes, but can be parsed as needed)
                spf_matrix.shape_dictionnary.shapes[i] = shape
                spf_matrix.shape_dictionnary.nshapes_dim_p[i] = dim_p_local
                spf_matrix.shape_dictionnary.shapes_nb_origins[i] = 0
                spf_matrix.shape_dictionnary.shapes_key[i] = 0

                if debug:
                    print(f"\t\tShape data (bytes): {shape.print}")

            # Read the origins
            norigins = struct.unpack('<i', f.read(4))[0]
            origin_byte_size = 0
            if dim_o == 1:
                origin_byte_size = 10
            elif dim_o == 2:
                origin_byte_size = 14
            elif dim_o == 3:
                origin_byte_size = 18

            for i in range(norigins):
                # Read the origin data
                shape_id, coordinates_y, coordinates_x, dataptr = struct.unpack('<hiii', f.read(origin_byte_size))
                
                spf_matrix.shape_dictionnary.shapes[shape_id].num_occurrences += 1
                spf_matrix.shape_dictionnary.shapes_nb_origins[shape_id] += 1
                spf_matrix.shape_dictionnary.shapes[shape_id].coordinates.append((coordinates_y, coordinates_x))
                spf_matrix.shape_dictionnary.shapes[shape_id].dataptr.append(dataptr)
                # if debug:
                #     print(f"\nReading origin #{ shape_id}")
                #     print(f"\tCoordinates: {coordinates_y, coordinates_x}")
                #     print(f"\tData ptr   : {dataptr}")

            if debug:
                print(f"Origins: {norigins}")

            return spf_matrix

    except FileNotFoundError:
        print(f"File not found: {spfpath}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# ------------------------------------------------------------------------------------------------------------------------
# Function to generate the optimized for loops as #define macros
def generate_optimized_header(spf_matrix, output_file):
    try:
        with open(output_file, 'w') as f:
            f.write("#ifndef GENERATED_SHAPES_H\n")
            f.write("#define GENERATED_SHAPES_H\n\n")

            shapes = spf_matrix.shape_dictionnary.shapes
            for idx, shape in enumerate(shapes):
                start = shape.start_vertex[0]
                end = shape.end_vertex[0]
                stride = shape.stride[0]
                lattice_y = shape.lattice[0][0]
                lattice_x = shape.lattice[1][0]

                offset_idx_y = lattice_y * stride
                offset_idx_x = lattice_x * stride

                # Generate the #define macro for this shape
                f.write(f"// Optimized loop for Shape#{idx}\n")
                f.write(f"#define LOOP_SHAPE_{idx} \\\n")
                # Generate the loop
                if stride == 1:
                    f.write(f"    for (int i = {start}; i <= {end}; i++) \\\n")
                    f.write(f"    {{ \\\n")
                    f.write(f"        /* Compute. */ \\\n")
                    f.write(f"        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \\\n")
                    f.write(f"        idx_y += {offset_idx_y}; \\\n")
                    f.write(f"        idx_x += {offset_idx_x}; \\\n")
                    f.write(f"    }} \\\n")
                else:
                    f.write(f"    for (int i = {start}; i <= {end}; i += {stride}) \\\n")
                    f.write(f"    {{ \\\n")
                    f.write(f"        /* Compute. */ \\\n")
                    f.write(f"        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \\\n")
                    f.write(f"        idx_y += {offset_idx_y}; \\\n")
                    f.write(f"        idx_x += {offset_idx_x}; \\\n")
                    f.write(f"    }} \\\n")
                f.write("\n\n")

            # Write the macro to call all LOOP_SHAPE_X macros
            f.write("#define fundecl_run_shape_o2d_multitype(datatypename)                      \\\n")
            f.write("static                                                                    \\\n")
            f.write("inline                                                                    \\\n")
            f.write("void run_shape_o2d_##datatypename(s_spf_structure_t* restrict spf_matrix, \\\n")
            f.write("                                  s_origin_2d_t orig,                     \\\n")
            f.write("                                  datatypename* restrict x,               \\\n")
            f.write("                                  datatypename* restrict y)               \\\n")
            f.write("{                                                                         \\\n")
            f.write("    datatypename* const data_vector = spf_matrix->data;                   \\\n")
            f.write("    long long idx_y = orig.coordinates[0];                                \\\n")
            f.write("    long long idx_x = orig.coordinates[1];                                \\\n")
            f.write("    int a_data_pos = orig.dataptr;                                        \\\n")
            f.write("                                                                          \\\n")
            f.write("    switch (orig.shape_id) {                                              \\\n")
            
            # Generate cases for each shape
            for i in range(len(shapes)):  # Assuming shapes is a list or range of shapes
                f.write(f"        case {i}: LOOP_SHAPE_{i}; break;                                \\\n")

            # Close the switch statement and the function
            f.write("        default: break;                                                   \\\n")
            f.write("    }                                                                     \\\n")
            f.write("}\n\n")

            f.write("#endif // GENERATED_SHAPES_H\n")
        return True
    except Exception as e:
        print(f"An error occurred while generating the macro for loops file for spf shapes: {str(e)}")
        return False


if __name__ == "__main__":
    spfpath = sys.argv[1]
    output_header_file = sys.argv[2]

    spf_matrix = spf_matrix_read_from_file(spfpath)
    if spf_matrix is not None:
        print("Matrix read successfully!")
    else:
        print("Matrix read failed!")
        sys.exit(1)

    # Generate the .h file with optimized for loops
    generate_optimized_header(spf_matrix, output_header_file)

    print(f"Header file '{output_header_file}' generated successfully.")
    
