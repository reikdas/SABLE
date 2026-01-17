#ifndef GENERATED_SHAPES_H
#define GENERATED_SHAPES_H

// Optimized loop for Shape#0
#define LOOP_SHAPE_0 \
    for (int i = 0; i <= 119; i++) \
    { \
        /* Compute. */ \
        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
        idx_y += 1; \
        idx_x += 1; \
    } \


// Optimized loop for Shape#1
#define LOOP_SHAPE_1 \
    for (int i = 0; i <= 79; i++) \
    { \
        /* Compute. */ \
        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
        idx_y += 1; \
        idx_x += 1; \
    } \


// Optimized loop for Shape#2
#define LOOP_SHAPE_2 \
    for (int i = 0; i <= 71; i++) \
    { \
        /* Compute. */ \
        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
        idx_y += 1; \
        idx_x += 1; \
    } \


// Optimized loop for Shape#3
#define LOOP_SHAPE_3 \
    for (int i = 0; i <= 39; i++) \
    { \
        /* Compute. */ \
        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
        idx_y += 1; \
        idx_x += 1; \
    } \


// Optimized loop for Shape#4
#define LOOP_SHAPE_4 \
    for (int i = 0; i <= 6; i++) \
    { \
        /* Compute. */ \
        y[idx_y] += data_vector[a_data_pos++] * x[idx_x]; \
        idx_y += 1; \
        idx_x += 1; \
    } \


#define fundecl_run_shape_o2d_multitype(datatypename)                      \
static                                                                    \
inline                                                                    \
void run_shape_o2d_##datatypename(s_spf_structure_t* restrict spf_matrix, \
                                  s_origin_2d_t orig,                     \
                                  datatypename* restrict x,               \
                                  datatypename* restrict y)               \
{                                                                         \
    datatypename* const data_vector = spf_matrix->data;                   \
    long long idx_y = orig.coordinates[0];                                \
    long long idx_x = orig.coordinates[1];                                \
    int a_data_pos = orig.dataptr;                                        \
                                                                          \
    switch (orig.shape_id) {                                              \
        case 0: LOOP_SHAPE_0; break;                                \
        case 1: LOOP_SHAPE_1; break;                                \
        case 2: LOOP_SHAPE_2; break;                                \
        case 3: LOOP_SHAPE_3; break;                                \
        case 4: LOOP_SHAPE_4; break;                                \
        default: break;                                                   \
    }                                                                     \
}

#endif // GENERATED_SHAPES_H
