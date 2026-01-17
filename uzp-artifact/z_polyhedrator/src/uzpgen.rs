use std::{collections::HashMap, fs::File, io::{Seek, SeekFrom, Write}, path::PathBuf, time::Instant};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use colored::Colorize;
use itertools::Itertools;
use linked_hash_map::LinkedHashMap;
use linked_hash_set::LinkedHashSet;
use sprs::{CsMat, TriMat};

use crate::utils::{Pattern,Piece,Uwc,OriginUwc, MetaPattern, MetaPatternPiece, convex_hull_hyperrectangle_nd, metapattern_to_hyperrectangle_uwc};

pub struct UZPGen {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
    pub inc_nnz: usize,

    // ast_list: Vec<Piece>,
    // uwc_list: Vec<(Uwc, i32)>,
    // distinct_patterns: LinkedHashMap<Pattern, i32>,
    // distinct_uwc: LinkedHashMap<Uwc, i32>,

    meta_patterns: LinkedHashMap<i32, MetaPattern>,
    meta_pattern_pieces: LinkedHashMap<MetaPatternPiece, i32>
}

impl UZPGen {
    pub fn from_piece_list(ast_list: Vec<Piece>, nrows: usize, ncols: usize, nnz: usize) -> Self {
        let ninc_nnz = ast_list.iter().filter(|(_,_,(n,_,_))| *n == 1).count();
        let inc_nnz = nnz - ninc_nnz;

        // Create distinct pattern LinkedHashMap indexed 0..used_patterns with the help of an intermediate HashSet
        let mut meta_patterns: LinkedHashMap<i32, MetaPattern> = ast_list
            .iter()
            .map(|(_,_,pattern)| *pattern)
            .filter(|(i,_,_)| *i > 1)
            .collect::<LinkedHashSet<Pattern>>()
            .into_iter()
            .enumerate()
            .map(|(idx, pattern)| (idx as i32, (pattern, 1, None)))
            .collect();

        // Finally insert (1,0,0) pattern. (Has to be the last one):
        // This can not be removed, as we always take into account the single nonzero values,
        // even in the case of no single points remaining.
        meta_patterns.insert(-1i32, ((1,0,0), 1, None));

        let meta_pattern_pieces: LinkedHashMap<MetaPatternPiece, i32>;

        { // scoped to keep aux_meta_patterns in memory the least time possible
            let aux_meta_patterns: HashMap<Pattern, i32> = meta_patterns
                .iter()
                .map(|(key, mp)| (mp.0, *key))
                .collect();

            meta_pattern_pieces = ast_list
                .iter()
                .map(|(x,y,pattern)| ((*x,*y), *aux_meta_patterns.get(pattern).unwrap()))
                .collect();
        }

        return UZPGen {
            nrows,
            ncols,
            nnz,
            inc_nnz,
            meta_patterns,
            meta_pattern_pieces
        };
    }

    pub fn from_metapatterns_list(meta_patterns: LinkedHashMap<i32, MetaPattern>, meta_pattern_pieces: LinkedHashMap<MetaPatternPiece, i32>, nrows: usize, ncols: usize, nnz: usize, inc_nnz: usize) -> Self {
        // Reorder id = -1
        let ninc_nnz_lhm: LinkedHashMap<MetaPatternPiece, i32> = meta_pattern_pieces.iter().filter(|(_,id)| **id == -1).map(|(rowcol, id)| (*rowcol, *id)).collect();

        let mut meta_pattern_pieces: LinkedHashMap<MetaPatternPiece, i32> = meta_pattern_pieces.into_iter().filter(|(_,id)| *id != -1).collect();
        meta_pattern_pieces.extend(ninc_nnz_lhm);

        // Not necessary
        // let mut mp = meta_patterns;
        // mp.get_refresh(&-1);

        return UZPGen {
            nrows,
            ncols,
            nnz,
            inc_nnz,
            meta_patterns,
            meta_pattern_pieces
        }
    }

    // #[allow(dead_code)] // UNUSED AND REDUNDANT. TBDeleted
    // pub fn print_ast_list(&self) {
    //     println!("AST_List:\nRow\tCol\tN\tI\tJ");
    //     self.meta_pattern_pieces.iter().for_each(|((row, col), id)| {
    //         println!("{}\t{}\t{}", row, col, id);
    //     });
    // }

    pub fn print_uwc_list(&self, show_eqs: bool) {
        println!("Uwc List:\nid\tU\t\tw\tc");
        self.meta_pattern_pieces.iter().for_each(|(_, id)| {
            let (u,w,c) = metapattern_to_hyperrectangle_uwc(*id, &self.meta_patterns);
            print!("{:?}\t{:?}\t{:?}\t{:?}{}", id, u, w, c, { if show_eqs { format_eqs(&u, &w) } else { "\n".to_string() } });
        });
    }

    pub fn print_distinct_uwc_list(&self, show_eqs: bool) {
        println!("Distinct Uwc List:\nid\tU\t\tw\tc");
        self.meta_patterns.iter().for_each(|(id, _)| {
            let (u,w,c) = metapattern_to_hyperrectangle_uwc(*id, &self.meta_patterns);
            print!("{:?}\t{:?}\t{:?}\t{:?}{}", id, u, w, c, { if show_eqs { format_eqs(&u, &w) } else { "\n".to_string() } });


            // DEBUG
            // let ch = convex_hull_hyperrectangle_nd(&u, &w, false);
            // println!("DEBUG -- Convex Hull: {:?}", ch);
            // let ch = convex_hull_hyperrectangle_nd(&u, &w, true);
            // println!("DEBUG -- Dense \"Convex Hull\": {:?}", ch);
        });
    }

    #[allow(dead_code)]
    pub fn get_uwc_list(&self) -> Vec<(Uwc, i32)> {
        self.meta_pattern_pieces
            .iter()
            .map(|(_, id)| {
                let uwc = metapattern_to_hyperrectangle_uwc(*id, &self.meta_patterns);
                (uwc, *id)
            })
            .collect::<Vec<(Uwc, i32)>>()
    }

    pub fn get_orig_uwc_list(&self) -> Vec<(OriginUwc, i32)> {
        self.meta_pattern_pieces
            .iter()
            .map(|((row,col), id)| {
                let uwc = metapattern_to_hyperrectangle_uwc(*id, &self.meta_patterns);
                ((*row,*col,uwc), *id)
            })
            .collect::<Vec<(OriginUwc, i32)>>()
    }

    pub fn write_uzp(&self, input_value_matrix: &str, output_file_path: &str, transpose_input: bool, transpose_output: bool, uninc_as_patterns: bool) {
        // Read matrixmarket f64 value matrix
        let f64_value_matrix: CsMat<f64> = crate::utils::read_matrix_market_csr(input_value_matrix, transpose_input);

        // Quick sanity check
        if f64_value_matrix.nnz() != self.nnz {
            panic!("NNZ of value matrix and pattern list do not match. Maybe double check your params?");
        }

        let mut file = File::create(output_file_path).expect(format!("Unable to create file {}", output_file_path).as_str());

        let path = PathBuf::from(output_file_path);
        eprintln!("Writing to file {}", path.to_str().unwrap().bright_blue());

        // Write header
        file.write_i32::<LittleEndian>(self.nnz as i32).unwrap();
        file.write_i32::<LittleEndian>( if uninc_as_patterns { self.nnz as i32 } else { self.inc_nnz as i32 } ).unwrap();
        if !transpose_output {
            // Write matrix in a normal way
            file.write_i32::<LittleEndian>(self.nrows as i32).unwrap();
            file.write_i32::<LittleEndian>(self.ncols as i32).unwrap();
        } else {
            // Write it transposed
            file.write_i32::<LittleEndian>(self.ncols as i32).unwrap();
            file.write_i32::<LittleEndian>(self.nrows as i32).unwrap();
        }

        // Get index of single nonzeros (not in a pattern to filter them out of the next foreach)
        // let ninc_nonzero_pattern_id = self.distinct_patterns.get(&(1,0,0)).unwrap();
        // If we want to dump everything as patterns, we can just filter the single nonzeros with a never-used id, like min i32
        let ninc_nonzero_pattern_id = if uninc_as_patterns { std::i32::MIN } else { -1i32 };

        // This has been brought from down below, as it is useful for the following computation
        // We also know that regular pieces are at the end of the list
        let piece_cutoff = self.meta_pattern_pieces.iter().filter(|(_, id)| **id != ninc_nonzero_pattern_id).count();
        // println!("Piece cutoff = {}", piece_cutoff);

        // Write dimensions
        file.write_i16::<LittleEndian>(2i16).unwrap();
        // number of base shapes is actual found shapes, not unfound ones. Also we have to take into account removing the single nonzeros
        file.write_i32::<LittleEndian>((self.meta_pattern_pieces.iter().filter(|(_, id)| **id != ninc_nonzero_pattern_id).unique_by(|(_, id)| **id).count()) as i32).unwrap();
        // write zero hierarchical shapes
        file.write_i32::<LittleEndian>(0i32).unwrap();
        // write TEMPORARY ZERO as pointer to start of data. Will need to fseek to position 26 later
        //  (python code `f.seek ( 26 )` on write_uzp func at around line 810)
        file.write_i32::<LittleEndian>(0i32).unwrap();

        let shape_dims_max: i16 = {
            if piece_cutoff == 0 { 0i16 }
            else {
                let (_, (_, max_order, _)) = self.meta_patterns.iter().max_by_key(|(_,(_,order,_))| *order).unwrap();
                *max_order as i16
            }
        };

        // Write maximum dimensionality of iP for vertex_rec
        file.write_i16::<LittleEndian>(shape_dims_max).unwrap();

        for _ in 0..shape_dims_max {
            file.write_i32::<LittleEndian>(0i32).unwrap();
        }

        // Create REORDER dictionary
        let reorder: LinkedHashMap<i32, usize> = self.meta_pattern_pieces
            .iter()
            .filter(|(_, id)| **id != ninc_nonzero_pattern_id)
            .unique_by(|(_, id)| **id)
            .enumerate()
            .map(|(idx, (_, id))| (*id, idx))
            .collect();

        // DEBUG -- eprintln!("REORDER: {:?}", reorder);

        self.meta_pattern_pieces
            .iter()
            .filter(|(_, id)| **id != ninc_nonzero_pattern_id)
            .unique_by(|(_, id)| **id)
            .for_each(|(_, id)| {

            let (u,w,c) = metapattern_to_hyperrectangle_uwc(*id, &self.meta_patterns);

            // Write shape id
            file.write_i16::<LittleEndian>( *reorder.get(id).unwrap() as i16 ).unwrap();
            // Write type of encoding. 0 = vertex_rec, 1 = vertex_gen, 2 = ineqs . FIXME only writes vertex_rec
            file.write_i16::<LittleEndian>(0i16).unwrap();
            // Write dimension of i_p. Hardcodec for vertex_rec
            // INFO This can also be done by accessing self.metapatterns and checking ORDER field
            file.write_i16::<LittleEndian>(u[0].len() as i16).unwrap();
            // println!("    - Dimension of i_p = {}", u[0].len());

            // Get convex_hull FIXED for n-dimensional hyperrectangles
            let ch: Vec<Vec<i32>> = convex_hull_hyperrectangle_nd(&u, &w, false);

            // Write minimal point
            for i in 0..ch[0].len() {
                file.write_i32::<LittleEndian>(ch[0][i]).unwrap();
                // println!("    - Minimal point from ch[0] = {}", ch[0][i]);
            }

            // Write lenghts along axis
            for i in 0..ch[0].len() {
                // taking shortcut as minimal points are always [0,0,...,0] (N times)
                file.write_i32::<LittleEndian>(ch[ch.len()-1][i]).unwrap();
                // println!("    - Lenghts along axis from ch[ch.len()-1][i] {:?}", ch[ch.len()-1][i]);
            }

            // "Hardcoded stride at this time"
            for _ in 0..u[0].len() {
                file.write_i32::<LittleEndian>(1i32).unwrap();
            }

            // Write lattice
            if !transpose_output {
                // println!("c = {:?}", c);
                for cc in c {
                    file.write_i32::<LittleEndian>(cc).unwrap();
                }
            } else {
                // BUGFIX FOR TRANSPOSING. If the array is [0,1,2,3,4,5], it must write [1,0,3,2,5,4]
                for (a,b) in c.iter().tuples() {
                    file.write_i32::<LittleEndian>(*b).unwrap();
                    file.write_i32::<LittleEndian>(*a).unwrap();
                }
            }
        });

        // Write total number of origins
        file.write_i32::<LittleEndian>(piece_cutoff as i32).unwrap();

        let mut data_offset: i32 = 0;
        let mut mpp_iter = self.meta_pattern_pieces.iter();
        for _ in 0..piece_cutoff {
            let ((row,col),id) = mpp_iter.next().unwrap();

            let (u,w,_) = metapattern_to_hyperrectangle_uwc(*id, &self.meta_patterns);

            // Write shape id
            file.write_i16::<LittleEndian>( *reorder.get(id).unwrap() as i16 ).unwrap();

            // Get convex_hull
            let ch: Vec<Vec<i32>> = convex_hull_hyperrectangle_nd(&u, &w, true);

            // Write coordinates of AST's starting point
            if !transpose_output {
                file.write_i32::<LittleEndian>(*row as i32).unwrap(); // row
                file.write_i32::<LittleEndian>(*col as i32).unwrap(); // col
            } else {
                file.write_i32::<LittleEndian>(*col as i32).unwrap(); // col
                file.write_i32::<LittleEndian>(*row as i32).unwrap(); // row
            }
            file.write_i32::<LittleEndian>(data_offset).unwrap();                 // data offset
            data_offset += ch.len() as i32;   // Offset in elements. no judgment about data type
        }

        // Codes here:
        //  -> CSR = 0
        //  -> ??? = 1
        //  -> COO = 2
        let csr_size = self.nrows + 1 + (self.nnz - self.inc_nnz);
        let coo_size = 2 * (self.nnz - self.inc_nnz);

        // If we are dumping uninc as patterns, we have to use COO format for zero space wasted
        let uninc_format: u8 = if uninc_as_patterns { 2u8 }
                               else {
                                   if csr_size <= coo_size { 0u8 }
                                   else { 2u8 }
                               };

        eprintln!("Writing uninc_format = {} to offset 0x{:X}...\n", uninc_format, file.seek(SeekFrom::Current(0)).unwrap());
        file.write_u8(uninc_format).unwrap();

        // Set iterator
        let mut mpp_iter = self.meta_pattern_pieces.iter().skip(piece_cutoff);

        match uninc_format {
            0 => {  // DEBUG -- eprintln!("Writing CSR");
                    let mut local_coo_mat: TriMat<u8> = TriMat::new((self.nrows, self.ncols));
                    // DEBUG -- eprint!("Writing points: [");

                    for _ in piece_cutoff..self.meta_pattern_pieces.len() {
                        let (row, col) = mpp_iter.next().unwrap().0;
                        // DEBUG -- eprint!("({},{}) ", *row, *col);
                        local_coo_mat.add_triplet(*row, *col, 1u8);
                    }

                    let local_csr_mat: CsMat<u8>;
                    if !transpose_output {
                        local_csr_mat = local_coo_mat.to_csr();
                    } else {
                        local_csr_mat = local_coo_mat.transpose_view().to_csr();
                    }

                    // DEBUG -- eprintln!("\x08] with:\nind_ptr: {:?}", local_csr_mat.proper_indptr());
                    // DEBUG -- eprintln!("indices: {:?}", local_csr_mat.indices());

                    // Write rowptr/indptr
                    local_csr_mat.proper_indptr().iter().for_each(|iptr_val| {
                        file.write_i32::<LittleEndian>(*iptr_val as i32).unwrap();
                    });
                    // Write colptr/indices
                    local_csr_mat.indices().iter().for_each(|ind_val| {
                        file.write_i32::<LittleEndian>(*ind_val as i32).unwrap();
                    });
                 },
            2 => {  // DEBUG -- eprintln!("Writing COO");

                    let mut rowvec: Vec<i32> = vec![];
                    let mut colvec: Vec<i32> = vec![];

                    for _ in piece_cutoff..self.meta_pattern_pieces.len() {
                        let (row, col) = mpp_iter.next().unwrap().0;
                        rowvec.push(*row as i32);
                        colvec.push(*col as i32);
                    }

                    if transpose_output {
                        std::mem::swap(&mut rowvec, &mut colvec);
                    }

                    // DEBUG -- eprint!("Writing Rowptr: ");
                    for row in rowvec {
                        // DEBUG -- eprint!("{} ", row);
                        file.write_i32::<LittleEndian>(row).unwrap(); // Write rowptr
                    }
                    // DEBUG -- eprint!("\nWriting Colptr: ");
                    for col in colvec {
                        // DEBUG -- eprint!("{} ", col);
                        file.write_i32::<LittleEndian>(col).unwrap(); // Write colptr
                    }
                    // DEBUG -- eprintln!();
                 },
            _ => { panic!("{} uninc_format internal variable was {} and was set incorrectly. Fix this immediately and/or report this issue.", "[ERROR]".bold().red(), uninc_format) }
        }

        // Save current position for later
        let curr_pos = file.seek(SeekFrom::Current(0)).unwrap();

        // And rewrite pointer to start of data
        file.seek(SeekFrom::Start(26)).unwrap();
        file.write_i32::<LittleEndian>(curr_pos as i32).unwrap();
        file.seek(SeekFrom::Start(curr_pos)).unwrap();


        // f.write( struct.pack( len(self.mask)*"d", *mat.data[self.reorder] ) )
        self.meta_pattern_pieces.iter().for_each(|((row,col),id)| {
            for val in recursive_traverse(&(*row,*col), *id, &self.meta_patterns, &f64_value_matrix){
                file.write_f64::<LittleEndian>(val).unwrap();
            }
        });
    }
}

pub fn convert_uzp (input_uzp_file_path: &str, output_mtx_file_path: &str, csr: bool, print_ast_list: bool) {
    let mut file = File::open(input_uzp_file_path).expect(format!("Unable to open uzp file {}", input_uzp_file_path).as_str());

    // Read header
    let nnz = file.read_i32::<LittleEndian>().unwrap();
    let inc_nnz = file.read_i32::<LittleEndian>().unwrap();
    let nrows = file.read_i32::<LittleEndian>().unwrap();
    let ncols = file.read_i32::<LittleEndian>().unwrap();

    let dims = file.read_i16::<LittleEndian>().unwrap();
    if dims != 2 {
        panic!("Only 2D matrices are supported at the moment");
    }

    let num_shapes = file.read_i32::<LittleEndian>().unwrap();
    // skip num_hier_shapes
    file.seek(SeekFrom::Current(32/8)).unwrap();
    // let num_hier_shapes = file.read_i32::<LittleEndian>().unwrap();

    // Pointer to start of data
    let data_ptr = file.read_i32::<LittleEndian>().unwrap();

    let max_dims = file.read_i16::<LittleEndian>().unwrap();
    // Skip max_dims data
    file.seek(SeekFrom::Current((32/8)*(max_dims as i64))).unwrap();

    // Create sprs triplet matrix for insertion
    // let mut triplet_matrix: TriMat<f64> = TriMat::new((nrows as usize, ncols as usize));
    // initialize three vecs with capacity nnz
    let mut rowvec: Vec<usize> = Vec::with_capacity(nnz as usize);
    let mut colvec: Vec<usize> = Vec::with_capacity(nnz as usize);
    let mut datavec: Vec<f64> = Vec::with_capacity(nnz as usize);

    //                       shape_id, (dim_of_ip, lengths_along_axis, c)
    let mut shapes_map: HashMap<i16, (i16, Vec<i32>, Vec<i32>)> = HashMap::with_capacity(num_shapes as usize);

    for _ in 0..(num_shapes as usize) {
        // Read shapes
        // shapes_vec[shape_id].0 =

        let l_shape_id = file.read_i16::<LittleEndian>().unwrap();

        let type_of_encoding = file.read_i16::<LittleEndian>().unwrap();
        if type_of_encoding != 0 {
            panic!("Only vertex_rec encoding is supported at the moment");
        }

        // shapes_vec[shape_id].1

        let l_dim_of_ip = file.read_i16::<LittleEndian>().unwrap();

        // for i in 0..dim_of_ip {
        //     // let min_point = file.read_i32::<LittleEndian>().unwrap();
        //     // let len_along_axis = file.read_i32::<LittleEndian>().unwrap();
        //     // let stride = file.read_i32::<LittleEndian>().unwrap();
        // }
        // Skip min_point
        file.seek(SeekFrom::Current((32/8)*(l_dim_of_ip as i64))).unwrap();

        // Read len_along_axis
        let mut l_len_along_axis: Vec<i32> = Vec::with_capacity(l_dim_of_ip as usize);
        for _ in 0..l_dim_of_ip {
            l_len_along_axis.push(file.read_i32::<LittleEndian>().unwrap());
        }

        // Skip stride
        file.seek(SeekFrom::Current((32/8)*(l_dim_of_ip as i64))).unwrap();

        // read 2*dim_of_ip c values into shapes_vec[shape_id].2
        let mut l_c: Vec<i32> = Vec::with_capacity(2*l_dim_of_ip as usize);
        for _ in 0..2*l_dim_of_ip {
            l_c.push(file.read_i32::<LittleEndian>().unwrap());
        }

        shapes_map.insert(l_shape_id,(l_dim_of_ip, l_len_along_axis, l_c));
    }

    // Read total number of origins
    let num_origins = file.read_i32::<LittleEndian>().unwrap();
    // let mut data_offset: i32 = 0;

    if print_ast_list {
        eprintln!("{} Printing AST List:", "[INFO]".cyan().bold());
        println!("Row\tCol\tN\tI\tJ");
    }

    for _ in 0..num_origins {
        let shape_id = file.read_i16::<LittleEndian>().unwrap();
        let base_row = file.read_i32::<LittleEndian>().unwrap();
        let base_col = file.read_i32::<LittleEndian>().unwrap();
        // skip reading data_offset
        file.seek(SeekFrom::Current(32/8)).unwrap();

        // NO NEED TO JUMP TO DATA OFFSET, AS WE READ IT ALL TOGETHER AT THE END. WE JUST POPULATE ROW AND COL VECTORS
        // Read data offset as the num of elements. We can calculate this, but this is easier
        // data_offset = file.read_i32::<LittleEndian>().unwrap() - data_offset;
        // let curr_pos = file.seek(SeekFrom::Current(0)).unwrap();
        // file.seek(SeekFrom::Start(data_ptr as u64 + data_offset as u64)).unwrap();

        // traverse row and col from l_row and l_col, and push index into rowvec and colvec
        let (l_dim_of_ip, l_len_along_axis, l_c) = shapes_map.get(&shape_id).unwrap();

        recursive_populate_row_col_vec(*l_dim_of_ip, l_len_along_axis, l_c, &mut rowvec, &mut colvec, base_row, base_col);

        if print_ast_list {
            println!("{}\t{}\t{}\t{}\t{}", base_row, base_col, l_len_along_axis[0]+1, l_c[0], l_c[1]);
        }

        // Return to previous position
        // file.seek(SeekFrom::Start(curr_pos)).unwrap();
    }

    // Read uninc_format
    let uninc_format = file.read_u8().unwrap();
    match uninc_format {
        0 => {  //eprintln!("Reading CSR");
                let mut last_row_cnt = file.read_i32::<LittleEndian>().unwrap();

                // Read rowptr
                for curr_row in 0..nrows {
                    let row_cnt = file.read_i32::<LittleEndian>().unwrap();
                    // Insert row_cnt - last_row_cnt, curr_row values into rowvec
                    rowvec.extend(vec![curr_row as usize; (row_cnt - last_row_cnt) as usize]);
                    last_row_cnt = row_cnt;
                }
                // Read colidx
                for _ in 0..nnz-inc_nnz {
                    colvec.push(file.read_i32::<LittleEndian>().unwrap() as usize);
                }
             },
        2 => {  // eprintln!("Reading COO");
                // Read rowptr
                for _ in 0..nnz-inc_nnz {
                    rowvec.push(file.read_i32::<LittleEndian>().unwrap() as usize);
                }
                // Read colptr
                for _ in 0..nnz-inc_nnz {
                    colvec.push(file.read_i32::<LittleEndian>().unwrap() as usize);
                }
             },
        _ => { panic!("{} Read uninc_format variable value was {} and is unsupported at the moment. Report this issue.", "[ERROR]".bold().red(), uninc_format) }
    }

    // seek to data_ptr
    file.seek(SeekFrom::Start(data_ptr as u64)).unwrap();

    // Read data
    for _ in 0..nnz {
        datavec.push(file.read_f64::<LittleEndian>().unwrap());
    }

    // println!("DEBUG -- rowvec = {:?}", rowvec);
    // println!("DEBUG -- colvec = {:?}", colvec);
    // println!("DEBUG -- datavec = {:?}", datavec);
    // println!("DEBUG -- lengths (row,col,data) = {:?}, nnz = {:?}", (rowvec.len(), colvec.len(), datavec.len()), nnz);

    let coo_mat = TriMat::from_triplets((nrows as usize,ncols as usize), rowvec, colvec, datavec);

    let csx_matrix: CsMat<f64>;
    if csr {
        csx_matrix = coo_mat.to_csr();
    } else {
        csx_matrix = coo_mat.to_csc();
    }

    // Write matrix to file
    sprs::io::write_matrix_market(output_mtx_file_path, &csx_matrix).unwrap();
}

pub fn convert_uzp_for_timing (input_uzp_file_path: &str, output_mtx_file_path: &str, csr: bool) {
    let mut file = File::open(input_uzp_file_path).expect(format!("Unable to open uzp file {}", input_uzp_file_path).as_str());

    // Read header
    let nnz = file.read_i32::<LittleEndian>().unwrap();
    let inc_nnz = file.read_i32::<LittleEndian>().unwrap();
    let nrows = file.read_i32::<LittleEndian>().unwrap();
    let ncols = file.read_i32::<LittleEndian>().unwrap();

    let dims = file.read_i16::<LittleEndian>().unwrap();
    if dims != 2 {
        panic!("Only 2D matrices are supported at the moment");
    }

    let num_shapes = file.read_i32::<LittleEndian>().unwrap();
    // skip num_hier_shapes
    file.seek(SeekFrom::Current(32/8)).unwrap();
    // let num_hier_shapes = file.read_i32::<LittleEndian>().unwrap();

    // Pointer to start of data
    let data_ptr = file.read_i32::<LittleEndian>().unwrap();

    let max_dims = file.read_i16::<LittleEndian>().unwrap();
    // Skip max_dims data
    file.seek(SeekFrom::Current((32/8)*(max_dims as i64))).unwrap();

    // Create sprs triplet matrix for insertion
    // let mut triplet_matrix: TriMat<f64> = TriMat::new((nrows as usize, ncols as usize));
    // initialize three vecs with capacity nnz
    let mut rowvec: Vec<usize> = Vec::with_capacity(nnz as usize);
    let mut colvec: Vec<usize> = Vec::with_capacity(nnz as usize);
    let mut datavec: Vec<f64> = Vec::with_capacity(nnz as usize);

    //                       shape_id, (dim_of_ip, lengths_along_axis, c)
    let mut shapes_map: HashMap<i16, (i16, Vec<i32>, Vec<i32>)> = HashMap::with_capacity(num_shapes as usize);

    for _ in 0..(num_shapes as usize) {
        // Read shapes
        // shapes_vec[shape_id].0 =

        let l_shape_id = file.read_i16::<LittleEndian>().unwrap();

        let type_of_encoding = file.read_i16::<LittleEndian>().unwrap();
        if type_of_encoding != 0 {
            panic!("Only vertex_rec encoding is supported at the moment");
        }

        // shapes_vec[shape_id].1

        let l_dim_of_ip = file.read_i16::<LittleEndian>().unwrap();

        // for i in 0..dim_of_ip {
        //     // let min_point = file.read_i32::<LittleEndian>().unwrap();
        //     // let len_along_axis = file.read_i32::<LittleEndian>().unwrap();
        //     // let stride = file.read_i32::<LittleEndian>().unwrap();
        // }
        // Skip min_point
        file.seek(SeekFrom::Current((32/8)*(l_dim_of_ip as i64))).unwrap();

        // Read len_along_axis
        let mut l_len_along_axis: Vec<i32> = Vec::with_capacity(l_dim_of_ip as usize);
        for _ in 0..l_dim_of_ip {
            l_len_along_axis.push(file.read_i32::<LittleEndian>().unwrap());
        }

        // Skip stride
        file.seek(SeekFrom::Current((32/8)*(l_dim_of_ip as i64))).unwrap();

        // read 2*dim_of_ip c values into shapes_vec[shape_id].2
        let mut l_c: Vec<i32> = Vec::with_capacity(2*l_dim_of_ip as usize);
        for _ in 0..2*l_dim_of_ip {
            l_c.push(file.read_i32::<LittleEndian>().unwrap());
        }

        shapes_map.insert(l_shape_id,(l_dim_of_ip, l_len_along_axis, l_c));
    }

    // Read total number of origins
    let num_origins = file.read_i32::<LittleEndian>().unwrap();
    // let mut data_offset: i32 = 0;

    //                          shape_id, base_row, base_col
    let mut origin_shapes: Vec<(i16,i32,i32)> = Vec::with_capacity(num_origins as usize);
    for _ in 0..num_origins {
        let shape_id = file.read_i16::<LittleEndian>().unwrap();
        let base_row = file.read_i32::<LittleEndian>().unwrap();
        let base_col = file.read_i32::<LittleEndian>().unwrap();
        // skip reading data_offset
        file.seek(SeekFrom::Current(32/8)).unwrap();

        // NO NEED TO JUMP TO DATA OFFSET, AS WE READ IT ALL TOGETHER AT THE END. WE JUST POPULATE ROW AND COL VECTORS
        // Read data offset as the num of elements. We can calculate this, but this is easier
        // data_offset = file.read_i32::<LittleEndian>().unwrap() - data_offset;
        // let curr_pos = file.seek(SeekFrom::Current(0)).unwrap();
        // file.seek(SeekFrom::Start(data_ptr as u64 + data_offset as u64)).unwrap();

        origin_shapes.push((shape_id, base_row, base_col));

        // Return to previous position
        // file.seek(SeekFrom::Start(curr_pos)).unwrap();
    }

    // Load ninc coords in memory
    let mut ninc_rowvec: Vec<usize> = Vec::with_capacity((nnz - inc_nnz) as usize);
    let mut ninc_colvec: Vec<usize> = Vec::with_capacity((nnz - inc_nnz) as usize);

    // Read uninc_format
    let uninc_format = file.read_u8().unwrap();
    match uninc_format {
        0 => {  //eprintln!("Reading CSR");
                let mut last_row_cnt = file.read_i32::<LittleEndian>().unwrap();

                // Read rowptr
                for curr_row in 0..nrows {
                    let row_cnt = file.read_i32::<LittleEndian>().unwrap();
                    // Insert row_cnt - last_row_cnt, curr_row values into rowvec
                    ninc_rowvec.extend(vec![curr_row as usize; (row_cnt - last_row_cnt) as usize]);
                    last_row_cnt = row_cnt;
                }
                // Read colidx
                for _ in 0..nnz-inc_nnz {
                    ninc_colvec.push(file.read_i32::<LittleEndian>().unwrap() as usize);
                }
             },
        2 => {  // eprintln!("Reading COO");
                // Read rowptr
                for _ in 0..nnz-inc_nnz {
                    ninc_rowvec.push(file.read_i32::<LittleEndian>().unwrap() as usize);
                }
                // Read colptr
                for _ in 0..nnz-inc_nnz {
                    ninc_colvec.push(file.read_i32::<LittleEndian>().unwrap() as usize);
                }
             },
        _ => { panic!("{} Read uninc_format variable value was {} and is unsupported at the moment. Report this issue.", "[ERROR]".bold().red(), uninc_format) }
    }

    // seek to data_ptr
    file.seek(SeekFrom::Start(data_ptr as u64)).unwrap();

    // Read data
    for _ in 0..nnz {
        datavec.push(file.read_f64::<LittleEndian>().unwrap());
    }

    let now = Instant::now();
    /*********************************** NOW PROCESS THE DATA IN MEMORY INTO A CSx MATRIX ***********************************/
    origin_shapes.iter().for_each(|(shape_id, base_row, base_col)| {
        // traverse row and col from l_row and l_col, and push index into rowvec and colvec
        let (l_dim_of_ip, l_len_along_axis, l_c) = shapes_map.get(shape_id).unwrap();

        recursive_populate_row_col_vec(*l_dim_of_ip, l_len_along_axis, l_c, &mut rowvec, &mut colvec, *base_row, *base_col);
    });
    // Add ninc coords to the end
    rowvec.extend(ninc_rowvec);
    colvec.extend(ninc_colvec);

    let coo_mat = TriMat::from_triplets((nrows as usize,ncols as usize), rowvec, colvec, datavec);

    let csx_matrix: CsMat<f64>;
    if csr {
        csx_matrix = coo_mat.to_csr();
    } else {
        csx_matrix = coo_mat.to_csc();
    }
    /************************************************************************************************************************/
    let elapsed = now.elapsed();
    println!("{} Converting UZP file: {} took: {}.{:09} seconds", "[TIME]".green().bold(), input_uzp_file_path, elapsed.as_secs(), elapsed.subsec_nanos());
    std::io::stdout().flush().unwrap();

    // Write matrix to file
    sprs::io::write_matrix_market(output_mtx_file_path, &csx_matrix).unwrap();
}

#[inline(always)]
#[allow(dead_code)]
fn recursive_populate_row_col_vec(l_dim_of_ip: i16, l_len_along_axis: &[i32], l_c: &[i32], rowvec: &mut Vec<usize>, colvec: &mut Vec<usize>, base_row: i32, base_col: i32) {
    if l_dim_of_ip < 2 /* l_dim_of_ip == 1 basically */ {
        for ii in 0..=l_len_along_axis[0] {
            rowvec.push((base_row + (l_c[0] * ii)) as usize);
            colvec.push((base_col + (l_c[1] * ii)) as usize);
        }
    } else {
        for ii in 0..=l_len_along_axis[0] {
            recursive_populate_row_col_vec(
                l_dim_of_ip-1, &l_len_along_axis[1..], &l_c[2..], rowvec, colvec,
                base_row + (l_c[0] * ii),
                base_col + (l_c[1] * ii)
            )
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn recursive_traverse(metapattern_piece: &MetaPatternPiece, metapattern_id: i32, meta_patterns: &LinkedHashMap<i32, MetaPattern>, f64_value_matrix: &CsMat<f64>) -> Vec<f64> {
    let (row,col) = metapattern_piece;
    let ((n,i,j), order, subpat) = meta_patterns.get(&metapattern_id).unwrap();
    let mut v = vec![];
    if *order < 2 {
        for ii in 0..*n {
            let value = f64_value_matrix.get((*row as i64 + (*i as i64 * ii as i64)) as usize, (*col as i64 + (*j as i64 * ii as i64)) as usize).unwrap();
            v.push(*value);
        }
    } else {
        for ii in 0..*n {
            v.append(
                &mut recursive_traverse(
                    &(
                        (*row as i64 + (*i as i64 * ii as i64)) as usize,
                        (*col as i64 + (*j as i64 * ii as i64)) as usize
                    ),
                    subpat.unwrap(),
                    meta_patterns,
                    f64_value_matrix
                )
            )
        }
    }

    return v;
}

#[inline(always)]
#[allow(dead_code)]
fn format_eqs(u: &Vec<Vec<i32>>, w: &Vec<i32>) -> String {
    let mut str_list: Vec<String> = vec![];

    let idx_values = vec!["i", "j", "k", "l"];

    for i in 0..u.len() {
        str_list.push("   ===   ".to_string());
        for j in 0..u[i].len(){
            str_list.push(format!("{sign}{variable} {weight} >= 0", sign={
                match u[i][j] {
                    1 => "+".to_string(),
                    -1 => "-".to_string(),
                    0 => "".to_string(),
                    _ => u[i][j].to_string()
                }
            }, variable={
                match u[i][j] {
                    0 => "",
                    _ => idx_values[j]
                }
            }, weight={
                match w[i] {
                    0 => "\x08".to_string(),    // This adds a backspace for avoiding double space between variable and >=
                    _ => format!("{} {}", {if w[i] < 0 {"-"} else {"+"}}, w[i].abs())
                }
            }));
        }
    }
    str_list.push("\n".to_string());

    str_list.join("")
}
