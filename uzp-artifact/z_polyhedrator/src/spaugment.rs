use colored::Colorize;
use itertools::{Itertools, enumerate};
use linked_hash_map::LinkedHashMap;
use sprs::{CsMat, TriMat};

use crate::utils::{Pattern,Piece,OriginUwc,MetaPattern,MetaPatternPiece};
use crate::utils::orig_uwc_to_piece_1d;

#[allow(dead_code)]
pub struct SpAugment {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
    meta_patterns: LinkedHashMap<i32, MetaPattern>,
    meta_pattern_pieces: LinkedHashMap<MetaPatternPiece, i32>
}

impl SpAugment {
    pub fn from_1d_origin_uwc_list(origin_uwc_list: Vec<(OriginUwc, i32)>, nrows: usize, ncols: usize, nnz: usize) -> Self {

        // DEBUG UNCOMMENT
        // println!("Printing orig_uwc_list from spaugment!");
        // origin_uwc_list.iter().for_each(|((x,y,(u,w,c)), id)| {
        //     print!("({:?},{:?})\t   =\t{:?}\t{:?}\t{:?}\t{:?}\n", x, y, id, u, w, c);
        // });

        let meta_pattern_pieces = origin_uwc_list
            .iter()
            .map(|(ouwc, id)| {
                let (x,y,_) = orig_uwc_to_piece_1d(ouwc);
                ((x,y), *id)
            })
            .collect::<LinkedHashMap<MetaPatternPiece, i32>>();

        // NOTE HERE! The key -1 does not have to be in this list! It is only included on the distinct patterns list from uzpgen temporarily, but it is NOT necessary.
        let mut meta_patterns: LinkedHashMap<i32, MetaPattern> = LinkedHashMap::new();
        for ((_,_,(n,i,j)), id) in origin_uwc_list.iter().map(|(ouwc, id)| (orig_uwc_to_piece_1d(ouwc), id)) {
            if meta_patterns.get(id).is_some() { continue; }
            else {
                meta_patterns.insert(*id, ((n,i,j), 1, None));
            }
        };

        // DEBUG UNCOMMENT
        // println!("MP_Pieces:\n{:?}", meta_pattern_pieces);
        // println!("MP_Dict:\n{:?}\nLen: {}", meta_patterns, meta_patterns.len());

        SpAugment { 
            nrows: nrows, 
            ncols: ncols,
            nnz: nnz,
            meta_patterns: meta_patterns,
            meta_pattern_pieces: meta_pattern_pieces
        }
    }

    pub fn augment_dimensionality(&mut self, target_dim: usize, piece_cutoff: usize, min_stride: usize, max_stride: usize) {

        if piece_cutoff < 2 {
            panic!("\n{} How are you supposed to make length={} pieces?", "[spaugment]".red().bold(), piece_cutoff);
        }

        // DEBUG UNCOMMENT
        println!("\n------- AUGMENT DIMENSIONALITY -------\n");

        let single_compensation: i64 = match self.meta_patterns.get(&-1) {
            Some(_) => -1i64,
            None => 0i64
        };
        // DEBUG UNCOMMENT
        // println!("Compensation = {} (this accounts for elements with id=-1)", single_compensation);

        // PRECONDITION: Metapatterns are not necessarily ordered, but same metapatterns are consecutive in the list

        // FIXME Maybe generalize this to support any-dimensional input
        for curr_dim in 2..=target_dim {
            // DEBUG UNCOMMENT
            // println!("Searching for {}D", curr_dim);

            // Aux variables for processing
            let mut origins_list: Vec<(i32, i32)> = vec![];
            let mut curr_id: i32 = 0; // Zero here to save an iteration (only if meta_pattern_pieces is sorted). Not really needed apart from that
            
            // Advance pointer to current dimensionality pieces
            let mut metapat_pieces_iter = self.meta_pattern_pieces.iter()
                .filter(|(_,id)| **id != -1);

            let mut start_id = (self.meta_patterns.len() as i64 + single_compensation) as i32;
            let curr_dim_start_id = start_id;

            let mut new_metapats: LinkedHashMap<i32, MetaPattern> = LinkedHashMap::new();
            let mut new_metapat_pieces: LinkedHashMap<MetaPatternPiece, i32> = LinkedHashMap::new();

            // This loop loads all consecutive metapatterns with the same id into origins_list, then executes the "if opt.is_none() [...]"
            loop {
                let opt = metapat_pieces_iter.next();
                
                if opt.is_none() || matches!(opt, Some((_,id)) if curr_id != *id) {
                    // DEBUG UNCOMMENT
                    println!("\n------- compute_metapatterns for id = {} -------", curr_id);

                    // Compute metapatterns FIXME parametrize max and min strides
                    match compute_metapatterns(&mut origins_list, piece_cutoff, start_id, curr_id, min_stride, max_stride) {
                        Some((l_new_metapats, l_new_metapat_pieces)) => {
                            start_id += l_new_metapats.len() as i32;

                            // Extend new metapats
                            new_metapats.extend(l_new_metapats);
                            new_metapat_pieces.extend(l_new_metapat_pieces);
                        },
                        None => {},
                    }

                    // And prepare for next batch
                    origins_list.clear();

                    // Breaking if end was reached
                    if opt.is_none() { break; }

                    // Else update curr_id
                    let (_,id) = opt.unwrap();
                    curr_id = *id;
                }

                // Append consecutive same-id metapattern pieces to origins_list
                let ((x,y),_) = opt.unwrap();
                origins_list.push((*x as i32, *y as i32));
            }

            // DEBUG UNCOMMENT
            // println!("Startptr: --. Curr_id: {:?}", curr_id);

            // Add new_metapat_pieces and new_metapats to current ones
            let new_metapats_len = new_metapats.len() as i32;
            self.meta_patterns.extend(new_metapats);
            self.meta_pattern_pieces.extend(new_metapat_pieces);

            // In the case of the metapattern pieces, this is a little bit more troublesome, as we have to invalidate
            // last-d pieces contained in higher order pieces.
            // As we are traversing high order pieces, we now rewrite *order* value.
            let pieces = self.meta_pattern_pieces.iter()
                .filter(|(_,val)| (**val >= curr_dim_start_id) && (**val < (curr_dim_start_id+new_metapats_len)))
                .map(|((x,y),id)| ((*x,*y), *id))
                .collect::<Vec<_>>();

            // DEBUG UNCOMMENT
            println!(" ------ INVALIDATE AND UPDATE ORDER ------ ");
            for ((orig_x, orig_y), low_order_id) in pieces {
                println!(" - {:?}: {}", (orig_x, orig_y), low_order_id);

                // get pattern
                let mp = self.meta_patterns.get(&low_order_id).unwrap();

                let (n,i,j) = mp.0;

                // Update dimensionality of current metapattern
                self.meta_patterns.get_mut(&low_order_id).unwrap().1 = curr_dim as i32;

                // Skip the first one as it is already updated (hashmap properties)
                for ii in 1..n {
                    self.meta_pattern_pieces.remove(
                        &((orig_x as i64 + (i as i64 * ii as i64)) as usize, (orig_y as i64 + (j as i64 * ii as i64)) as usize)
                    );
                }
            }

            // DEBUG UNCOMMENT
            // println!("for loop going from [{} to {})", curr_dim_start_id, curr_dim_start_id + new_metapats_len);
            // println!("MP_Pieces:\n{:?}", self.meta_pattern_pieces);q
            // println!("MP_Dict:\n{:?}\nLen: {}", self.meta_patterns, self.meta_patterns.len());
        } // for dims
    }

    pub fn get_metapatterns(&self) -> LinkedHashMap<i32, MetaPattern> {
        return self.meta_patterns.clone();
    }

    pub fn get_metapattern_pieces(&self) -> LinkedHashMap<MetaPatternPiece, i32> {
        return self.meta_pattern_pieces.clone();
    }
}

#[inline(always)]
#[allow(dead_code)]
fn compute_metapatterns(origins_list: &mut Vec<(i32, i32)>, piece_cutoff: usize, start_id: i32, low_order_id: i32, min_stride: usize, max_stride: usize) -> Option<(LinkedHashMap<i32, MetaPattern>, LinkedHashMap<MetaPatternPiece, i32>)> {
    // DEBUG UNCOMMENT
    // println!("Metapatterns: {:?}", origins_list);

    let origin_list_len = origins_list.len();

    // No feasible higher order metapatterns
    if origin_list_len < piece_cutoff {
        println!("  -> Skip for pieces from id={} as len = {} < {} = piece cutoff", low_order_id, origin_list_len, piece_cutoff);
        return None;
    }

    let mut meta_pattern_list: LinkedHashMap<i32, MetaPattern> = LinkedHashMap::new();
    let mut meta_pattern_piece_list: LinkedHashMap<MetaPatternPiece, i32> = LinkedHashMap::new();

    let fn_tuple_sub = |(x1,y1):(i32,i32),(x2,y2):(i32,i32)| (x1-x2, y1-y2);

    let (_,max_col) = *origins_list.iter().max_by_key(|(_,col)| *col).unwrap();
    let (max_row,_) = *origins_list.iter().max_by_key(|(row,_)| *row).unwrap();

    // DEBUG UNCOMMENT
    // println!("Max col = {}, Max row = {}", max_col, max_row);

    // Get all strides between pieces
    let strides = origins_list
        .iter()
        .tuple_combinations()
        .map(|(a,b)| fn_tuple_sub (*b, *a))
        .filter(|(sx,sy)| {
            let (absx, absy) = (i32::abs(*sx) as usize, i32::abs(*sy) as usize);
            absx <= max_stride && absy <= max_stride && absx >= min_stride && absy >= min_stride
        })
        .collect::<Vec<(i32,i32)>>();

    // DEBUG UNCOMMENT
    // println!("STRIDES: {:?}", strides);

    let mut occurrences = strides
        .iter()
        .into_group_map_by(|x| **x)
        .into_iter()
        // Very important to add one to the length of the strides. Each time that a piece is found, one has to be added too.
        .map(|(k,v)| (k, v.len() as u32 + 1u32))
        // solve tie on equal reps by prioritizing closer pieces. i64 to avoid OF
        .sorted_by_key(|((stride_x, stride_y),reps)| std::cmp::Reverse((*reps , ( -((*stride_x) as i64 * (*stride_x) as i64) ) as i64 - ((*stride_y) as i64 *(*stride_y) as i64) as i64 )))
        .collect::<LinkedHashMap<(i32,i32),u32>>();

    // println!("Strides = {:?}\nOccurrences = {:?}", strides, occurrences);

    // DEBUG UNCOMMENT
    // println!("OCCURRENCES: {:?}", occurrences);

    // Compose sparse matrix with origins_list
    let mut expl_matrix: TriMat<u8> = TriMat::with_capacity((max_row as usize +1, max_col as usize +1), origins_list.len());
    for (row,col) in origins_list {
        expl_matrix.add_triplet(*row as usize, *col as usize, 1u8);
    }
    let mut expl_matrix: CsMat<bool> = expl_matrix.to_csr().map(|_| false);

    // DEBUG UNCOMMENT
    // println!("Mat = {:?}", expl_matrix);

    let mut best_piece: Piece;
    let mut found_piece: bool;

    'L1: loop {
        // Reset best_piece and found_piece
        best_piece = (0,0,((piece_cutoff-1) as i32,0,0));
        found_piece = false;

        'L2: for (((stride_x, stride_y), n), ((_, _), next_n)) in occurrences.iter().circular_tuple_windows::<((&(i32, i32), &u32), (&(i32, i32), &u32))>() {
            // println!("  -> Comparing {:?} vs {:?}", n, next_n);

            // Most repeated pattern first (MRPF)
            'L3: for (idx, (_,(e_row, e_col))) in enumerate(expl_matrix.iter()){
                best_piece = (|p1:Piece,p2:Piece| std::cmp::max_by_key(p1, p2, |(_,_,(n,_,_))| *n)) (check_metapattern_reps(&expl_matrix, (e_row,e_col), &(*n as i32 + 1,*stride_x,*stride_y)), best_piece);

                // This is equal to the piece of code above. TODO check speed difference.
                // let curr_piece = check_metapattern_reps(&expl_matrix, (e_row,e_col), &(*n as i32 + 1,*stride_x,*stride_y));
                // if curr_piece.2.0 > best_piece.2.0 {
                //     best_piece = curr_piece;
                // }

                // FIXME check this now that no LHM reordering is being done
                // if bp.n >= next_n        or         there are no remaining points to build a piece
                if best_piece.2.0 as u32 >= *next_n+1 || best_piece.2.0 as usize >= origin_list_len-(idx as usize)-1 {
                    break 'L3;
                }
            } // 'L3

            if best_piece.2.0 >= (piece_cutoff as i32) {
                /*** APPEND ROUTINE ***/
                // Get suitable pattern id
                let pat_id: i32 = match meta_pattern_list.back() {
                    Some((id,((v_n, v_i, v_j), _, _))) => {
                        if *v_n == best_piece.2.0 && *v_i == best_piece.2.1 && *v_j == best_piece.2.2 {
                            *id
                        } else {
                            (*id)+1
                        }
                    },
                    None => start_id,
                };

                // Insert into metapattern list     n,i,j from best piece.
                // If they are equal then nothing changes and we save an if statement (2Bbenchmarkd)
                meta_pattern_list.insert(pat_id, (best_piece.2, 0, Some(low_order_id)));
                //                                                ^^^ This has to be replaced out of this function

                // Insert piece intro metapattern piece list
                meta_pattern_piece_list.insert((best_piece.0, best_piece.1), pat_id);

                println!("  -> Found piece! {:?}", best_piece);
                // Set flag for reordering occurrence list
                found_piece = true;
                break 'L2;
            }
        } // 'L2

        if found_piece {
            /*** FIX OCCURRENCES IN LHM ***/
            let (x,y,(n,i,j)) = best_piece;
            let remainder;
            {// scoped so it does not interfere
                let p = occurrences.get_mut(&(i,j)).unwrap();
                // add one to account for the creation of an aditional vertex
                remainder = *p - (n as u32) + 1u32;
                *p = remainder;
            }

            // Set to found (true) members of the new pattern
            for ii in 0..n {
                let pos_val = expl_matrix.get_mut((x as i64 + (i as i64 * ii as i64)) as usize, (y as i64 + (j as i64 * ii as i64)) as usize).unwrap();
                *pos_val = true;
            }
        } else {
            break 'L1;
        }
    }

    // DEBUG UNCOMMENT
    // println!("{}: MetaPatternList = {:?}", "[DEBUG]".cyan().bold(), MetaPatternList);
    // println!("{}: MetaPatternPieceList = {:?}", "[DEBUG]".cyan().bold(), MetaPatternPieceList);

    if meta_pattern_list.len() > 0 {
        return Some((meta_pattern_list, meta_pattern_piece_list));
    } else {
        return None;
    }
}

#[inline(always)]
#[allow(dead_code)]
fn check_metapattern_reps(csmat: &CsMat<bool>, curr_pos: (usize, usize), pattern: &Pattern) -> Piece {
    let &(max_n,i,j) = pattern;
    let (x,y) = curr_pos;

    // Discard already dumped patterns without computing bounds first
    if *csmat.get(x,y).unwrap() {
        // Could be all zeros, important thing is that N = 0
        return (x,y,(0,i,j));
    }

    // We can start on the next pattern (length would be 2 if only the first iteration completes)
    for ii in 1..max_n {
        let position = csmat.get((x as i64 + (i as i64 * ii as i64)) as usize, (y as i64 + (j as i64 * ii as i64)) as usize);
        match position {
            Some(&is_in_pat) => {
                if is_in_pat {
                    return (x,y,(ii,i,j));
                } else {
                    continue;
                }
            },
            None => return (x,y,(ii,i,j)),
        }
    }

    return (x,y,(max_n,i,j));
}