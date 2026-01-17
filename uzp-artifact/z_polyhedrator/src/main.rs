extern crate sprs;
extern crate text_io;

use std::io::Write;
use std::process::exit;
use std::time::Instant;
use colored::Colorize;
use project_root::get_project_root;

mod spsearch;
#[allow(unused_imports)]
use crate::spsearch::{SpSearchMatrix,SpSearchPatternsFlags};

mod spaugment;
#[allow(unused_imports)]
use crate::spaugment::SpAugment;

mod uzpgen;
#[allow(unused_imports)]
use crate::uzpgen::UZPGen;

#[macro_use(c)]
extern crate cute;

mod utils;

mod flags {
    use std::path::PathBuf;

    xflags::xflags! {
        cmd z_polyhedrator {
            /// Search for (meta)patterns in a matrixmarket file. Optionally augment dimensionality and write to UZP file.
            cmd search {
                /// File containing pattern list
                required patterns_file_path: PathBuf

                /// Input MatrixMarket file
                required matrixmarket_file_path: PathBuf

                /// Print patterns parsed from pattern list
                optional --print-pattern-list

                /// Print 1D piece list (AST list) before any dimensionality augmentation
                optional --print-ast-list

                /// Print uwc and distinct uwc lists after dimensionality augmentation
                optional --print-uwc-list

                /// Transpose matrix at input
                optional -ti, --transpose-input

                /// Transpose matrix at output
                optional -to, --transpose-output

                /// [2D SEARCH] Search Flags. Valid options: {[PatternFirst], CellFirst} where [] = default.
                optional --search-flags search_flags: String

                /// Write to custom UZP file. Writes to <output_uzp_file_path>.<N>d.uzp
                optional -w,--write-uzp output_uzp_file_path: PathBuf

                /// Augment dimensionality
                optional -a, --augment-dimensionality augment_dimensionality: usize

                /// Minimum piece length for dimensionality augmentation
                optional -pl, --augment-dimensionality-piece-cutoff augment_dimensionality_piece_cutoff: usize

                /// Min stride for augment dimensionality search
                optional -psmin, --augment-dimensionality-piece-stride-min augment_dimensionality_piece_stride_min: usize

                /// Max stride for augment dimensionality search
                optional -psmax, --augment-dimensionality-piece-stride-max augment_dimensionality_piece_stride_max: usize

                /// Write not included single-points as 1-length patterns
                optional --write-uninc-as-patterns

                /// Enable experimental features
                optional --experimental
            }

            /// Convert UZP file to MTX file, in either CSC or CSR format
            cmd convert {
                /// Input UZP file
                required input_uzp_file_path: PathBuf

                /// Output mtx file
                required output_mtx_file_path: PathBuf

                /// Output in csc format (default)
                optional --csc

                /// Output in csr format
                optional --csr

                /// Print 1D piece list (AST list). It only works in 1D shapes. Useful for utils/plot_ast_2d.py
                optional --print-ast-list
            }

            /// Convert UZP file to MTX file, in either CSC or CSR format. Modified into a overall slower version for timing purposes (CPU and Disk operations separated in time)
            cmd convert_timing {
                /// Input UZP file
                required input_uzp_file_path: PathBuf

                /// Output mtx file
                required output_mtx_file_path: PathBuf

                /// Output in csc format (default)
                optional --csc

                /// Output in csr format
                optional --csr
            }
        }
    }
}

fn main() {
    let project_root = get_project_root();
    match project_root {
        Ok(path) => {
            eprintln!("{} Project root is: {:?}", "[INFO]".cyan().bold(), path);
        }
        Err(e) => {
            eprintln!("{} {:?}. Exiting...", "[ERROR]".red().bold(), e);
            exit(-1);
        }
    }

    match flags::Z_polyhedrator::from_env() {
        Ok(matrix_flags) => {
            match matrix_flags.subcommand {
                flags::Z_polyhedratorCmd::Search(flags) => {
                    let patterns_file_path = flags.patterns_file_path.to_str().unwrap();
                    let matrixmarket_file_path = flags.matrixmarket_file_path.to_str().unwrap();

                    /****** EXPERIMENTAL FLAGS SUMMARY ******/
                    if flags.experimental {
                        eprintln!("{} Experimental features enabled. Use with caution.", "[WARNING]".yellow().bold());
                        if flags.write_uninc_as_patterns {
                            eprintln!("{} Experimental feature {} enabled. Use with caution.", "[WARNING]".yellow().bold(), "--write-uninc-as-patterns".yellow().bold());
                        }
                    } else {
                        if flags.write_uninc_as_patterns {
                            eprintln!("{} Enable experimental features with {} flag.", "[ERROR]".red().bold(), "--experimental".yellow().bold());
                            exit(-1);
                        }
                    }
                    /****************************************/

                    let output_uzp_file_path: (bool, String);
                    output_uzp_file_path = {
                        if flags.write_uzp.as_ref().is_some() {
                            (true, String::from(flags.write_uzp.unwrap().to_str().unwrap()))
                        } else {
                            (false, String::new())
                        }
                    };

                    let mut search_flags_str: String = "[default]".to_string();

                    let search_flags = {
                        let mut l_search_flags = spsearch::SpSearchPatternsFlags::NoFlags;

                        if flags.search_flags.is_some() {
                            search_flags_str = flags.search_flags.unwrap().to_string();
                            match search_flags_str.as_str() {
                                "PatternFirst" => l_search_flags |= spsearch::SpSearchPatternsFlags::PatternFirst,
                                "CellFirst" => l_search_flags |= spsearch::SpSearchPatternsFlags::CellFirst,
                                def => {
                                    eprintln!("invalid value `{}` for `--search-flags`. Valid options: {{[PatternFirst], CellFirst}} where [] = default.", def);
                                    exit(-1);
                                }
                            }
                        } else {
                            l_search_flags |= spsearch::SpSearchPatternsFlags::PatternFirst;
                        }

                        l_search_flags
                    };

                    /* -------- PARSE -------- */
                    eprintln!("{} Opening matrixmarket file: {}", "[INFO]".cyan().bold(), matrixmarket_file_path);
                    std::io::stderr().flush().unwrap();
                    let now = Instant::now();

                    let mut base_matrix: SpSearchMatrix = SpSearchMatrix::from_file(matrixmarket_file_path, flags.transpose_input);

                    let elapsed = now.elapsed();
                    println!("{} Opening matrixmarket file: {} took: {}.{:03} seconds", "[TIME]".green().bold(), matrixmarket_file_path, elapsed.as_secs(), elapsed.subsec_millis());
                    std::io::stdout().flush().unwrap();

                    eprintln!("{} Opening patterns file: {}", "[INFO]".cyan().bold(), patterns_file_path);
                    std::io::stderr().flush().unwrap();
                    let now = Instant::now();

                    base_matrix.load_patterns(patterns_file_path);

                    let elapsed = now.elapsed();
                    println!("{} Opening patterns file: {} took: {}.{:03} seconds", "[TIME]".green().bold(), patterns_file_path, elapsed.as_secs(), elapsed.subsec_millis());
                    std::io::stdout().flush().unwrap();

                    if flags.print_pattern_list {
                        eprintln!("--- Pattern list ---");
                        base_matrix.print_patterns();
                    }

                    /* -------- SEARCH -------- */
                    eprintln!("{} Searching for patterns with flags {}... ", "[INFO]".cyan().bold(), search_flags_str);
                    std::io::stderr().flush().unwrap();
                    let now = Instant::now();

                    base_matrix.search_patterns(search_flags);

                    let elapsed = now.elapsed();
                    println!("{} Searching for patterns with flags {} took: {}.{:09} seconds", "[TIME]".green().bold(), search_flags_str, elapsed.as_secs(), elapsed.subsec_nanos());
                    std::io::stdout().flush().unwrap();

                    /* -------- PRINT AST LIST IF REQUIRED -------- */
                    if flags.print_ast_list {
                        base_matrix.print_pieces();
                    }

                    /* -------- AUGMENT DIMENSIONALITY AND WRITE UZP FILE IF REQUIRED -------- */
                    let augment_dimensionality: usize = flags.augment_dimensionality.unwrap_or(1);
                    let augment_dimensionality_piece_cutoff: usize = flags.augment_dimensionality_piece_cutoff.unwrap_or(2);
                    let augment_dimensionality_piece_stride_max: usize = flags.augment_dimensionality_piece_stride_max.unwrap_or(std::usize::MAX);
                    let augment_dimensionality_piece_stride_min: usize = flags.augment_dimensionality_piece_stride_min.unwrap_or(0);

                    if flags.print_uwc_list || output_uzp_file_path.0 || augment_dimensionality > 1 {
                        let mut uzpgen = UZPGen::from_piece_list(base_matrix.get_piece_list(), base_matrix.numrows, base_matrix.numcols, base_matrix.nonzeros);

                        let mut spaugment;
                        if augment_dimensionality > 1 {
                            // Augment dimensionality
                            spaugment = SpAugment::from_1d_origin_uwc_list(uzpgen.get_orig_uwc_list(), uzpgen.nrows, uzpgen.ncols, uzpgen.nnz);

                            eprintln!("{} Augmenting dimensionality... ", "[INFO]".cyan().bold());
                            std::io::stderr().flush().unwrap();
                            let now = Instant::now();

                            spaugment.augment_dimensionality(augment_dimensionality, augment_dimensionality_piece_cutoff, augment_dimensionality_piece_stride_min, augment_dimensionality_piece_stride_max);

                            let elapsed = now.elapsed();
                            println!("{} Augmenting dimensionality took: {}.{:03} seconds", "[TIME]".green().bold(), elapsed.as_secs(), elapsed.subsec_millis());
                            std::io::stdout().flush().unwrap();

                            // And update uzpgen accordingly
                            uzpgen = UZPGen::from_metapatterns_list(spaugment.get_metapatterns(), spaugment.get_metapattern_pieces(), uzpgen.nrows, uzpgen.ncols, uzpgen.nnz, uzpgen.inc_nnz);
                        }

                        if flags.print_uwc_list {
                            uzpgen.print_uwc_list(true);
                            uzpgen.print_distinct_uwc_list(true);
                        }

                        if output_uzp_file_path.0 {
                            eprintln!("{} Writing UZP file... ", "[INFO]".cyan().bold());
                            std::io::stderr().flush().unwrap();
                            let now = Instant::now();

                            uzpgen.write_uzp(&matrixmarket_file_path, &format!("{}.{}d.uzp", &output_uzp_file_path.1, augment_dimensionality), flags.transpose_input, flags.transpose_output, flags.write_uninc_as_patterns);

                            let elapsed = now.elapsed();
                            println!("{} Writing UZP file took: {}.{:03} seconds", "[TIME]".green().bold(), elapsed.as_secs(), elapsed.subsec_millis());
                            std::io::stdout().flush().unwrap();
                        }

                    }
                }

                flags::Z_polyhedratorCmd::Convert(flags) => {
                    let input_uzp_file_path = flags.input_uzp_file_path.to_str().unwrap();
                    let output_mtx_file_path = flags.output_mtx_file_path.to_str().unwrap();

                    eprintln!("{} Converting UZP file: {}... ", "[INFO]".cyan().bold(), input_uzp_file_path);
                    std::io::stderr().flush().unwrap();
                    let now = Instant::now();

                    uzpgen::convert_uzp(input_uzp_file_path, output_mtx_file_path, flags.csr && !flags.csc, flags.print_ast_list);

                    let elapsed = now.elapsed();
                    if !flags.print_ast_list {
                        println!("{} Converting UZP file: {} took: {}.{:03} seconds", "[TIME]".green().bold(), input_uzp_file_path, elapsed.as_secs(), elapsed.subsec_millis());
                    }
                    std::io::stdout().flush().unwrap();
                }

                flags::Z_polyhedratorCmd::Convert_timing(flags) => {
                    let input_uzp_file_path = flags.input_uzp_file_path.to_str().unwrap();
                    let output_mtx_file_path = flags.output_mtx_file_path.to_str().unwrap();

                    eprintln!("{} Converting UZP file: {}... ", "[INFO]".cyan().bold(), input_uzp_file_path);
                    std::io::stderr().flush().unwrap();

                    // Conversion time is measured inside the function
                    uzpgen::convert_uzp_for_timing(input_uzp_file_path, output_mtx_file_path, flags.csr && !flags.csc);
                }
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            exit(0);
        }
    }
}
