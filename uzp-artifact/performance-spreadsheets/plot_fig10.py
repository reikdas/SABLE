#! /usr/sbin/python

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import seaborn as sns

if __name__ == "__main__":
    sns.set_context("paper")
    sns.set( font_scale=1.5 )
    sns.set_style( "whitegrid" )
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['figure.figsize'] = 1920/100,1080/100

    if len( sys.argv ) < 5:
        print( "Usage: plot_fig10.py [ mkl | uzp | csr | csr5 | macveth ] [ mkl | uzp | csr | csr5 | macveth ] [ double | float ] [ hot | cold ] <input.csv>" )
        sys.exit( 1 )

    input_csv = None
    if len( sys.argv ) == 6: input_csv = sys.argv[5]

    # Validate params
    def validate_version( v ):
        match v:
            case "mkl" : pass
            case "uzp" : pass
            case "csr" : pass
            case "csr5": pass
            case "macveth": pass
            case _:
                print( f"[ERROR] Invalid version: {v}. Must be 'mkl', 'uzp', 'csr', 'csr5', or 'macveth'." )
                sys.exit( 1 )
    def validate_precision( p ):
        match p:
            case "double": return [0,10], "GFLOPS"
            case "float": return [0,17], "SP-GFLOPS"
            case _:
                print( f"[ERROR] Invalid precision: {p}. Must be 'double', or 'float'." )
                sys.exit( 1 )
    def validate_cachemode( cm ):
        match cm:
            case "hot": pass
            case "cold": pass
            case _:
                print( f"[ERROR] Invalid cache mode: {cm}. Must be 'hot', or 'cold'." )
                sys.exit( 1 )
    def validate_vp_pair( v, p ):
        match v:
            case "csr5":
                match p:
                    case "float":
                        print( f"[ERROR] Invalid version / precision combination: {v} does not support {p}." )
                        sys.exit(1)
                    case _: pass
            case "macveth":
                match p:
                    case "double":
                        print( f"[ERROR] Invalid version / precision combination: {v} does not support {p}." )
                        sys.exit(1)
                    case _: pass
            case _: pass

    version1  = sys.argv[1]
    version2  = sys.argv[2]
    precision = sys.argv[3]
    cache     = sys.argv[4]

    validate_version( version1 )
    validate_version( version2 )
    axeslim,gflopslabel = validate_precision( precision )
    validate_cachemode( cache )
    validate_vp_pair( version1, precision )
    validate_vp_pair( version2, precision )

    df_orig = pd.read_pickle( "pldi25_uzp_results_singlethread.pickle" ).query( "nnz > 50" )
    if input_csv:
        df = pd.read_csv( input_csv ).set_index( ["matrix","kernel","data_type","cache_mode"] )
        df['nnz'] = df_orig.nnz
    else: df = df_orig

    # Plot jointplot
    sns.jointplot( x=df.loc[:,version1,precision,cache].gflops, y=df.loc[:,version2,precision,cache].gflops, xlim=axeslim, ylim=axeslim, s=50, alpha=0.7, linewidth=1, edgecolor="k", kind="scatter")
    plt.axline((0, 0), slope=1, color="black", linewidth=1, linestyle=":" )
    plt.xlabel( f"{version1} {gflopslabel}" )
    plt.ylabel( f"{version2} {gflopslabel}" )

    plt.savefig( f"fig10_{version1}_{version2}_{precision}_{cache}.pdf" )
