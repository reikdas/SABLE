#! /usr/sbin/python

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    sns.set_context("paper")
    sns.set( font_scale=1 )
    sns.set_style( "whitegrid" )
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    markersize=50

    if len( sys.argv ) < 4 and len( sys.argv ) != 7:
        print( "Usage: plot_fig11.py [ mkl | uzp | csr5 ] [ double | float ] [ hot | cold ] <input1.csv> <input2.csv> <input3.csv>" )
        sys.exit( 1 )

    input_csv1, input_csv2, input_csv3 = None, None, None
    if len( sys.argv ) == 7:
        input_csv1, input_csv2, input_csv3 = sys.argv[4:]

    # Validate params
    def validate_version( v ):
        match v:
            case "mkl" : pass
            case "uzp" : pass
            case "csr5": pass
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

    version   = sys.argv[1]
    precision = sys.argv[2]
    cache     = sys.argv[3]

    validate_version( version )
    axeslim,gflopslabel = validate_precision( precision )
    validate_cachemode( cache )
    validate_vp_pair( version, precision )

    df_orig = pd.read_pickle( "pldi25_uzp_results_multithread.pickle" ).query( "nnz > 1e5" )
    if input_csv1:
        df1 = pd.read_csv( input_csv1 ).set_index( ["matrix","kernel","nthreads","data_type","cache_mode"] )
        df2 = pd.read_csv( input_csv1 ).set_index( ["matrix","kernel","nthreads","data_type","cache_mode"] )
        df3 = pd.read_csv( input_csv1 ).set_index( ["matrix","kernel","nthreads","data_type","cache_mode"] )
        df = pd.concat( [df1, df2, df3] )
    else: df = df_orig

    plt.scatter( df.loc[:,version,"1",precision,cache].gflops, df.loc[:,version,"2",precision,cache].gflops, color="b", label=f'2 threads (avg: {df.loc[:,version,"2",precision,cache].gflops.mean():.2f} {gflopslabel})', edgecolor="k", alpha=.7, s=markersize )
    plt.scatter( df.loc[:,version,"1",precision,cache].gflops, df.loc[:,version,"8",precision,cache].gflops, color="r", label=f'8 threads (avg: {df.loc[:,version,"8",precision,cache].gflops.mean():.2f} {gflopslabel})', edgecolor="k", alpha=.7, marker="x", linewidth=2, s=markersize  )

    # 8x line
    plt.axline((0, 0), slope=8, color="black", linewidth=2, linestyle=":" )
    x_text = 2
    y_text = 16.1
    plt.text( x_text, y_text, "speedup = 8x", ha="center", va="bottom" )

    # 8-threads regression
    model = LinearRegression( fit_intercept=False )
    X = df.loc[:,version,"1",precision,cache].gflops.values.reshape((-1,1))
    y = df.loc[:,version,"8",precision,cache].gflops
    model.fit(X,y)
    plt.axline( (0,model.intercept_), slope=model.coef_[0], linewidth=3, alpha=.7, color="r", label=f"8t reg. slope = {model.coef_[0]:.2f}" )

    # 2x line
    plt.axline((0, 0), slope=2, color="black", linewidth=2, linestyle=":" )
    x_text = 8
    y_text = 16.1
    plt.text( x_text, y_text, "speedup = 2x", ha="center", va="bottom" )

    # 2-threads regression
    model = LinearRegression( fit_intercept=False )
    X = df.loc[:,version,"1",precision,cache].gflops.values.reshape((-1,1))
    y = df.loc[:,version,"2",precision,cache].gflops
    model.fit(X,y)
    plt.axline( (0,model.intercept_), slope=model.coef_[0], linewidth=3, alpha=.7, label=f"2t reg. slope = {model.coef_[0]:.2f}" )


    plt.xlim( [0,10] )
    plt.ylim( [0,16] )
    plt.xlabel( f"{version} ST {gflopslabel}" )
    plt.ylabel( f"{version} MT {gflopslabel}" )
    plt.legend()

    plt.savefig( f"fig11_{version}_{precision}_{cache}.pdf", bbox_inches='tight' )
