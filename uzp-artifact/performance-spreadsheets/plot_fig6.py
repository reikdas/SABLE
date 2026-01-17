#! /usr/sbin/python

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def gflops_scatterplot( df, version ):

    df = df.loc[:,version,"double","cold"]

    df.query("nnz>10").plot(kind="scatter",x="nnz",y="gflops",logx=True,s=50,alpha=.7,edgecolor="k",label=f"{version} (avg. {df.gflops.mean():.2f} GFLOPS)",ax=plt.gca())
    plt.xlabel( "NNZ (log scale)" )
    plt.ylabel( f"{version} GFLOPS" )
    plt.ylim( [plt.ylim()[0], 10] )
    plt.legend( loc=3, bbox_to_anchor=(0,1) )

    plt.axhline( df.gflops.max(), color="r", linestyle=":", linewidth=3 )
    plt.annotate( "Peak performance: %.02f GFLOPS" % (df.gflops.max()), (10,df.gflops.max()+.1), font = {'weight' : 'bold', 'size' : 12}, color="r"  )#, xytext=None, xycoords='data', textcoords=None, arrowprops=None, annotation_clip=None, **kwargs)

    plt.savefig( f"fig6_{version}.pdf" )

if __name__ == "__main__":
    sns.set_context("paper")
    sns.set( font_scale=1 )
    sns.set_style( "whitegrid" )
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure( figsize=(854/100,480/100) )

    if len( sys.argv ) < 2:
        print( "Usage: plot_fig6.py [ mkl | uzp | csr | csr5 ] <input.csv>" )
        sys.exit( 1 )

    input_csv = None
    if len( sys.argv ) == 3: input_csv = sys.argv[2]

    # Validate param
    version = sys.argv[1]
    match version:
        case "mkl" : pass
        case "uzp" : pass
        case "csr" : pass
        case "csr5": pass
        case _:
            printf( f"[ERROR] Invalid version: {version}. Must be 'mkl', 'uzp', 'csr', or 'csr5'." )
            sys.exit( 1 )

    df_orig = pd.read_pickle( "pldi25_uzp_results_singlethread.pickle" )
    if input_csv: 
        df = pd.read_csv( input_csv ).set_index( ["matrix","kernel","data_type","cache_mode"] )
        df['nnz'] = df_orig.nnz
    else: df = df_orig
    gflops_scatterplot( df, version )
