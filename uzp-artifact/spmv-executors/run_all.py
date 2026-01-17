#! /usr/sbin/python

import sys,os,time

import pandas as pd
import numpy as np
import re
def parse_results( path ):
    f = open( path, "r" )

    run = 0
    rows =  []
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    for l in f:
        l = ansi_escape.sub( "", l.strip() )
        if not l: continue
        if l.startswith( "[SCRIPT] Running test" ):
            words = l.split()
            group = words[-1].split("/")[0]
            mat   = words[-1].split("/")[1]
            continue
        if l.startswith( "[SCRIPT] FLOAT_DATA_TYPE:" ):
            data_type = l.split()[-1]
            continue
        if l.startswith( "[SCRIPT] CACHE_MODE:" ):
            cache_mode = l.split()[-1]
            continue
        if l.startswith( "[SCRIPT] EXECUTION_THREADS:" ):
            execution_threads = l.split()[-1]
            continue
        if "GFlops" in l:
            words = l.split()
            match( words[0] ):
                case "UZP"      : variant="uzp"
                case "MKL"      : variant="mkl"
                case "CSR5"     : variant="csr5"
                case "CSR"      : variant="csr"
                case "MACVETH"  : variant="macveth"
                case _: raise ValueError(l)
            drow = [mat, variant, run, data_type, cache_mode, float(words[-1])]
            rows.append( drow )
            run += 1
            continue

    f.close()
    df = pd.DataFrame( rows, columns=["matrix","kernel","run","data_type","cache_mode","gflops"] )
    df.set_index( ["matrix","kernel","run"], inplace=True )
    df.replace( np.inf, 0, inplace=True )
    gb = df[["gflops"]].groupby( ["matrix","kernel"] )

    # Remove outliers (mean + 3·sigma)
    df = df.loc[(gb.apply( lambda x: x-x.mean() ).droplevel( [2,3] ).divide( gb.std().replace(0,np.inf) ).abs().fillna(0) <= 3 ).all(axis=1)]

    df = df.loc[df.groupby(["matrix","kernel"]).gflops.idxmax()].droplevel(2)
    df.set_index( ["data_type","cache_mode"], append=True, inplace=True )
    return df


selected_tests = [
 ('JGD_Kocay', 'Trec3'),
 ('JGD_Kocay', 'Trec4'),
 ('JGD_GL7d', 'GL7d10'),
 ('LPnetlib', 'lpi_galenet'),
 ('JGD_Homology', 'ch3-3-b2'),
 ('vanHeukelum', 'cage3'),
 ('Grund', 'b1_ss'),
 ('Pajek', 'GD98_a'),
 ('JGD_Homology', 'klein-b2'),
 ('LPnetlib', 'lpi_bgprtr'),
 ('JGD_Homology', 'n3c4-b4'),
 ('Pajek', 'Tina_DisCal'),
 ('Meszaros', 'problem'),
 ('JGD_Kocay', 'Trec6'),
 ('Pajek', 'Stranke94'),
 ('Pajek', 'GD96_c'),
 ('Pajek', 'GD97_a'),
 ('JGD_Margulies', 'flower_4_1'),
 ('Grund', 'd_dyn'),
 ('Bai', 'bfwb62'),
 ('Bai', 'bfwa62'),
 ('Pajek', 'GD95_c'),
 ('JGD_BIBD', 'bibd_9_3'),
 ('LPnetlib', 'lpi_forest6'),
 ('LPnetlib', 'lp_kb2'),
 ('Pajek', 'CSphd'),
 ('Newman', 'football'),
 ('JGD_Homology', 'n3c5-b5'),
 ('JGD_Forest', 'TF11'),
 ('Meszaros', 'model1'),
 ('Sandia', 'oscil_dcop_36'),
 ('LPnetlib', 'lp_bore3d'),
 ('Bai', 'qh768'),
 ('JGD_Homology', 'cis-n4c6-b15'),
 ('Meszaros', 'p0548'),
 ('JGD_Homology', 'ch7-6-b1'),
 ('JGD_Homology', 'mk11-b1'),
 ('LPnetlib', 'lp_gfrd_pnc'),
 ('Bai', 'olm500'),
 ('JGD_CAG', 'CAG_mat72'),
 ('JGD_BIBD', 'bibd_15_3'),
 ('Pajek', 'Cities'),
 ('LPnetlib', 'lp_etamacro'),
 ('Bai', 'pde225'),
 ('Gset', 'G52'),
 ('Newman', 'power'),
 ('JGD_GL6', 'GL6_D_8'),
 ('JGD_SPG', 'EX2'),
 ('Sandia', 'adder_dcop_06'),
 ('AG-Monien', 'stufe'),
 ('AG-Monien', 'grid2'),
 ('Sandia', 'adder_trans_01'),
 ('Grund', 'b2_ss'),
 ('Arenas', 'celegans_metabolic'),
 ('AG-Monien', 'L'),
 ('LPnetlib', 'lp_czprob'),
 ('AG-Monien', 'diag'),
 ('Bai', 'bfwa782'),
 ('LPnetlib', 'lp_25fv47'),
 ('Meszaros', 'nsic'),
 ('LPnetlib', 'lp_agg3'),
 ('Sandia', 'fpga_dcop_08'),
 ('Rommes', 'S80PI_n1'),
 ('Schulthess', 'N_biocarta'),
 ('VDOL', 'lowThrust_1'),
 ('MathWorks', 'Pd'),
 ('AG-Monien', 'ukerbe1_dual'),
 ('VDOL', 'spaceStation_3'),
 ('LPnetlib', 'lp_maros'),
 ('LPnetlib', 'lp_fit1p'),
 ('VDOL', 'tumorAntiAngiogenesis_8'),
 ('JGD_Trefethen', 'Trefethen_700'),
 ('Meszaros', 'cep1'),
 ('Bai', 'pde900'),
 ('Meszaros', 'l9'),
 ('LPnetlib', 'lpi_klein3'),
 ('Bates', 'Chem97ZtZ'),
 ('AG-Monien', '3elt_dual'),
 ('Gset', 'G26'),
 ('Meszaros', 'deter8'),
 ('Pajek', 'Zewail'),
 ('Meszaros', 'deter1'),
 ('SNAP', 'p2p-Gnutella08'),
 ('Gset', 'G3'),
 ('JGD_Homology', 'n3c6-b8'),
 ('Newman', 'hep-th'),
 ('JGD_Margulies', 'cat_ears_3_4'),
 ('LPnetlib', 'lp_d6cube'),
 ('Qaplib', 'lp_nug12'),
 ('Meszaros', 'co5'),
 ('Bomhof', 'circuit_2'),
 ('Rajat', 'rajat03'),
 ('Meszaros', 'air02'),
 ('Rommes', 'bips07_1693'),
 ('Hamrle', 'Hamrle2'),
 ('Grund', 'bayer07'),
 ('Rajat', 'rajat13'),
 ('FIDAP', 'ex14'),
 ('Rommes', 'bips98_1142'),
 ('LPnetlib', 'lp_degen3'),
 ('Meszaros', 'aa5'),
 ('FIDAP', 'ex4'),
 ('Meszaros', 'aa4'),
 ('Grund', 'meg1'),
 ('DRIVCAV', 'cavity08'),
 ('Zitney', 'rdist3a'),
 ('Boeing', 'bcsstk34'),
 ('VDOL', 'freeFlyingRobot_12'),
 ('DRIVCAV', 'cavity13'),
 ('VDOL', 'spaceShuttleEntry_2'),
 ('Shyy', 'shyy41'),
 ('FIDAP', 'ex7'),
 ('VDOL', 'dynamicSoaringProblem_8'),
 ('Mittelmann', 'fome13'),
 ('Oberwolfach', 'chipcool0'),
 ('Gset', 'G64'),
 ('DIMACS10', 'vsp_sctap1-2b_and_seymourl'),
 ('SNAP', 'wiki-Vote'),
 ('Priebel', '208bit'),
 ('SNAP', 'p2p-Gnutella31'),
 ('Sandia', 'mult_dcop_03'),
 ('Hollinger', 'mark3jac120sc'),
 ('Newman', 'cond-mat'),
 ('Qaplib', 'lp_nug20'),
 ('Pothen', 'tandem_vtx'),
 ('Meszaros', 'cq9'),
 ('Meszaros', 'kl02'),
 ('Hamm', 'memplus'),
 ('Hollinger', 'jan99jac060sc'),
 ('Schenk_IBMSDS', '3D_28984_Tetra'),
 ('Meszaros', 'rlfprim'),
 ('Rommes', 'juba40k'),
 ('FIDAP', 'ex31'),
 ('Hollinger', 'g7jac060'),
 ('FIDAP', 'ex19'),
 ('VDOL', 'lowThrust_12'),
 ('DIMACS10', 'vsp_c-30_data_data'),
 ('Boeing', 'crystm01'),
 ('DRIVCAV', 'cavity23'),
 ('DIMACS10', 'fe_sphere'),
 ('Mallya', 'lhr17c'),
 ('HVDC', 'hvdc1'),
 ('Meszaros', 'rlfdual'),
 ('FIDAP', 'ex9'),
 ('Cylshell', 's3rmt3m3'),
 ('Zhao', 'Zhao2'),
 ('Simon', 'raefsky2'),
 ('Meszaros', 'r05'),
 ('Meszaros', 'rat'),
 ('DIMACS10', 'al2010'),
 ('Pajek', 'patents_main'),
 ('DIMACS10', 'md2010'),
 ('GHS_indef', 'sparsine'),
 ('JGD_Homology', 'ch7-8-b5'),
 ('Mittelmann', 'fome21'),
 ('DIMACS10', 'wi2010'),
 ('FEMLAB', 'ns3Da'),
 ('JGD_Homology', 'shar_te2-b3'),
 ('LeGresley', 'LeGresley_87936'),
 ('DIMACS10', 'delaunay_n17'),
 ('DIMACS10', 'fe_rotor'),
 ('GHS_indef', 'olesnik0'),
 ('Rajat', 'rajat20'),
 ('ATandT', 'twotone'),
 ('VanVelzen', 'Zd_Jac6_db'),
 ('Li', 'li'),
 ('NYPA', 'Maragal_6'),
 ('Hollinger', 'g7jac160'),
 ('Schenk_IBMNA', 'c-67'),
 ('SNAP', 'web-NotreDame'),
 ('Meszaros', 'nemsemm1'),
 ('Meszaros', 'stat96v4'),
 ('HVDC', 'hvdc2'),
 ('GHS_psdef', 'gridgena'),
 ('TSOPF', 'TSOPF_RS_b162_c4'),
 ('TSOPF', 'TSOPF_FS_b162_c1'),
 ('DIMACS10', 'fe_ocean'),
 ('QLi', 'crashbasis'),
 ('Nemeth', 'nemeth11'),
 ('Oberwolfach', 'gas_sensor'),
 ('Lourakis', 'bundle1'),
 ('Oberwolfach', 'gyro_k'),
 ('Pajek', 'IMDB'),
 ('FEMLAB', 'sme3Dc'),
 ('SNAP', 'roadNet-PA'),
 ('SNAP', 'amazon0312'),
 ('DIMACS10', 'delaunay_n19'),
 ('Rajat', 'rajat30'),
 ('Schenk_ISEI', 'barrier2-2'),
 ('Rajat', 'rajat29'),
 ('Hamrle', 'Hamrle3'),
 ('UTEP', 'Dubcova3'),
 ('Rucci', 'Rucci1'),
 ('TSOPF', 'TSOPF_FS_b39_c30'),
 ('Bova', 'rma10'),
 ('Williams', 'cant'),
 ('ND', 'nd6k'),
 ('Rothberg', 'cfd2'),
 ('GHS_psdef', 'apache2'),
 # New additions, larger
 ('PARSEC','SiO2'),
 ('Mazaheri','bundle_adj'),
 ('DIMACS10', 'kron_g500-logn17'),
 ('Andrianov', 'mip1'),
 ('Meszaros', 'tp-6'),
 ('Mittelmann', 'rail4284'),
 ('Kim', 'kim2'),
 ('DNVS', 'fullb'),
 ('Sinclair', '3Dspectralwave2'),
 ('DIMACS10', 'delaunay_n21'),
 ('TSOPF', 'TSOPF_FS_b300_c3'),
 ('DIMACS10', 'rgg_n_2_20_s0'),
 ('Harvard_Seismology', 'JP'),
 ('Bodendiek', 'CurlCurl_3'),
 ('DIMACS10', 'italy_osm'),
 ('GHS_psdef', 'crankseg_2'),
 ('JGD_GL7d', 'GL7d16'),
 ('Freescale', 'circuit5M_dc'),
 ('Chen', 'pkustk14'),
 ('DIMACS10', 'venturiLevel3'),
 ('TSOPF', 'TSOPF_RS_b2383'),
 ('Freescale', 'Freescale1'),
 ('LAW', 'in-2004'),
 ('JGD_GL7d', 'GL7d21'),
 ('Belcastro', 'human_gene2'),
 ('Schenk_AFE', 'af_shell6'),
 ('Schenk_AFE', 'af_3_k101'),
 ('DIMACS10', 'hugetric-00010'),
 ('LAW', 'eu-2005'),
 ('INPRO', 'msdoor'),
]

def execute_all_tests( data_type, cache_mode, execution_threads, csr5_exec=False, macveth_exec=False, output_file="/dev/null" ):

    for g, mat in selected_tests:
        os.system( f"echo [SCRIPT] Running test {g}/{mat}|tee -a {output_file}" )
        os.system( f"./uzp_spmv.sh {g} {mat} {data_type} {cache_mode} {execution_threads}|tee -a {output_file}" )
        os.system( f"./mkl_spmv.sh {g} {mat} {data_type} {cache_mode} {execution_threads}|tee -a {output_file}" )
        if macveth_exec: os.system( f"./macveth_spmv.sh {g} {mat} {data_type} {cache_mode} {execution_threads}|tee -a {output_file}" )
        if csr5_exec: os.system( f"./csr5_spmv.sh {g} {mat} {data_type} {cache_mode} {execution_threads}|tee -a {output_file}" )
        os.system( f"./csr_spmv.sh {g} {mat} {data_type} {cache_mode} {execution_threads}|tee -a {output_file}" )

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print( "Usage: ./run_all.py [float|double] [cold|hot] [1th|2th|8th]" )
        sys.exit(1)

    csr5_exec = True
    macveth_exec = True

    # Validate data type
    data_type = sys.argv[1]
    match data_type:
        case "float": csr5_exec = False
        case "double": macveth_exec = False
        case _:
            print( f"[ERROR] Invalid data type: {data_type}. Must be 'float' or 'double'." )
            sys.exit(1)

    # Validate cache mode
    cache_mode = sys.argv[2]
    match cache_mode:
        case "cold": pass
        case "hot": pass
        case _:
            print( f"[ERROR] Invalid cache mode: {cache_mode}. Must be 'cold' or 'hot'." )
            sys.exit(1)

    # Validate number of threads
    execution_threads = sys.argv[3]
    match execution_threads:
        case "1th": pass
        case "2th": macveth_exec = False
        case "8th": macveth_exec = False
        case _:
            print( f"[ERROR] Invalid number of threads: {execution_threads}. Must be '1th', '2th', or '8th'." )
            sys.exit(1)

    output_file = f"results_run_all_{data_type}_{cache_mode}_{execution_threads}_{int(time.time())}.out"
    print( f"[SCRIPT] Output logged to {output_file}." )
    execute_all_tests( data_type, cache_mode, execution_threads, csr5_exec, macveth_exec, output_file )

    # Parse results and generate DataFrame
    df = parse_results( output_file )
    df['nthreads'] = int( execution_threads[0] )
    df.to_csv( output_file.replace( ".out", ".csv" ) )
