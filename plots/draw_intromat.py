import scipy
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300

def save_mtx(mtx, mtx_name, rpntr, cpntr):
    fig, ax = plt.subplots()
    ax.spy(mtx, markersize = 0.5)

    for pos in rpntr:
        ax.axhline(pos, color="#d55e00", linewidth=0.9)
    ax.set_yticks(rpntr)

    for pos in cpntr:
        ax.axvline(pos, color="#d55e00", linewidth=0.9)
    ax.set_xticks(cpntr)

    # fig.set_size_inches((12,12))
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()
    plt.savefig(f"images/{mtx_name[:-4]}.pdf")

def save_dia_mtx(mtx, mtx_name, bands, window):
    # A (V)DIA region is a set of diagonals at distinct offsets.  Each band is
    # given as (off_lo, off_hi): the diagonal stripe covering columns
    # i + off_lo .. i + off_hi for row i.  We demarcate each band by tracing its
    # two diagonal edges (no horizontal closers).  `window` = (lo, hi) is the
    # row/column range we zoom into so the individual diagonals are large enough
    # to tell apart, while sparse off-band entries remain visible as the CSR
    # fallback.
    lo, hi = window
    fig, ax = plt.subplots()
    ax.spy(mtx, markersize = 1.5)

    r0, re = lo, hi - 1
    for (off_lo, off_hi) in bands:
        ax.plot([r0 + off_lo, re + off_lo], [r0, re], color="#d55e00", linewidth=1.0)
        ax.plot([r0 + off_hi, re + off_hi], [r0, re], color="#d55e00", linewidth=1.0)

    ax.set_xlim(lo, hi)
    ax.set_ylim(hi, lo)
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()
    plt.savefig(f"images/{mtx_name[:-4]}.pdf")

if __name__ == "__main__":
    d = {}

    d["bcspwr06.mtx"] = [
        [0, 208, 288, 340, 392, 445, 500, 576, 840, 995, 1032, 1088, 1118, 1154, 1236, 1296, 1340, 1356, 1424, 1454],
        [0, 208, 288, 340, 392, 445, 500, 576, 840, 995, 1032, 1088, 1118, 1154, 1236, 1296, 1340, 1356, 1424, 1454]
    ]

    for mtx_name, (rpntr, cpntr) in d.items():
        mtx = scipy.io.mmread(mtx_name)
        save_mtx(mtx, mtx_name, rpntr, cpntr)

    # VDIA + CSR example (ex18 from SuiteSparse): three diagonals at distinct
    # offsets (the main diagonal and a pair of off-diagonals at +-78 within the
    # top-left corner shown) are captured by VDIA, while the sparse entries
    # outside the diagonals are handled by the CSR fallback.  We zoom into the
    # top-left corner so the individual diagonals are large enough to
    # distinguish.  Each band is (off_lo, off_hi); `window` is the (row, col)
    # range shown.
    dia = {}
    dia["ex18.mtx"] = (
        [
            (-82, -74),
            (-5, 5),
            (74, 82),
        ],
        (0, 250),  # window
    )
    for mtx_name, (bands, window) in dia.items():
        mtx = scipy.io.mmread(mtx_name)
        save_dia_mtx(mtx, mtx_name, bands, window)
