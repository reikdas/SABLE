import pathlib
import re
import sys

import numpy
import scipy.sparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sable import Matrix, Plan
from sable.extractors import CSRConvertor
from sable.kernels import GradientDescent, NaiveCSRSpmv
from sable.tensor import DenseInput


def _write_vector(path, values):
    path.write_text(",".join(f"{v:.17g}" for v in values) + "\n")


def test_iterate_codegen_wraps_dispatches_in_while_loop(tmp_path):
    """The update kernel's loop hooks (state/init/condition) drive the loop."""
    rhs_path = tmp_path / "x.vector"
    _write_vector(rhs_path, [1.0, 1.0])
    A = scipy.sparse.csr_matrix(numpy.array([[0.5, 0.0], [0.0, 0.5]]))

    plan = Plan(Matrix(A, name="iterate_codegen"), artifact_dir=str(tmp_path))
    plan.rhs(DenseInput.vector(str(rhs_path), size=2))
    class QuarterDelta(GradientDescent):
        DELTA = 0.25

    plan.dispatch(plan.extract(CSRConvertor()), NaiveCSRSpmv())
    plan.dispatch(plan.accumulate(), QuarterDelta([0.5, 0.5], alpha=0.5))

    source = pathlib.Path(plan.compile_loop(iters=7, filename="iterate_codegen", bench=2).c_path).read_text()
    assert "double sable_rn;" in source
    assert "sable_rn = DBL_MAX;" in source
    assert "while (sable_iteration < 7 && (sable_rn >= sable_b_norm * 0.25 * 0.25)) {" in source
    assert "memcpy(x, sable_x0, 2 * sizeof(double));" in source
    assert "read_dense_input(rhs_file, sable_x0, 2);" in source
    # Times accumulate across the iterations of one benchmark repetition.
    assert re.search(r"dispatch_part_times\[0\]\[iter\] \+=", source)
    assert 'printf("Iterations: ");' in source


def test_iterate_codegen_fixed_count_without_condition_kernel(tmp_path):
    """iters alone gives a fixed-count loop."""
    rhs_path = tmp_path / "x.vector"
    _write_vector(rhs_path, [1.0, 1.0])

    plan = Plan(Matrix(scipy.sparse.eye(2, format="csr"), name="fixed_count"), artifact_dir=str(tmp_path))
    plan.rhs(DenseInput.vector(str(rhs_path), size=2))
    plan.dispatch(plan.extract(CSRConvertor()), NaiveCSRSpmv())

    source = pathlib.Path(plan.compile_loop(iters=7, filename="fixed_count", bench=1).c_path).read_text()
    assert "while (sable_iteration < 7) {" in source
