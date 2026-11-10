from __future__ import annotations

from sable.formats import Rep


class GradientDescent:
    DELTA = 0.01

    def __init__(self, b, alpha: float):
        if self.DELTA <= 0:
            raise ValueError(f"DELTA must be positive, got {self.DELTA}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = float(alpha)
        if hasattr(b, "ravel"):
            b = b.ravel().tolist()
        b = [float(v) for v in b]
        if not b:
            raise ValueError("b must not be empty")
        self.b = Rep(b, label="gd_b")

    def emit_includes(self) -> list[str]:
        return ["#include <float.h>"]

    def emit_loop_state(self) -> str:
        return "    double sable_rn;\n"

    def emit_loop_init(self) -> str:
        return "        sable_rn = DBL_MAX;\n"

    def emit_loop_condition(self) -> str:
        return f"sable_rn >= sable_b_norm * {self.DELTA:.17g} * {self.DELTA:.17g}"

    def emit_setup(self, fmt, rhs) -> str:
        n = len(self.b.values)
        b = self.b
        return f"""\
double sable_b_norm = 0.0;
for (long i = 0; i < {n}; i++) {{
    sable_b_norm += {b}[i] * {b}[i];
}}
"""

    def emit_call(self, fmt, y: str, x: str, rhs) -> str:
        n = len(self.b.values)
        b = self.b
        return f"""\
sable_rn = 0.0;
for (long i = 0; i < {n}; i++) {{
    double r = {y}[i] - {b}[i];
    sable_rn += r * r;
    {x}[i] = {x}[i] - {self.alpha:.17g} * r;
    {y}[i] = {x}[i];
}}
"""
