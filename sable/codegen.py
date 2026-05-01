from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OutOfLineCode:
    body: str
    name: str | None = None
    parameters: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()


def out_of_line(
    body: str,
    name: str | None = None,
    parameters: Sequence[str] | None = None,
    arguments: Sequence[object] | None = None,
) -> OutOfLineCode:
    parameters = tuple(parameters or ())
    arguments = tuple(str(argument) for argument in (arguments or ()))
    if len(parameters) != len(arguments):
        raise ValueError("out_of_line parameters and arguments must have the same length")
    return OutOfLineCode(body=body, name=name, parameters=parameters, arguments=arguments)
