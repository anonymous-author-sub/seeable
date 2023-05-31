"""utils for parsing omegaconf """
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import DictConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf


def omegaconf_to_dict(x: Any) -> Dict:
    if isinstance(x, DictConfig):
        out = OmegaConf.to_container(x)
        assert isinstance(out, dict), out
        return out
    elif isinstance(x, dict):
        return x


def parse_opcode(op: str) -> str:
    """parse opcode from string"""
    OPS = {
        "==": "eq",
        "!=": "ne",
        ">": "gt",
        "<": "lt",
        ">=": "ge",
        "<=": "le",
        "in": "isin",
    }
    assert isinstance(op, str), f"opcode must be a string, got {type(op)}"
    op = op.strip().lower()
    op = OPS[op] if op not in OPS.values() else op
    return op


def parse_selector(selector: Optional[Dict]) -> Callable:
    """parse the dict to generate a selection function
    for the moment handle only dict of the type
    {"c1": "v1", "c2": ("v2", "op")}
    Output: lambda x: reduce([x["c1"] == "v1", x["c2"] op "v2"])
    TODO: handle value being another col

    Return: Callable that takes a dataframe and return a bool mask array
    """
    if not selector or not isinstance(selector, dict):
        return lambda df: np.ones(len(df), dtype=bool)  # return all
    conditions = []
    for col, value in selector.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            assert len(value) == 2, f"invalid value {value}"
            value, op = value
        else:
            if isinstance(value, (list, tuple)):
                op = "in"
            else:
                op = "eq"

        conditions.append((col, value, parse_opcode(op)))
    # compose functions
    reductor = np.logical_and.reduce
    F = lambda df: reductor(
        [getattr(df[left], op)(right) for (left, right, op) in conditions]
    )
    return F
