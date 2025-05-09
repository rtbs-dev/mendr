#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x851f1c5b

# Compiled with Coconut version 3.1.2

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys
import os as _coconut_os
_coconut_header_info = ('3.1.2', '', False)
_coconut_cached__coconut__ = _coconut_sys.modules.get(str('__coconut__'))
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
_coconut_pop_path = False
if _coconut_cached__coconut__ is None or getattr(_coconut_cached__coconut__, "_coconut_header_info", None) != _coconut_header_info and _coconut_os.path.dirname(_coconut_cached__coconut__.__file__ or "") != _coconut_file_dir:  # type: ignore
    if _coconut_cached__coconut__ is not None:
        _coconut_sys.modules[str('_coconut_cached__coconut__')] = _coconut_cached__coconut__
        del _coconut_sys.modules[str('__coconut__')]
    _coconut_sys.path.insert(0, _coconut_file_dir)
    _coconut_pop_path = True
    _coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
    if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):  # type: ignore
        _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")  # type: ignore
        import __coconut__ as _coconut__coconut__
        _coconut__coconut__.__name__ = _coconut_full_module_name
        for _coconut_v in vars(_coconut__coconut__).values():  # type: ignore
            if getattr(_coconut_v, "__module__", None) == str('__coconut__'):  # type: ignore
                try:
                    _coconut_v.__module__ = _coconut_full_module_name
                except AttributeError:
                    _coconut_v_type = type(_coconut_v)  # type: ignore
                    if getattr(_coconut_v_type, "__module__", None) == str('__coconut__'):  # type: ignore
                        _coconut_v_type.__module__ = _coconut_full_module_name
        _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _coconut_handle_cls_kwargs, _coconut_handle_cls_stargs, _namedtuple_of, _coconut, _coconut_Expected, _coconut_MatchError, _coconut_SupportsAdd, _coconut_SupportsMinus, _coconut_SupportsMul, _coconut_SupportsPow, _coconut_SupportsTruediv, _coconut_SupportsFloordiv, _coconut_SupportsMod, _coconut_SupportsAnd, _coconut_SupportsXor, _coconut_SupportsOr, _coconut_SupportsLshift, _coconut_SupportsRshift, _coconut_SupportsMatmul, _coconut_SupportsInv, _coconut_iter_getitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_complex_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_raise, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec, _coconut_comma_op, _coconut_arr_concat_op, _coconut_mk_anon_namedtuple, _coconut_matmul, _coconut_py_str, _coconut_flatten, _coconut_multiset, _coconut_back_none_pipe, _coconut_back_none_star_pipe, _coconut_back_none_dubstar_pipe, _coconut_forward_none_compose, _coconut_back_none_compose, _coconut_forward_none_star_compose, _coconut_back_none_star_compose, _coconut_forward_none_dubstar_compose, _coconut_back_none_dubstar_compose, _coconut_call_or_coefficient, _coconut_in, _coconut_not_in, _coconut_attritemgetter, _coconut_if_op, _coconut_CoconutWarning
if _coconut_pop_path:
    _coconut_sys.path.pop(0)
try:
    __file__ = _coconut_os.path.abspath(__file__) if __file__ else __file__
except NameError:
    pass
else:
    if __file__ and str('__coconut_cache__') in __file__:
        _coconut_file_comps = []
        while __file__:
            __file__, _coconut_file_comp = _coconut_os.path.split(__file__)
            if not _coconut_file_comp:
                _coconut_file_comps.append(__file__)
                break
            if _coconut_file_comp != str('__coconut_cache__'):
                _coconut_file_comps.append(_coconut_file_comp)
        __file__ = _coconut_os.path.join(*reversed(_coconut_file_comps))

# Compiled Coconut: -----------------------------------------------------------

import numpy as np  #1 (line in Coconut source)
# from scipy import sparse
import sparse  #3 (line in Coconut source)
from jaxtyping import Int  #4 (line in Coconut source)
from jaxtyping import jaxtyped  #4 (line in Coconut source)
from beartype import beartype  #5 (line in Coconut source)

RW_Array = Int[np.ndarray, 'n_walks n_jumps']  # type: _coconut.typing.TypeAlias  #7 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #7 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #7 (line in Coconut source)
__annotations__["RW_Array"] = _coconut.typing.TypeAlias  #7 (line in Coconut source)
OneHot_SPArray = Int[sparse.COO, 'n_walks n_nodes']  # type: _coconut.typing.TypeAlias  #8 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #8 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #8 (line in Coconut source)
__annotations__["OneHot_SPArray"] = _coconut.typing.TypeAlias  #8 (line in Coconut source)


@jaxtyped(typechecker=beartype)  #11 (line in Coconut source)
def rw_jumps_to_coords(rw,  # type: RW_Array  #12 (line in Coconut source)
    num_nodes=None  # type: _coconut.typing.Union[int, None]  #12 (line in Coconut source)
    ):  #12 (line in Coconut source)
# type: (...) -> OneHot_SPArray

    num_walks = rw.shape[0]  #14 (line in Coconut source)
    num_nodes = np.max(rw) + 1 if num_nodes is None else num_nodes  # need at least enough nodes for biggest node index  #15 (line in Coconut source)

# trick to one-hot encode occurrences from walks
    row_idx = np.repeat(np.arange(num_walks)[:, None], rw.shape[1], 1).flatten()  #18 (line in Coconut source)

    col_idx = rw.flatten()  #24 (line in Coconut source)

    idx = np.unique(np.array([row_idx, col_idx]), axis=1)  # no duplicates  #26 (line in Coconut source)

    return sparse.COO(idx, data=1, shape=(num_walks, num_nodes))  #28 (line in Coconut source)

# assert G.mat.shape[0]>=col_idx.max(), f'{G.mat.shape} A is bad for col pointer {col_idx.max()}'
# X = sparse.coo_matrix(
#     (np.ones_like(idx[0]), (idx[0], idx[1])),
#     shape=(rw.shape[0], n)
# ).astype(bool)
