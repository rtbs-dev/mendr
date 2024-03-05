#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x80e1e5ce

# Compiled with Coconut version 3.0.4

"""
Inspired by BinSparse specification
"""

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys
import os as _coconut_os
_coconut_header_info = ('3.0.4', '', False)
_coconut_cached__coconut__ = _coconut_sys.modules.get(str('__coconut__'))
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
_coconut_pop_path = False
if _coconut_cached__coconut__ is None or getattr(_coconut_cached__coconut__, "_coconut_header_info", None) != _coconut_header_info and _coconut_os.path.dirname(_coconut_cached__coconut__.__file__ or "") != _coconut_file_dir:
    if _coconut_cached__coconut__ is not None:
        _coconut_sys.modules[str('_coconut_cached__coconut__')] = _coconut_cached__coconut__
        del _coconut_sys.modules[str('__coconut__')]
    _coconut_sys.path.insert(0, _coconut_file_dir)
    _coconut_pop_path = True
    _coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
    if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):
        _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")
        import __coconut__ as _coconut__coconut__
        _coconut__coconut__.__name__ = _coconut_full_module_name
        for _coconut_v in vars(_coconut__coconut__).values():
            if getattr(_coconut_v, "__module__", None) == str('__coconut__'):
                try:
                    _coconut_v.__module__ = _coconut_full_module_name
                except AttributeError:
                    _coconut_v_type = type(_coconut_v)
                    if getattr(_coconut_v_type, "__module__", None) == str('__coconut__'):
                        _coconut_v_type.__module__ = _coconut_full_module_name
        _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _coconut_handle_cls_kwargs, _coconut_handle_cls_stargs, _namedtuple_of, _coconut, _coconut_Expected, _coconut_MatchError, _coconut_SupportsAdd, _coconut_SupportsMinus, _coconut_SupportsMul, _coconut_SupportsPow, _coconut_SupportsTruediv, _coconut_SupportsFloordiv, _coconut_SupportsMod, _coconut_SupportsAnd, _coconut_SupportsXor, _coconut_SupportsOr, _coconut_SupportsLshift, _coconut_SupportsRshift, _coconut_SupportsMatmul, _coconut_SupportsInv, _coconut_iter_getitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_complex_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_raise, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec, _coconut_comma_op, _coconut_multi_dim_arr, _coconut_mk_anon_namedtuple, _coconut_matmul, _coconut_py_str, _coconut_flatten, _coconut_multiset, _coconut_back_none_pipe, _coconut_back_none_star_pipe, _coconut_back_none_dubstar_pipe, _coconut_forward_none_compose, _coconut_back_none_compose, _coconut_forward_none_star_compose, _coconut_back_none_star_compose, _coconut_forward_none_dubstar_compose, _coconut_back_none_dubstar_compose, _coconut_call_or_coefficient, _coconut_in, _coconut_not_in, _coconut_attritemgetter
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


from serde import serde  #4 (line in Coconut source)
from serde import AdjacentTagging  #4 (line in Coconut source)
from dataclasses import dataclass  #5 (line in Coconut source)
import numpy as np  #6 (line in Coconut source)
import numpy.typing as npt  #7 (line in Coconut source)
import networkx as nx  #8 (line in Coconut source)
import sparse  #9 (line in Coconut source)
import csrgraph as cg  #10 (line in Coconut source)
from beartype.door import is_bearable  #11 (line in Coconut source)
try:  #12 (line in Coconut source)
    _coconut_sys_0 = sys  # type: ignore  #12 (line in Coconut source)
except _coconut.NameError:  #12 (line in Coconut source)
    _coconut_sys_0 = _coconut_sentinel  #12 (line in Coconut source)
sys = _coconut_sys  #12 (line in Coconut source)
if sys.version_info >= (3, 6):  #12 (line in Coconut source)
    if _coconut.typing.TYPE_CHECKING:  #12 (line in Coconut source)
        from typing import Type  #12 (line in Coconut source)
    else:  #12 (line in Coconut source)
        try:  #12 (line in Coconut source)
            Type = _coconut.typing.Type  #12 (line in Coconut source)
        except _coconut.AttributeError as _coconut_imp_err:  #12 (line in Coconut source)
            raise _coconut.ImportError(_coconut.str(_coconut_imp_err))  #12 (line in Coconut source)
else:  #12 (line in Coconut source)
    from typing_extensions import Type  #12 (line in Coconut source)
if _coconut_sys_0 is not _coconut_sentinel:  #12 (line in Coconut source)
    sys = _coconut_sys_0  #12 (line in Coconut source)
from jaxtyping import Int  #13 (line in Coconut source)
from jaxtyping import Shaped  #13 (line in Coconut source)
from numbers import Number  #14 (line in Coconut source)
from scipy.sparse import spmatrix  #15 (line in Coconut source)
from scipy.sparse import sparray  #15 (line in Coconut source)
from scipy.sparse import coo_matrix  #15 (line in Coconut source)
# import

# @dataclass
# class BSFmt:
#     version:str = "0.1"

# @dataclass
# class SparseFMT:
#     shape: (int; int)

SparseValues = Shaped[np.ndarray, "#nnz"]  # type: _coconut.typing.TypeAlias  # could be scalar  #26 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  # could be scalar  #26 (line in Coconut source)
    __annotations__ = {}  # type: ignore  # could be scalar  #26 (line in Coconut source)
__annotations__["SparseValues"] = _coconut.typing.TypeAlias  # could be scalar  #26 (line in Coconut source)

@serde  #28 (line in Coconut source)
@dataclass  #29 (line in Coconut source)
class COO(_coconut.object):  #30 (line in Coconut source)
    indices_0 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #31 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #31 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #31 (line in Coconut source)
    __annotations__["indices_0"] = npt.NDArray[int]  #31 (line in Coconut source)
    indices_1 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #32 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #32 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #32 (line in Coconut source)
    __annotations__["indices_1"] = npt.NDArray[int]  #32 (line in Coconut source)
    values = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: int  #| np.number  #33 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #| np.number  #33 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #| np.number  #33 (line in Coconut source)
    __annotations__["values"] = int  #| np.number  #33 (line in Coconut source)

_coconut_call_set_names(COO)  #35 (line in Coconut source)
@serde  #35 (line in Coconut source)
@dataclass  #36 (line in Coconut source)
class CSC(_coconut.object):  #37 (line in Coconut source)
    pointers_to_1 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #38 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #38 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #38 (line in Coconut source)
    __annotations__["pointers_to_1"] = npt.NDArray[int]  #38 (line in Coconut source)
    indices_1 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #39 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #39 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #39 (line in Coconut source)
    __annotations__["indices_1"] = npt.NDArray[int]  #39 (line in Coconut source)
    values = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: int  #| np.number  #40 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #| np.number  #40 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #| np.number  #40 (line in Coconut source)
    __annotations__["values"] = int  #| np.number  #40 (line in Coconut source)


_coconut_call_set_names(CSC)  #43 (line in Coconut source)
SparseArrayType = (_coconut.typing.Union[COO, CSC])  # type: _coconut.typing.TypeAlias  #43 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #43 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #43 (line in Coconut source)
__annotations__["SparseArrayType"] = _coconut.typing.TypeAlias  #43 (line in Coconut source)

@serde(tagging=AdjacentTagging("format", "data_types"))  #48 (line in Coconut source)
@dataclass  #49 (line in Coconut source)
class SerialSparse(_coconut.object):  #50 (line in Coconut source)
# version:str = "0.1"
    shape = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: tuple[int, int]  #52 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #52 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #52 (line in Coconut source)
    __annotations__["shape"] = tuple[int, int]  #52 (line in Coconut source)
    array = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: SparseArrayType  #53 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #53 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #53 (line in Coconut source)
    __annotations__["array"] = SparseArrayType  #53 (line in Coconut source)


    _coconut_typevar_T_0 = _coconut.typing.TypeVar("_coconut_typevar_T_0")  #56 (line in Coconut source)

    @classmethod  #56 (line in Coconut source)
    def from_array(cls,  # type: Type[_coconut_typevar_T_0]  #57 (line in Coconut source)
        a):  #57 (line in Coconut source)
# type: (...) -> _coconut_typevar_T_0
        return _from_sparse(a)  #58 (line in Coconut source)



    _coconut_typevar_T_1 = _coconut.typing.TypeVar("_coconut_typevar_T_1")  #61 (line in Coconut source)

    def to_array(self):  #61 (line in Coconut source)
# type: (...) -> _coconut_typevar_T_1
        return _to_sparse(self.shape, self.array)  #62 (line in Coconut source)


# @addpattern

_coconut_call_set_names(SerialSparse)  #66 (line in Coconut source)
@_coconut_mark_as_match  #66 (line in Coconut source)
def _from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #66 (line in Coconut source)
    _coconut_match_check_0 = False  #66 (line in Coconut source)
    _coconut_match_set_name_a = _coconut_sentinel  #66 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #66 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #66 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #66 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "a" in _coconut_match_kwargs)) == 1):  #66 (line in Coconut source)
        _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("a")  #66 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_0, sparse.COO):  #66 (line in Coconut source)
            _coconut_match_set_name_a = _coconut_match_temp_0  #66 (line in Coconut source)
            if not _coconut_match_kwargs:  #66 (line in Coconut source)
                _coconut_match_check_0 = True  #66 (line in Coconut source)
    if _coconut_match_check_0:  #66 (line in Coconut source)
        if _coconut_match_set_name_a is not _coconut_sentinel:  #66 (line in Coconut source)
            a = _coconut_match_set_name_a  #66 (line in Coconut source)
    if not _coconut_match_check_0:  #66 (line in Coconut source)
        raise _coconut_FunctionMatchError('match def _from_sparse(a `is_bearable` sparse.COO):', _coconut_match_args)  #66 (line in Coconut source)

    idx = a.coords  #67 (line in Coconut source)
    values = 1  # ignore a.data for the purposes of this work... for now  #68 (line in Coconut source)
# if np.all(np.isclose(a.data, a.data[0])):
#     values = a.data[0]
# else:
#     values = a.data

    return SerialSparse(a.shape, COO(idx[0], idx[1], values))  #74 (line in Coconut source)


try:  #76 (line in Coconut source)
    _coconut_addpattern_0 = _coconut_addpattern(_from_sparse)  # type: ignore  #76 (line in Coconut source)
except _coconut.NameError:  #76 (line in Coconut source)
    _coconut_addpattern_0 = lambda f: f  #76 (line in Coconut source)
@_coconut_addpattern_0  #76 (line in Coconut source)
@_coconut_mark_as_match  #76 (line in Coconut source)
def _from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #76 (line in Coconut source)
    _coconut_match_check_1 = False  #76 (line in Coconut source)
    _coconut_match_set_name_s = _coconut_sentinel  #76 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #76 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #76 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #76 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "s" in _coconut_match_kwargs)) == 1):  #76 (line in Coconut source)
        _coconut_match_temp_1 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("s")  #76 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_1, spmatrix):  #76 (line in Coconut source)
            _coconut_match_set_name_s = _coconut_match_temp_1  #76 (line in Coconut source)
            if not _coconut_match_kwargs:  #76 (line in Coconut source)
                _coconut_match_check_1 = True  #76 (line in Coconut source)
    if _coconut_match_check_1:  #76 (line in Coconut source)
        if _coconut_match_set_name_s is not _coconut_sentinel:  #76 (line in Coconut source)
            s = _coconut_match_set_name_s  #76 (line in Coconut source)
    if not _coconut_match_check_1:  #76 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def _from_sparse(s `is_bearable` spmatrix) = s |> _from_sparse .. sparse.COO.from_scipy_sparse', _coconut_match_args)  #76 (line in Coconut source)

    return (_coconut_forward_compose(sparse.COO.from_scipy_sparse, _from_sparse))(s)  #76 (line in Coconut source)

try:  #77 (line in Coconut source)
    _coconut_addpattern_1 = _coconut_addpattern(_from_sparse)  # type: ignore  #77 (line in Coconut source)
except _coconut.NameError:  #77 (line in Coconut source)
    _coconut_addpattern_1 = lambda f: f  #77 (line in Coconut source)
@_coconut_addpattern_1  #77 (line in Coconut source)
@_coconut_mark_as_match  #77 (line in Coconut source)
def _from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #77 (line in Coconut source)
    _coconut_match_check_2 = False  #77 (line in Coconut source)
    _coconut_match_set_name_s = _coconut_sentinel  #77 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #77 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #77 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #77 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "s" in _coconut_match_kwargs)) == 1):  #77 (line in Coconut source)
        _coconut_match_temp_2 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("s")  #77 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_2, sparray):  #77 (line in Coconut source)
            _coconut_match_set_name_s = _coconut_match_temp_2  #77 (line in Coconut source)
            if not _coconut_match_kwargs:  #77 (line in Coconut source)
                _coconut_match_check_2 = True  #77 (line in Coconut source)
    if _coconut_match_check_2:  #77 (line in Coconut source)
        if _coconut_match_set_name_s is not _coconut_sentinel:  #77 (line in Coconut source)
            s = _coconut_match_set_name_s  #77 (line in Coconut source)
    if not _coconut_match_check_2:  #77 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def _from_sparse(s `is_bearable` sparray) = s |> _from_sparse .. coo_matrix', _coconut_match_args)  #77 (line in Coconut source)

    return (_coconut_forward_compose(coo_matrix, _from_sparse))(s)  #77 (line in Coconut source)

try:  #78 (line in Coconut source)
    _coconut_addpattern_2 = _coconut_addpattern(_from_sparse)  # type: ignore  #78 (line in Coconut source)
except _coconut.NameError:  #78 (line in Coconut source)
    _coconut_addpattern_2 = lambda f: f  #78 (line in Coconut source)
@_coconut_addpattern_2  #78 (line in Coconut source)
@_coconut_mark_as_match  #78 (line in Coconut source)
def _from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #78 (line in Coconut source)
    _coconut_match_check_3 = False  #78 (line in Coconut source)
    _coconut_match_set_name_g = _coconut_sentinel  #78 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #78 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #78 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #78 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "g" in _coconut_match_kwargs)) == 1):  #78 (line in Coconut source)
        _coconut_match_temp_3 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("g")  #78 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_3, nx.Graph):  #78 (line in Coconut source)
            _coconut_match_set_name_g = _coconut_match_temp_3  #78 (line in Coconut source)
            if not _coconut_match_kwargs:  #78 (line in Coconut source)
                _coconut_match_check_3 = True  #78 (line in Coconut source)
    if _coconut_match_check_3:  #78 (line in Coconut source)
        if _coconut_match_set_name_g is not _coconut_sentinel:  #78 (line in Coconut source)
            g = _coconut_match_set_name_g  #78 (line in Coconut source)
    if not _coconut_match_check_3:  #78 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def _from_sparse(g `is_bearable` nx.Graph) = g |> _from_sparse .. nx.to_scipy_sparse_array', _coconut_match_args)  #78 (line in Coconut source)

    return (_coconut_forward_compose(nx.to_scipy_sparse_array, _from_sparse))(g)  #78 (line in Coconut source)

try:  #79 (line in Coconut source)
    _coconut_addpattern_3 = _coconut_addpattern(_from_sparse)  # type: ignore  #79 (line in Coconut source)
except _coconut.NameError:  #79 (line in Coconut source)
    _coconut_addpattern_3 = lambda f: f  #79 (line in Coconut source)
@_coconut_addpattern_3  #79 (line in Coconut source)
@_coconut_mark_as_match  #79 (line in Coconut source)
def _from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #79 (line in Coconut source)
    _coconut_match_check_4 = False  #79 (line in Coconut source)
    _coconut_match_set_name_g = _coconut_sentinel  #79 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #79 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #79 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #79 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "g" in _coconut_match_kwargs)) == 1):  #79 (line in Coconut source)
        _coconut_match_temp_4 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("g")  #79 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_4, cg.csrgraph):  #79 (line in Coconut source)
            _coconut_match_set_name_g = _coconut_match_temp_4  #79 (line in Coconut source)
            if not _coconut_match_kwargs:  #79 (line in Coconut source)
                _coconut_match_check_4 = True  #79 (line in Coconut source)
    if _coconut_match_check_4:  #79 (line in Coconut source)
        if _coconut_match_set_name_g is not _coconut_sentinel:  #79 (line in Coconut source)
            g = _coconut_match_set_name_g  #79 (line in Coconut source)
    if not _coconut_match_check_4:  #79 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def _from_sparse(g `is_bearable` cg.csrgraph) = _from_sparse(g.mat)', _coconut_match_args)  #79 (line in Coconut source)

    return _from_sparse(g.mat)  #79 (line in Coconut source)



@_coconut_mark_as_match  #82 (line in Coconut source)
def _to_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #82 (line in Coconut source)
    _coconut_match_check_5 = False  #82 (line in Coconut source)
    _coconut_match_set_name_shape = _coconut_sentinel  #82 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #82 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #82 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #82 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) == 2) and ("shape" not in _coconut_match_kwargs):  #82 (line in Coconut source)
        _coconut_match_temp_5 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("shape")  #82 (line in Coconut source)
        _coconut_match_temp_6 = _coconut.getattr(COO, "_coconut_is_data", False) or _coconut.isinstance(COO, _coconut.tuple) and _coconut.all(_coconut.getattr(_coconut_x, "_coconut_is_data", False) for _coconut_x in COO)  # type: ignore  #82 (line in Coconut source)
        _coconut_match_set_name_shape = _coconut_match_temp_5  #82 (line in Coconut source)
        if not _coconut_match_kwargs:  #82 (line in Coconut source)
            _coconut_match_check_5 = True  #82 (line in Coconut source)
    if _coconut_match_check_5:  #82 (line in Coconut source)
        _coconut_match_check_5 = False  #82 (line in Coconut source)
        if not _coconut_match_check_5:  #82 (line in Coconut source)
            _coconut_match_set_name_row = _coconut_sentinel  #82 (line in Coconut source)
            _coconut_match_set_name_col = _coconut_sentinel  #82 (line in Coconut source)
            _coconut_match_set_name_vals = _coconut_sentinel  #82 (line in Coconut source)
            if (_coconut_match_temp_6) and (_coconut.isinstance(_coconut_match_args[1], COO)) and (_coconut.len(_coconut_match_args[1]) >= 3):  #82 (line in Coconut source)
                _coconut_match_set_name_row = _coconut_match_args[1][0]  #82 (line in Coconut source)
                _coconut_match_set_name_col = _coconut_match_args[1][1]  #82 (line in Coconut source)
                _coconut_match_set_name_vals = _coconut_match_args[1][2]  #82 (line in Coconut source)
                _coconut_match_temp_7 = _coconut.len(_coconut_match_args[1]) <= _coconut.max(3, _coconut.len(_coconut_match_args[1].__match_args__)) and _coconut.all(i in _coconut.getattr(_coconut_match_args[1], "_coconut_data_defaults", {}) and _coconut_match_args[1][i] == _coconut.getattr(_coconut_match_args[1], "_coconut_data_defaults", {})[i] for i in _coconut.range(3, _coconut.len(_coconut_match_args[1].__match_args__))) if _coconut.hasattr(_coconut_match_args[1], "__match_args__") else _coconut.len(_coconut_match_args[1]) == 3  # type: ignore  #82 (line in Coconut source)
                if _coconut_match_temp_7:  #82 (line in Coconut source)
                    _coconut_match_check_5 = True  #82 (line in Coconut source)
            if _coconut_match_check_5:  #82 (line in Coconut source)
                if _coconut_match_set_name_row is not _coconut_sentinel:  #82 (line in Coconut source)
                    row = _coconut_match_set_name_row  #82 (line in Coconut source)
                if _coconut_match_set_name_col is not _coconut_sentinel:  #82 (line in Coconut source)
                    col = _coconut_match_set_name_col  #82 (line in Coconut source)
                if _coconut_match_set_name_vals is not _coconut_sentinel:  #82 (line in Coconut source)
                    vals = _coconut_match_set_name_vals  #82 (line in Coconut source)

        if not _coconut_match_check_5:  #82 (line in Coconut source)
            if (not _coconut_match_temp_6) and (_coconut.isinstance(_coconut_match_args[1], COO)):  #82 (line in Coconut source)
                _coconut_match_check_5 = True  #82 (line in Coconut source)
            if _coconut_match_check_5:  #82 (line in Coconut source)
                _coconut_match_check_5 = False  #82 (line in Coconut source)
                if not _coconut_match_check_5:  #82 (line in Coconut source)
                    if _coconut.type(_coconut_match_args[1]) in _coconut_self_match_types:  #82 (line in Coconut source)
                        raise _coconut.TypeError("too many positional args in class match (pattern requires 3; 'COO' only supports 1)")  #82 (line in Coconut source)
                        _coconut_match_check_5 = True  #82 (line in Coconut source)

                if not _coconut_match_check_5:  #82 (line in Coconut source)
                    _coconut_match_set_name_row = _coconut_sentinel  #82 (line in Coconut source)
                    _coconut_match_set_name_col = _coconut_sentinel  #82 (line in Coconut source)
                    _coconut_match_set_name_vals = _coconut_sentinel  #82 (line in Coconut source)
                    if not _coconut.type(_coconut_match_args[1]) in _coconut_self_match_types:  #82 (line in Coconut source)
                        _coconut_match_temp_8 = _coconut.getattr(COO, '__match_args__', ())  # type: _coconut.typing.Any  # type: ignore  #82 (line in Coconut source)
                        if not _coconut.isinstance(_coconut_match_temp_8, _coconut.tuple):  #82 (line in Coconut source)
                            raise _coconut.TypeError("COO.__match_args__ must be a tuple")  #82 (line in Coconut source)
                        if _coconut.len(_coconut_match_temp_8) < 3:  #82 (line in Coconut source)
                            raise _coconut.TypeError("too many positional args in class match (pattern requires 3; 'COO' only supports %s)" % (_coconut.len(_coconut_match_temp_8),))  #82 (line in Coconut source)
                        _coconut_match_temp_9 = _coconut.getattr(_coconut_match_args[1], _coconut_match_temp_8[0], _coconut_sentinel)  #82 (line in Coconut source)
                        _coconut_match_temp_10 = _coconut.getattr(_coconut_match_args[1], _coconut_match_temp_8[1], _coconut_sentinel)  #82 (line in Coconut source)
                        _coconut_match_temp_11 = _coconut.getattr(_coconut_match_args[1], _coconut_match_temp_8[2], _coconut_sentinel)  #82 (line in Coconut source)
                        if (_coconut_match_temp_9 is not _coconut_sentinel) and (_coconut_match_temp_10 is not _coconut_sentinel) and (_coconut_match_temp_11 is not _coconut_sentinel):  #82 (line in Coconut source)
                            _coconut_match_set_name_row = _coconut_match_temp_9  #82 (line in Coconut source)
                            _coconut_match_set_name_col = _coconut_match_temp_10  #82 (line in Coconut source)
                            _coconut_match_set_name_vals = _coconut_match_temp_11  #82 (line in Coconut source)
                            _coconut_match_check_5 = True  #82 (line in Coconut source)
                    if _coconut_match_check_5:  #82 (line in Coconut source)
                        if _coconut_match_set_name_row is not _coconut_sentinel:  #82 (line in Coconut source)
                            row = _coconut_match_set_name_row  #82 (line in Coconut source)
                        if _coconut_match_set_name_col is not _coconut_sentinel:  #82 (line in Coconut source)
                            col = _coconut_match_set_name_col  #82 (line in Coconut source)
                        if _coconut_match_set_name_vals is not _coconut_sentinel:  #82 (line in Coconut source)
                            vals = _coconut_match_set_name_vals  #82 (line in Coconut source)




    if _coconut_match_check_5:  #82 (line in Coconut source)
        if _coconut_match_set_name_shape is not _coconut_sentinel:  #82 (line in Coconut source)
            shape = _coconut_match_set_name_shape  #82 (line in Coconut source)
    if not _coconut_match_check_5:  #82 (line in Coconut source)
        raise _coconut_FunctionMatchError('match def _to_sparse(shape, COO(row,col,vals)):', _coconut_match_args)  #82 (line in Coconut source)

    coords = np.array([row, col])  #83 (line in Coconut source)
    return sparse.COO(coords, data=vals, shape=shape)  #84 (line in Coconut source)



@serde  #87 (line in Coconut source)
@dataclass  #88 (line in Coconut source)
class SerialRandWalks(_coconut.object):  #89 (line in Coconut source)
    graph = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: SerialSparse  #90 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #90 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #90 (line in Coconut source)
    __annotations__["graph"] = SerialSparse  #90 (line in Coconut source)
    jumps = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray  #91 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #91 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #91 (line in Coconut source)
    __annotations__["jumps"] = npt.NDArray  #91 (line in Coconut source)
    activations = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: SerialSparse  #92 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #92 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #92 (line in Coconut source)
    __annotations__["activations"] = SerialSparse  #92 (line in Coconut source)

_coconut_call_set_names(SerialRandWalks)  #94 (line in Coconut source)
