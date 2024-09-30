#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xa2646c58

# Compiled with Coconut version 3.1.2

"""
Inspired by BinSparse specification
"""

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
from jaxtyping import Shaped  #13 (line in Coconut source)
from scipy.sparse import spmatrix  #14 (line in Coconut source)
from scipy.sparse import sparray  #14 (line in Coconut source)
from scipy.sparse import coo_matrix  #14 (line in Coconut source)
# import

# @dataclass
# class BSFmt:
#     version:str = "0.1"

# @dataclass
# class SparseFMT:
#     shape: (int; int)

SparseValues = Shaped[np.ndarray, "#nnz"]  # type: _coconut.typing.TypeAlias  # could be scalar  #25 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  # could be scalar  #25 (line in Coconut source)
    __annotations__ = {}  # type: ignore  # could be scalar  #25 (line in Coconut source)
__annotations__["SparseValues"] = _coconut.typing.TypeAlias  # could be scalar  #25 (line in Coconut source)

@serde  #27 (line in Coconut source)
@dataclass  #28 (line in Coconut source)
class COO(_coconut.object):  #29 (line in Coconut source)
    """
    Data model for COO sparse arrays.

    Args:
        indices_0: npt.NDArray[int]
        indices_1: npt.NDArray[int]
        values: int: only supports binary sparse for now
    """  #37 (line in Coconut source)
    indices_0 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #38 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #38 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #38 (line in Coconut source)
    __annotations__["indices_0"] = npt.NDArray[int]  #38 (line in Coconut source)
    indices_1 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #39 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #39 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #39 (line in Coconut source)
    __annotations__["indices_1"] = npt.NDArray[int]  #39 (line in Coconut source)
    values = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: int  #| np.number  #40 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #| np.number  #40 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #| np.number  #40 (line in Coconut source)
    __annotations__["values"] = int  #| np.number  #40 (line in Coconut source)

_coconut_call_set_names(COO)  #42 (line in Coconut source)
@serde  #42 (line in Coconut source)
@dataclass  #43 (line in Coconut source)
class CSC(_coconut.object):  #44 (line in Coconut source)
    """
    Data model for CSC sparse arrays.

    Args:
        pointers_to_1: npt.NDArray[int]
        indices_1: npt.NDArray[int]
        values: int: only supports binary sparse for now
    """  #52 (line in Coconut source)
    pointers_to_1 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #53 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #53 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #53 (line in Coconut source)
    __annotations__["pointers_to_1"] = npt.NDArray[int]  #53 (line in Coconut source)
    indices_1 = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray[int]  #54 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #54 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #54 (line in Coconut source)
    __annotations__["indices_1"] = npt.NDArray[int]  #54 (line in Coconut source)
    values = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: int  #| np.number  #55 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #| np.number  #55 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #| np.number  #55 (line in Coconut source)
    __annotations__["values"] = int  #| np.number  #55 (line in Coconut source)


_coconut_call_set_names(CSC)  #58 (line in Coconut source)
SparseArrayType = (_coconut.typing.Union[COO, CSC])  # type: _coconut.typing.TypeAlias  #58 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #58 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #58 (line in Coconut source)
__annotations__["SparseArrayType"] = _coconut.typing.TypeAlias  #58 (line in Coconut source)

@serde(tagging=AdjacentTagging("format", "data_types"))  #63 (line in Coconut source)
@dataclass  #64 (line in Coconut source)
class SerialSparse(_coconut.object):  #65 (line in Coconut source)
    """
    Sparse, serializable array, typed with shape information.

    Args:
        shape: tuple[int,int]
        array: COO|CSC
    """  #72 (line in Coconut source)
# version:str = "0.1"
    shape = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: tuple[int, int]  #74 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #74 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #74 (line in Coconut source)
    __annotations__["shape"] = tuple[int, int]  #74 (line in Coconut source)
    array = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: SparseArrayType  #75 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #75 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #75 (line in Coconut source)
    __annotations__["array"] = SparseArrayType  #75 (line in Coconut source)


    _coconut_typevar_T_0 = _coconut.typing.TypeVar("_coconut_typevar_T_0")  #78 (line in Coconut source)

    @classmethod  #78 (line in Coconut source)
    def from_array(cls,  # type: Type[_coconut_typevar_T_0]  #79 (line in Coconut source)
        a):  #79 (line in Coconut source)
# type: (...) -> _coconut_typevar_T_0
        """parse supported array type into an instance of this class
        """  #81 (line in Coconut source)
        return ser_from_sparse(a)  #82 (line in Coconut source)



    _coconut_typevar_T_1 = _coconut.typing.TypeVar("_coconut_typevar_T_1")  #85 (line in Coconut source)

    def to_array(self):  #85 (line in Coconut source)
# type: (...) -> _coconut_typevar_T_1
        return ser_to_sparse(self.shape, self.array)  #86 (line in Coconut source)


# @addpattern

_coconut_call_set_names(SerialSparse)  #90 (line in Coconut source)
@_coconut_mark_as_match  #90 (line in Coconut source)
def ser_from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #90 (line in Coconut source)
    """parse a supported array or graph into a serializeable representation"""  #91 (line in Coconut source)
    _coconut_match_check_0 = False  #92 (line in Coconut source)
    _coconut_match_set_name_a = _coconut_sentinel  #92 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #92 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #92 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #92 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "a" in _coconut_match_kwargs)) == 1):  #92 (line in Coconut source)
        _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("a")  #92 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_0, sparse.COO):  #92 (line in Coconut source)
            _coconut_match_set_name_a = _coconut_match_temp_0  #92 (line in Coconut source)
            if not _coconut_match_kwargs:  #92 (line in Coconut source)
                _coconut_match_check_0 = True  #92 (line in Coconut source)
    if _coconut_match_check_0:  #92 (line in Coconut source)
        if _coconut_match_set_name_a is not _coconut_sentinel:  #92 (line in Coconut source)
            a = _coconut_match_set_name_a  #92 (line in Coconut source)
    if not _coconut_match_check_0:  #92 (line in Coconut source)
        raise _coconut_FunctionMatchError('match def ser_from_sparse(a `is_bearable` sparse.COO):', _coconut_match_args)  #92 (line in Coconut source)

    idx = a.coords  #92 (line in Coconut source)
    values = 1  # ignore a.data for the purposes of this work... for now  #93 (line in Coconut source)
# if np.all(np.isclose(a.data, a.data[0])):
#     values = a.data[0]
# else:
#     values = a.data
    return SerialSparse(a.shape, COO(idx[0], idx[1], values))  #98 (line in Coconut source)


try:  #100 (line in Coconut source)
    _coconut_addpattern_0 = _coconut_addpattern(ser_from_sparse)  # type: ignore  #100 (line in Coconut source)
except _coconut.NameError:  #100 (line in Coconut source)
    _coconut.warnings.warn("Deprecated use of 'addpattern def ser_from_sparse' with no pre-existing 'ser_from_sparse' function (use 'match def ser_from_sparse' for the first definition or switch to 'case def' syntax)", _coconut_CoconutWarning)  #100 (line in Coconut source)
    _coconut_addpattern_0 = lambda f: f  #100 (line in Coconut source)
@_coconut_addpattern_0  #100 (line in Coconut source)
@_coconut_mark_as_match  #100 (line in Coconut source)
def ser_from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #100 (line in Coconut source)
    _coconut_match_check_1 = False  #100 (line in Coconut source)
    _coconut_match_set_name_s = _coconut_sentinel  #100 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #100 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #100 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #100 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "s" in _coconut_match_kwargs)) == 1):  #100 (line in Coconut source)
        _coconut_match_temp_1 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("s")  #100 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_1, spmatrix):  #100 (line in Coconut source)
            _coconut_match_set_name_s = _coconut_match_temp_1  #100 (line in Coconut source)
            if not _coconut_match_kwargs:  #100 (line in Coconut source)
                _coconut_match_check_1 = True  #100 (line in Coconut source)
    if _coconut_match_check_1:  #100 (line in Coconut source)
        if _coconut_match_set_name_s is not _coconut_sentinel:  #100 (line in Coconut source)
            s = _coconut_match_set_name_s  #100 (line in Coconut source)
    if not _coconut_match_check_1:  #100 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def ser_from_sparse(s `is_bearable` spmatrix) = s |> ser_from_sparse .. sparse.COO.from_scipy_sparse', _coconut_match_args)  #100 (line in Coconut source)

    return (_coconut_forward_compose(sparse.COO.from_scipy_sparse, ser_from_sparse))(s)  #100 (line in Coconut source)

try:  #101 (line in Coconut source)
    _coconut_addpattern_1 = _coconut_addpattern(ser_from_sparse)  # type: ignore  #101 (line in Coconut source)
except _coconut.NameError:  #101 (line in Coconut source)
    _coconut.warnings.warn("Deprecated use of 'addpattern def ser_from_sparse' with no pre-existing 'ser_from_sparse' function (use 'match def ser_from_sparse' for the first definition or switch to 'case def' syntax)", _coconut_CoconutWarning)  #101 (line in Coconut source)
    _coconut_addpattern_1 = lambda f: f  #101 (line in Coconut source)
@_coconut_addpattern_1  #101 (line in Coconut source)
@_coconut_mark_as_match  #101 (line in Coconut source)
def ser_from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #101 (line in Coconut source)
    _coconut_match_check_2 = False  #101 (line in Coconut source)
    _coconut_match_set_name_s = _coconut_sentinel  #101 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #101 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #101 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #101 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "s" in _coconut_match_kwargs)) == 1):  #101 (line in Coconut source)
        _coconut_match_temp_2 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("s")  #101 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_2, sparray):  #101 (line in Coconut source)
            _coconut_match_set_name_s = _coconut_match_temp_2  #101 (line in Coconut source)
            if not _coconut_match_kwargs:  #101 (line in Coconut source)
                _coconut_match_check_2 = True  #101 (line in Coconut source)
    if _coconut_match_check_2:  #101 (line in Coconut source)
        if _coconut_match_set_name_s is not _coconut_sentinel:  #101 (line in Coconut source)
            s = _coconut_match_set_name_s  #101 (line in Coconut source)
    if not _coconut_match_check_2:  #101 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def ser_from_sparse(s `is_bearable` sparray) = s |> ser_from_sparse .. coo_matrix', _coconut_match_args)  #101 (line in Coconut source)

    return (_coconut_forward_compose(coo_matrix, ser_from_sparse))(s)  #101 (line in Coconut source)

try:  #102 (line in Coconut source)
    _coconut_addpattern_2 = _coconut_addpattern(ser_from_sparse)  # type: ignore  #102 (line in Coconut source)
except _coconut.NameError:  #102 (line in Coconut source)
    _coconut.warnings.warn("Deprecated use of 'addpattern def ser_from_sparse' with no pre-existing 'ser_from_sparse' function (use 'match def ser_from_sparse' for the first definition or switch to 'case def' syntax)", _coconut_CoconutWarning)  #102 (line in Coconut source)
    _coconut_addpattern_2 = lambda f: f  #102 (line in Coconut source)
@_coconut_addpattern_2  #102 (line in Coconut source)
@_coconut_mark_as_match  #102 (line in Coconut source)
def ser_from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #102 (line in Coconut source)
    _coconut_match_check_3 = False  #102 (line in Coconut source)
    _coconut_match_set_name_g = _coconut_sentinel  #102 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #102 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #102 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #102 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "g" in _coconut_match_kwargs)) == 1):  #102 (line in Coconut source)
        _coconut_match_temp_3 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("g")  #102 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_3, nx.Graph):  #102 (line in Coconut source)
            _coconut_match_set_name_g = _coconut_match_temp_3  #102 (line in Coconut source)
            if not _coconut_match_kwargs:  #102 (line in Coconut source)
                _coconut_match_check_3 = True  #102 (line in Coconut source)
    if _coconut_match_check_3:  #102 (line in Coconut source)
        if _coconut_match_set_name_g is not _coconut_sentinel:  #102 (line in Coconut source)
            g = _coconut_match_set_name_g  #102 (line in Coconut source)
    if not _coconut_match_check_3:  #102 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def ser_from_sparse(g `is_bearable` nx.Graph) = g |> ser_from_sparse .. nx.to_scipy_sparse_array', _coconut_match_args)  #102 (line in Coconut source)

    return (_coconut_forward_compose(nx.to_scipy_sparse_array, ser_from_sparse))(g)  #102 (line in Coconut source)

try:  #103 (line in Coconut source)
    _coconut_addpattern_3 = _coconut_addpattern(ser_from_sparse)  # type: ignore  #103 (line in Coconut source)
except _coconut.NameError:  #103 (line in Coconut source)
    _coconut.warnings.warn("Deprecated use of 'addpattern def ser_from_sparse' with no pre-existing 'ser_from_sparse' function (use 'match def ser_from_sparse' for the first definition or switch to 'case def' syntax)", _coconut_CoconutWarning)  #103 (line in Coconut source)
    _coconut_addpattern_3 = lambda f: f  #103 (line in Coconut source)
@_coconut_addpattern_3  #103 (line in Coconut source)
@_coconut_mark_as_match  #103 (line in Coconut source)
def ser_from_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #103 (line in Coconut source)
    _coconut_match_check_4 = False  #103 (line in Coconut source)
    _coconut_match_set_name_g = _coconut_sentinel  #103 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #103 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #103 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #103 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "g" in _coconut_match_kwargs)) == 1):  #103 (line in Coconut source)
        _coconut_match_temp_4 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("g")  #103 (line in Coconut source)
        if (is_bearable)(_coconut_match_temp_4, cg.csrgraph):  #103 (line in Coconut source)
            _coconut_match_set_name_g = _coconut_match_temp_4  #103 (line in Coconut source)
            if not _coconut_match_kwargs:  #103 (line in Coconut source)
                _coconut_match_check_4 = True  #103 (line in Coconut source)
    if _coconut_match_check_4:  #103 (line in Coconut source)
        if _coconut_match_set_name_g is not _coconut_sentinel:  #103 (line in Coconut source)
            g = _coconut_match_set_name_g  #103 (line in Coconut source)
    if not _coconut_match_check_4:  #103 (line in Coconut source)
        raise _coconut_FunctionMatchError('addpattern def ser_from_sparse(g `is_bearable` cg.csrgraph) = ser_from_sparse(g.mat)', _coconut_match_args)  #103 (line in Coconut source)

    return ser_from_sparse(g.mat)  #103 (line in Coconut source)



@_coconut_mark_as_match  #106 (line in Coconut source)
def ser_to_sparse(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #106 (line in Coconut source)
    """convert to a tensor/sparse-array representation"""  #107 (line in Coconut source)
    _coconut_match_check_5 = False  #108 (line in Coconut source)
    _coconut_match_set_name_shape = _coconut_sentinel  #108 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #108 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #108 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #108 (line in Coconut source)
    if (_coconut.len(_coconut_match_args) == 2) and ("shape" not in _coconut_match_kwargs):  #108 (line in Coconut source)
        _coconut_match_temp_5 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("shape")  #108 (line in Coconut source)
        _coconut_match_temp_6 = _coconut.getattr(COO, "_coconut_is_data", False) or _coconut.isinstance(COO, _coconut.tuple) and _coconut.all(_coconut.getattr(_coconut_x, "_coconut_is_data", False) for _coconut_x in COO)  # type: ignore  #108 (line in Coconut source)
        _coconut_match_set_name_shape = _coconut_match_temp_5  #108 (line in Coconut source)
        if not _coconut_match_kwargs:  #108 (line in Coconut source)
            _coconut_match_check_5 = True  #108 (line in Coconut source)
    if _coconut_match_check_5:  #108 (line in Coconut source)
        _coconut_match_check_5 = False  #108 (line in Coconut source)
        if not _coconut_match_check_5:  #108 (line in Coconut source)
            _coconut_match_set_name_row = _coconut_sentinel  #108 (line in Coconut source)
            _coconut_match_set_name_col = _coconut_sentinel  #108 (line in Coconut source)
            _coconut_match_set_name_vals = _coconut_sentinel  #108 (line in Coconut source)
            if (_coconut_match_temp_6) and (_coconut.isinstance(_coconut_match_args[1], COO)) and (_coconut.len(_coconut_match_args[1]) >= 3):  #108 (line in Coconut source)
                _coconut_match_set_name_row = _coconut_match_args[1][0]  #108 (line in Coconut source)
                _coconut_match_set_name_col = _coconut_match_args[1][1]  #108 (line in Coconut source)
                _coconut_match_set_name_vals = _coconut_match_args[1][2]  #108 (line in Coconut source)
                _coconut_match_temp_7 = _coconut.len(_coconut_match_args[1]) <= _coconut.max(3, _coconut.len(_coconut_match_args[1].__match_args__)) and _coconut.all(i in _coconut.getattr(_coconut_match_args[1], "_coconut_data_defaults", {}) and _coconut_match_args[1][i] == _coconut.getattr(_coconut_match_args[1], "_coconut_data_defaults", {})[i] for i in _coconut.range(3, _coconut.len(_coconut_match_args[1].__match_args__))) if _coconut.hasattr(_coconut_match_args[1], "__match_args__") else _coconut.len(_coconut_match_args[1]) == 3  # type: ignore  #108 (line in Coconut source)
                if _coconut_match_temp_7:  #108 (line in Coconut source)
                    _coconut_match_check_5 = True  #108 (line in Coconut source)
            if _coconut_match_check_5:  #108 (line in Coconut source)
                if _coconut_match_set_name_row is not _coconut_sentinel:  #108 (line in Coconut source)
                    row = _coconut_match_set_name_row  #108 (line in Coconut source)
                if _coconut_match_set_name_col is not _coconut_sentinel:  #108 (line in Coconut source)
                    col = _coconut_match_set_name_col  #108 (line in Coconut source)
                if _coconut_match_set_name_vals is not _coconut_sentinel:  #108 (line in Coconut source)
                    vals = _coconut_match_set_name_vals  #108 (line in Coconut source)

        if not _coconut_match_check_5:  #108 (line in Coconut source)
            if (not _coconut_match_temp_6) and (_coconut.isinstance(_coconut_match_args[1], COO)):  #108 (line in Coconut source)
                _coconut_match_check_5 = True  #108 (line in Coconut source)
            if _coconut_match_check_5:  #108 (line in Coconut source)
                _coconut_match_check_5 = False  #108 (line in Coconut source)
                if not _coconut_match_check_5:  #108 (line in Coconut source)
                    if _coconut.type(_coconut_match_args[1]) in _coconut_self_match_types:  #108 (line in Coconut source)
                        raise _coconut.TypeError("too many positional args in class match (pattern requires 3; 'COO' only supports 1)")  #108 (line in Coconut source)
                        _coconut_match_check_5 = True  #108 (line in Coconut source)

                if not _coconut_match_check_5:  #108 (line in Coconut source)
                    _coconut_match_set_name_row = _coconut_sentinel  #108 (line in Coconut source)
                    _coconut_match_set_name_col = _coconut_sentinel  #108 (line in Coconut source)
                    _coconut_match_set_name_vals = _coconut_sentinel  #108 (line in Coconut source)
                    if not _coconut.type(_coconut_match_args[1]) in _coconut_self_match_types:  #108 (line in Coconut source)
                        _coconut_match_temp_8 = _coconut.getattr(COO, '__match_args__', ())  # type: _coconut.typing.Any  # type: ignore  #108 (line in Coconut source)
                        if not _coconut.isinstance(_coconut_match_temp_8, _coconut.tuple):  #108 (line in Coconut source)
                            raise _coconut.TypeError("COO.__match_args__ must be a tuple")  #108 (line in Coconut source)
                        if _coconut.len(_coconut_match_temp_8) < 3:  #108 (line in Coconut source)
                            raise _coconut.TypeError("too many positional args in class match (pattern requires 3; 'COO' only supports %s)" % (_coconut.len(_coconut_match_temp_8),))  #108 (line in Coconut source)
                        _coconut_match_temp_9 = _coconut.getattr(_coconut_match_args[1], _coconut_match_temp_8[0], _coconut_sentinel)  #108 (line in Coconut source)
                        _coconut_match_temp_10 = _coconut.getattr(_coconut_match_args[1], _coconut_match_temp_8[1], _coconut_sentinel)  #108 (line in Coconut source)
                        _coconut_match_temp_11 = _coconut.getattr(_coconut_match_args[1], _coconut_match_temp_8[2], _coconut_sentinel)  #108 (line in Coconut source)
                        if (_coconut_match_temp_9 is not _coconut_sentinel) and (_coconut_match_temp_10 is not _coconut_sentinel) and (_coconut_match_temp_11 is not _coconut_sentinel):  #108 (line in Coconut source)
                            _coconut_match_set_name_row = _coconut_match_temp_9  #108 (line in Coconut source)
                            _coconut_match_set_name_col = _coconut_match_temp_10  #108 (line in Coconut source)
                            _coconut_match_set_name_vals = _coconut_match_temp_11  #108 (line in Coconut source)
                            _coconut_match_check_5 = True  #108 (line in Coconut source)
                    if _coconut_match_check_5:  #108 (line in Coconut source)
                        if _coconut_match_set_name_row is not _coconut_sentinel:  #108 (line in Coconut source)
                            row = _coconut_match_set_name_row  #108 (line in Coconut source)
                        if _coconut_match_set_name_col is not _coconut_sentinel:  #108 (line in Coconut source)
                            col = _coconut_match_set_name_col  #108 (line in Coconut source)
                        if _coconut_match_set_name_vals is not _coconut_sentinel:  #108 (line in Coconut source)
                            vals = _coconut_match_set_name_vals  #108 (line in Coconut source)




    if _coconut_match_check_5:  #108 (line in Coconut source)
        if _coconut_match_set_name_shape is not _coconut_sentinel:  #108 (line in Coconut source)
            shape = _coconut_match_set_name_shape  #108 (line in Coconut source)
    if not _coconut_match_check_5:  #108 (line in Coconut source)
        raise _coconut_FunctionMatchError('match def ser_to_sparse(shape, COO(row,col,vals)):', _coconut_match_args)  #108 (line in Coconut source)

    coords = np.array([row, col])  #108 (line in Coconut source)
    return sparse.COO(coords, data=vals, shape=shape)  #109 (line in Coconut source)



@serde  #112 (line in Coconut source)
@dataclass  #113 (line in Coconut source)
class SerialRandWalks(_coconut.object):  #114 (line in Coconut source)
    """MENDR 'problem' container,

    Holds a sparse adjacency matrix (graph) and an array of random walks (node ID jumps)

    The "activations" sparse array is the representation to be used in MENDR challenges,
    with all ordering removed; visited nodes are instead marked with a "1".

    Args:
        graph: SerialSparse
        jumps: npt.NDArray
        activations: SerialSparse

    """  #127 (line in Coconut source)
    graph = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: SerialSparse  #128 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #128 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #128 (line in Coconut source)
    __annotations__["graph"] = SerialSparse  #128 (line in Coconut source)
    jumps = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: npt.NDArray  #129 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #129 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #129 (line in Coconut source)
    __annotations__["jumps"] = npt.NDArray  #129 (line in Coconut source)
    activations = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: SerialSparse  #130 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #130 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #130 (line in Coconut source)
    __annotations__["activations"] = SerialSparse  #130 (line in Coconut source)

_coconut_call_set_names(SerialRandWalks)  #132 (line in Coconut source)
