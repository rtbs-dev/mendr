#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xa5eb051b

# Compiled with Coconut version 3.1.0

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys
import os as _coconut_os
_coconut_header_info = ('3.1.0', '', False)
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
from __coconut__ import _coconut_tail_call, _coconut_tco, _coconut_call_set_names, _coconut_handle_cls_kwargs, _coconut_handle_cls_stargs, _namedtuple_of, _coconut, _coconut_Expected, _coconut_MatchError, _coconut_SupportsAdd, _coconut_SupportsMinus, _coconut_SupportsMul, _coconut_SupportsPow, _coconut_SupportsTruediv, _coconut_SupportsFloordiv, _coconut_SupportsMod, _coconut_SupportsAnd, _coconut_SupportsXor, _coconut_SupportsOr, _coconut_SupportsLshift, _coconut_SupportsRshift, _coconut_SupportsMatmul, _coconut_SupportsInv, _coconut_iter_getitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_complex_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_raise, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec, _coconut_comma_op, _coconut_arr_concat_op, _coconut_mk_anon_namedtuple, _coconut_matmul, _coconut_py_str, _coconut_flatten, _coconut_multiset, _coconut_back_none_pipe, _coconut_back_none_star_pipe, _coconut_back_none_dubstar_pipe, _coconut_forward_none_compose, _coconut_back_none_compose, _coconut_forward_none_star_compose, _coconut_back_none_star_compose, _coconut_forward_none_dubstar_compose, _coconut_back_none_dubstar_compose, _coconut_call_or_coefficient, _coconut_in, _coconut_not_in, _coconut_attritemgetter, _coconut_if_op
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
import csrgraph as cg  #2 (line in Coconut source)
from scipy import sparse as sp  #3 (line in Coconut source)
import networkx as nx  #4 (line in Coconut source)
from beartype import beartype  #5 (line in Coconut source)
from beartype.typing import Literal  #6 (line in Coconut source)

from .gmd_types import PosInt  #ReturnsGraph  #8 (line in Coconut source)

sparseG_from_nx = _coconut_forward_compose(nx.to_scipy_sparse_array, sp.csr_matrix, cg.csrgraph)  #10 (line in Coconut source)
DEFAULT_RNG = np.random.default_rng()  #11 (line in Coconut source)

RandGraphType = (_coconut.typing.Union[Literal['tree'], Literal['block']])  # type: _coconut.typing.TypeAlias  #13 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #13 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #13 (line in Coconut source)
__annotations__["RandGraphType"] = _coconut.typing.TypeAlias  #13 (line in Coconut source)

try:  #20 (line in Coconut source)
    _coconut_addpattern_0 = _coconut_addpattern(_graph_gen_dispatch)  # type: ignore  #20 (line in Coconut source)
except _coconut.NameError:  #20 (line in Coconut source)
    _coconut_addpattern_0 = lambda f: f  #20 (line in Coconut source)
@_coconut_addpattern_0  #20 (line in Coconut source)
@_coconut_mark_as_match  #20 (line in Coconut source)
def _graph_gen_dispatch(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #20 (line in Coconut source)
    _coconut_match_check_0 = False  #20 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #20 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #20 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #20 (line in Coconut source)
    if _coconut.len(_coconut_match_args) == 1:  #20 (line in Coconut source)
        if _coconut_match_args[0] == 'tree':  #20 (line in Coconut source)
            if not _coconut_match_kwargs:  #20 (line in Coconut source)
                _coconut_match_check_0 = True  #20 (line in Coconut source)
    if not _coconut_match_check_0:  #20 (line in Coconut source)
        raise _coconut_FunctionMatchError("addpattern def _graph_gen_dispatch('tree') = nx.random_labeled_tree", _coconut_match_args)  #20 (line in Coconut source)

    return nx.random_labeled_tree  #20 (line in Coconut source)

try:  #21 (line in Coconut source)
    _coconut_addpattern_1 = _coconut_addpattern(_graph_gen_dispatch)  # type: ignore  #21 (line in Coconut source)
except _coconut.NameError:  #21 (line in Coconut source)
    _coconut_addpattern_1 = lambda f: f  #21 (line in Coconut source)
@_coconut_addpattern_1  #21 (line in Coconut source)
@_coconut_mark_as_match  #21 (line in Coconut source)
def _graph_gen_dispatch(_coconut_match_first_arg=_coconut_sentinel, *_coconut_match_args, **_coconut_match_kwargs):  #21 (line in Coconut source)
    _coconut_match_check_1 = False  #21 (line in Coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #21 (line in Coconut source)
    if _coconut_match_first_arg is not _coconut_sentinel:  #21 (line in Coconut source)
        _coconut_match_args = (_coconut_match_first_arg,) + _coconut_match_args  #21 (line in Coconut source)
    if _coconut.len(_coconut_match_args) == 1:  #21 (line in Coconut source)
        if _coconut_match_args[0] == 'block':  #21 (line in Coconut source)
            if not _coconut_match_kwargs:  #21 (line in Coconut source)
                _coconut_match_check_1 = True  #21 (line in Coconut source)
    if not _coconut_match_check_1:  #21 (line in Coconut source)
        raise _coconut_FunctionMatchError("addpattern def _graph_gen_dispatch('block') =", _coconut_match_args)  #21 (line in Coconut source)

    return lambda n, *args, **kws: (nx.line_graph)(_graph_gen_dispatch('tree')(n + 1, *args, **kws))  #22 (line in Coconut source)
# addpattern def _graph_gen_dispatch(f `is_bearable` ReturnsGraph) = f



@beartype  #26 (line in Coconut source)
@_coconut_tco  #27 (line in Coconut source)
def graph_gen(kind,  # type: RandGraphType  #27 (line in Coconut source)
    n,  # type: PosInt  #27 (line in Coconut source)
    rng=DEFAULT_RNG, **kws):  #27 (line in Coconut source)
    """generate a supported random graph of size `n`

    supports the use of a custom random state (e.g. `numpy.random.default_rng`)
    Other keyword args get passed to the underlying random graph generator function
    """  #32 (line in Coconut source)
    return _coconut_tail_call(_graph_gen_dispatch(kind), n, seed=rng, **kws)  #33 (line in Coconut source)


@beartype  #35 (line in Coconut source)
@_coconut_tco  #36 (line in Coconut source)
def walk_randomly(graph,  # type: cg.csrgraph  #36 (line in Coconut source)
    n_jumps=None,  # type: _coconut.typing.Union[int, None]  #36 (line in Coconut source)
    n_walks=None,  # type: _coconut.typing.Union[int, None]  #36 (line in Coconut source)
    rng=DEFAULT_RNG):  #36 (line in Coconut source)
    """vectorized wrapper on CSGraph, with uniformly random starting nodes.
    TODO: allow arbitrary nodes selection distribution to be passed.
    """  #44 (line in Coconut source)
    n = graph.mat.shape[0]  #45 (line in Coconut source)
# rate_param = 1/size
    n_jumps = rng.geometric(1 / n) + 5 if n_jumps is None else n_jumps  #47 (line in Coconut source)
    n_walks = rng.negative_binomial(2, 1 / n) + 10 if n_walks is None else n_walks  #48 (line in Coconut source)
    starts = rng.choice(n, size=n_walks)  #49 (line in Coconut source)
    return _coconut_tail_call(graph.random_walks, walklen=n_jumps, start_nodes=starts, seed=rng)  #50 (line in Coconut source)





"""
def sim_graph_and_data(
    graph_gen_func,
    n_nodes,
    n_jumps,
    n_obs,
    rng=np.random.default_rng(2)
):
    Gnx = graph_gen_func(n_nodes)
    n = nx.number_of_nodes(Gnx) # graph_gen_func could change it!
    G = sparseG_from_nx(Gnx)
    starts = rng.choice(n, size=n_obs)
    rw = G.random_walks(walklen = n_jumps, start_nodes=starts) #TODO no seed... :(
    # trick to one-hot encode occurrences from walks
    row_idx = np.repeat(np.arange(rw.shape[0])[:,None], rw.shape[1],1).flatten()
    col_idx = rw.flatten()
    idx = np.unique(np.array([row_idx,col_idx]), axis=1)  # no duplicates
    # assert G.mat.shape[0]>=col_idx.max(), f'{G.mat.shape} A is bad for col pointer {col_idx.max()}'
    X = sparse.coo_matrix(
        (np.ones_like(idx[0]), (idx[0], idx[1])),
        shape=(rw.shape[0], n)
    ).astype(bool)
    return G.mat, X
"""  #78 (line in Coconut source)
