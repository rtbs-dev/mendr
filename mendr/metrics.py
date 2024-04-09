#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x528c6227

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
from scipy.integrate import trapezoid  #2 (line in Coconut source)
from scipy.integrate import cumulative_trapezoid  #2 (line in Coconut source)
# from sklearn.metrics import precision_recall_curve, fbeta_score
from scipy.linalg import sqrtm  #4 (line in Coconut source)
from jaxtyping import Bool  #5 (line in Coconut source)
from jaxtyping import jaxtyped  #5 (line in Coconut source)
from jaxtyping import Float  #5 (line in Coconut source)
from beartype import beartype  #6 (line in Coconut source)
from dataclasses import dataclass  #7 (line in Coconut source)
from dataclasses import field  #7 (line in Coconut source)

PredProb = Float[np.ndarray, 'features']  # type: _coconut.typing.TypeAlias  #9 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #9 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #9 (line in Coconut source)
__annotations__["PredProb"] = _coconut.typing.TypeAlias  #9 (line in Coconut source)
ProbThres = Float[np.ndarray, 'batch']  # type: _coconut.typing.TypeAlias  #10 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #10 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #10 (line in Coconut source)
__annotations__["ProbThres"] = _coconut.typing.TypeAlias  #10 (line in Coconut source)
PredThres = Bool[np.ndarray, 'batch features']  # type: _coconut.typing.TypeAlias  #11 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #11 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #11 (line in Coconut source)
__annotations__["PredThres"] = _coconut.typing.TypeAlias  #11 (line in Coconut source)

@_coconut_tco  #13 (line in Coconut source)
def gen_thres_vals(x  # type: PredProb  #13 (line in Coconut source)
    ):  #13 (line in Coconut source)
# type: (...) -> ProbThres
    return _coconut_tail_call((_coconut_complex_partial(np.pad, {1: ((1, 1))}, 2, (), constant_values=(0, 1))), (np.unique)(x))  # implicit 0,1 thres endpts  #14 (line in Coconut source)




@_coconut_tco  #25 (line in Coconut source)
def _all_thres(x,  # type: PredProb  #25 (line in Coconut source)
    t  # type: ProbThres  #25 (line in Coconut source)
    ):  #25 (line in Coconut source)
# type: (...) -> PredThres
    return _coconut_tail_call(np.less_equal.outer, t, x)  #26 (line in Coconut source)



@_coconut_tco  #29 (line in Coconut source)
def _bool_contract(A,  # type: PredThres  #29 (line in Coconut source)
    B  # type: PredThres  #29 (line in Coconut source)
    ):  #29 (line in Coconut source)
    return _coconut_tail_call((A * B).sum, axis=-1)  #29 (line in Coconut source)

@_coconut_tco  #30 (line in Coconut source)
def _TP(Yt,  # type: PredThres  #30 (line in Coconut source)
    Pt  # type: PredThres  #30 (line in Coconut source)
    ):  #30 (line in Coconut source)
    return _coconut_tail_call(_bool_contract, Pt, Yt)  #30 (line in Coconut source)

@_coconut_tco  #31 (line in Coconut source)
def _FP(Yt,  # type: PredThres  #31 (line in Coconut source)
    Pt  # type: PredThres  #31 (line in Coconut source)
    ):  #31 (line in Coconut source)
    return _coconut_tail_call(_bool_contract, Pt, ~Yt)  #31 (line in Coconut source)

@_coconut_tco  #32 (line in Coconut source)
def _FN(Yt,  # type: PredThres  #32 (line in Coconut source)
    Pt  # type: PredThres  #32 (line in Coconut source)
    ):  #32 (line in Coconut source)
    return _coconut_tail_call(_bool_contract, ~Pt, Yt)  #32 (line in Coconut source)

@_coconut_tco  #33 (line in Coconut source)
def _TN(Yt,  # type: PredThres  #33 (line in Coconut source)
    Pt  # type: PredThres  #33 (line in Coconut source)
    ):  #33 (line in Coconut source)
    return _coconut_tail_call(_bool_contract, ~Pt, ~Yt)  #33 (line in Coconut source)


@dataclass  #35 (line in Coconut source)
class Contingent(_coconut.object):  #36 (line in Coconut source)
    y_true = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: PredThres  #37 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #37 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #37 (line in Coconut source)
    __annotations__["y_true"] = PredThres  #37 (line in Coconut source)
    y_pred = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: PredThres  #38 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #38 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #38 (line in Coconut source)
    __annotations__["y_pred"] = PredThres  #38 (line in Coconut source)

    TP = field(init=False)  # type: ProbThres  #40 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #40 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #40 (line in Coconut source)
    __annotations__["TP"] = ProbThres  #40 (line in Coconut source)
    FP = field(init=False)  # type: ProbThres  #41 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #41 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #41 (line in Coconut source)
    __annotations__["FP"] = ProbThres  #41 (line in Coconut source)
    FN = field(init=False)  # type: ProbThres  #42 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #42 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #42 (line in Coconut source)
    __annotations__["FN"] = ProbThres  #42 (line in Coconut source)
    TN = field(init=False)  # type: ProbThres  #43 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #43 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #43 (line in Coconut source)
    __annotations__["TN"] = ProbThres  #43 (line in Coconut source)

    PP = field(init=False)  # type: ProbThres  #45 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #45 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #45 (line in Coconut source)
    __annotations__["PP"] = ProbThres  #45 (line in Coconut source)
    PN = field(init=False)  # type: ProbThres  #46 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #46 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #46 (line in Coconut source)
    __annotations__["PN"] = ProbThres  #46 (line in Coconut source)
    P = field(init=False)  # type: ProbThres  #47 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #47 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #47 (line in Coconut source)
    __annotations__["P"] = ProbThres  #47 (line in Coconut source)
    N = field(init=False)  # type: ProbThres  #48 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #48 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #48 (line in Coconut source)
    __annotations__["N"] = ProbThres  #48 (line in Coconut source)


    PPV = field(init=False)  # type: ProbThres  #51 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #51 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #51 (line in Coconut source)
    __annotations__["PPV"] = ProbThres  #51 (line in Coconut source)
    NPV = field(init=False)  # type: ProbThres  #52 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #52 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #52 (line in Coconut source)
    __annotations__["NPV"] = ProbThres  #52 (line in Coconut source)
    TPR = field(init=False)  # type: ProbThres  #53 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #53 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #53 (line in Coconut source)
    __annotations__["TPR"] = ProbThres  #53 (line in Coconut source)
    TNR = field(init=False)  # type: ProbThres  #54 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #54 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #54 (line in Coconut source)
    __annotations__["TNR"] = ProbThres  #54 (line in Coconut source)

    def __post_init__(self):  #56 (line in Coconut source)
        self.y_true = np.atleast_2d(self.y_true)  #57 (line in Coconut source)
        self.y_pred = np.atleast_2d(self.y_pred)  #58 (line in Coconut source)
        self.TP = _TP(self.y_true, self.y_pred)  #59 (line in Coconut source)
        self.FP = _FP(self.y_true, self.y_pred)  #60 (line in Coconut source)
        self.FN = _FN(self.y_true, self.y_pred)  #61 (line in Coconut source)
        self.TN = _TN(self.y_true, self.y_pred)  #62 (line in Coconut source)

        self.PP = self.TP + self.FP  #64 (line in Coconut source)
        self.PN = self.FN + self.TN  #65 (line in Coconut source)
        self.P = self.TP + self.FN  #66 (line in Coconut source)
        self.N = self.FP + self.TN  #67 (line in Coconut source)

# self.PPV = np.divide(self.TP, self.PP, out=np.ones_like(self.TP), where=self.PP!=0.)
        self.PPV = np.ma.divide(self.TP, self.PP)  #70 (line in Coconut source)
        self.NPV = np.ma.divide(self.TN, self.PN)  #71 (line in Coconut source)
        self.TPR = np.ma.divide(self.TP, self.P)  #72 (line in Coconut source)
        self.TNR = np.ma.divide(self.TN, self.N)  #73 (line in Coconut source)



    @property  #76 (line in Coconut source)
    def f_beta(self, beta):  #77 (line in Coconut source)
        return (1 + beta**2) * np.divide(self.PPV * self.TPR, beta**2 * self.PPV + self.TPR)  #77 (line in Coconut source)


    @property  #82 (line in Coconut source)
    @_coconut_tco  #83 (line in Coconut source)
    def f1(self):  #83 (line in Coconut source)
        return _coconut_tail_call(self.f_beta, beta=1)  #83 (line in Coconut source)


    @property  #85 (line in Coconut source)
    @_coconut_tco  #86 (line in Coconut source)
    def recall(self):  #86 (line in Coconut source)
        return _coconut_tail_call(self.TPR.filled, 0)  #86 (line in Coconut source)


    @property  #88 (line in Coconut source)
    @_coconut_tco  #89 (line in Coconut source)
    def precision(self):  #89 (line in Coconut source)
        return _coconut_tail_call(self.PPV.filled, 1)  #89 (line in Coconut source)


# def PPV(Yt:PredThres,Pt:PredThres) = TP/PP
# def NPV(Yt:PredThres,Pt:PredThres) = TN/PN
# def TPR(Yt:PredThres,Pt:PredThres) = TP/
# def TNR(Yt:PredThres,Pt:PredThres) = _bool_contract(~Pt,~Yt)


_coconut_call_set_names(Contingent)  #97 (line in Coconut source)
@_coconut_tco  #97 (line in Coconut source)
def f_beta(beta,  # type: float  #97 (line in Coconut source)
    Y  # type: Contingent  #97 (line in Coconut source)
    ):  #97 (line in Coconut source)
# type: (...) -> ProbThres
    top = (1 + beta**2) * Y.PPV * Y.TPR  #98 (line in Coconut source)
    bottom = beta**2 * Y.PPV + Y.TPR  #99 (line in Coconut source)

    return _coconut_tail_call(np.ma.divide(top, bottom).filled, 0.)  #101 (line in Coconut source)


@_coconut_tco  #103 (line in Coconut source)
def F1(Y  # type: Contingent  #103 (line in Coconut source)
    ):  #103 (line in Coconut source)
# type: (...) -> ProbThres
    return _coconut_tail_call(f_beta, 1., Y)  #103 (line in Coconut source)


@_coconut_tco  #105 (line in Coconut source)
def recall(Y  # type: Contingent  #105 (line in Coconut source)
    ):  #105 (line in Coconut source)
# type: (...) -> ProbThres
    return _coconut_tail_call(Y.TPR.filled, 0.)  #105 (line in Coconut source)


@_coconut_tco  #107 (line in Coconut source)
def precision(Y  # type: Contingent  #107 (line in Coconut source)
    ):  #107 (line in Coconut source)
# type: (...) -> ProbThres
    return _coconut_tail_call(Y.PPV.filled, 1.)  #107 (line in Coconut source)


def matt_corrcoef(Y  # type: Contingent  #109 (line in Coconut source)
    ):  #109 (line in Coconut source)
# type: (...) -> ProbThres
    return 1 - cdist(Y.y_pred, Y.y_true, "correlation")[:, 0]  #110 (line in Coconut source)

# def precision(y_true, y_pred):
#     TP,FP,TN,FN = _retrieval_square(y_true, p_pred)


def _wasserstein_gaussian(C1, C2):  #115 (line in Coconut source)
    a = np.trace(C1 + C2)  #116 (line in Coconut source)
    sqrtC1 = sqrtm(C1)  #117 (line in Coconut source)
    b = np.trace(sqrtm(_coconut_matmul(_coconut_matmul(sqrtC1, C2), sqrtC1)))  #118 (line in Coconut source)

    X = rw.to_array()  #120 (line in Coconut source)
# print(a,b)
    return a - 2 * b  #122 (line in Coconut source)


@jaxtyped(typechecker=beartype)  #124 (line in Coconut source)
@_coconut_tco  #125 (line in Coconut source)
def bhattacharyya(a,  # type: PredProb  #125 (line in Coconut source)
    b  # type: PredProb  #125 (line in Coconut source)
    ):  #125 (line in Coconut source)
    """non-metric distance between distributions"""  #126 (line in Coconut source)
    return _coconut_tail_call(np.sqrt(a * b).sum, axis=0)  #127 (line in Coconut source)



@jaxtyped(typechecker=beartype)  #130 (line in Coconut source)
@_coconut_tco  #131 (line in Coconut source)
def hellinger(a,  # type: PredProb  #131 (line in Coconut source)
    b  # type: PredProb  #131 (line in Coconut source)
    ):  #131 (line in Coconut source)
    """distance metric between binary distributions"""  #132 (line in Coconut source)
    return _coconut_tail_call(np.sqrt, 1 - bhattacharyya(a, b))  #133 (line in Coconut source)


@jaxtyped(typechecker=beartype)  #135 (line in Coconut source)
@_coconut_tco  #136 (line in Coconut source)
def thres_expect(x_thres,  # type: Float[np.ndarray, 't']  #136 (line in Coconut source)
    score  # type: Float[np.ndarray, 't']  #136 (line in Coconut source)
    ):  #136 (line in Coconut source)
# type: (...) -> float
# return 0.5*thres_expect(stats.beta(0.5,0.5),x_thres, score)+0.5*thres_expect(stats.beta(2.5,1.7),x_thres,score)
# return thres_expect(stats.beta(2.5,1.7), x_thres,score)
    return _coconut_tail_call(trapezoid, score, x=x_thres)  #139 (line in Coconut source)
