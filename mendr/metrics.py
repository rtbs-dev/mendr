#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xe8d27d78

# Compiled with Coconut version 3.1.1

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys
import os as _coconut_os
_coconut_header_info = ('3.1.1', '', False)
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
from scipy.integrate import trapezoid  #2 (line in Coconut source)
from scipy.integrate import cumulative_trapezoid  #2 (line in Coconut source)
# from sklearn.metrics import precision_recall_curve, fbeta_score
from scipy.linalg import sqrtm  #4 (line in Coconut source)
from scipy.spatial.distance import cdist  #5 (line in Coconut source)
from scipy.stats import ecdf  #6 (line in Coconut source)
from jaxtyping import Bool  #7 (line in Coconut source)
from jaxtyping import jaxtyped  #7 (line in Coconut source)
from jaxtyping import Float  #7 (line in Coconut source)
from beartype import beartype  #8 (line in Coconut source)
from dataclasses import dataclass  #9 (line in Coconut source)
from dataclasses import field  #9 (line in Coconut source)
from sklearn.preprocessing import minmax_scale  #10 (line in Coconut source)
import warnings  #11 (line in Coconut source)

PredProb = Float[np.ndarray, 'features']  # type: _coconut.typing.TypeAlias  #13 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #13 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #13 (line in Coconut source)
__annotations__["PredProb"] = _coconut.typing.TypeAlias  #13 (line in Coconut source)
ProbThres = Float[np.ndarray, 'batch']  # type: _coconut.typing.TypeAlias  #14 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #14 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #14 (line in Coconut source)
__annotations__["ProbThres"] = _coconut.typing.TypeAlias  #14 (line in Coconut source)
PredThres = Bool[np.ndarray, 'batch features']  # type: _coconut.typing.TypeAlias  #15 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #15 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #15 (line in Coconut source)
__annotations__["PredThres"] = _coconut.typing.TypeAlias  #15 (line in Coconut source)


def quantile_tf(x  # type: PredProb  #18 (line in Coconut source)
    ):  #18 (line in Coconut source)
# type: (...) -> (ProbThres, PredProb)
    cdf = ecdf(x).cdf  #19 (line in Coconut source)
    p = (_coconut_complex_partial(np.pad, {1: ((1, 1))}, 2, (), constant_values=(0, 1)))(cdf.probabilities)  #20 (line in Coconut source)
    return p, cdf.evaluate(x)  #21 (line in Coconut source)


def minmax_tf(x  # type: PredProb  #23 (line in Coconut source)
    ):  #23 (line in Coconut source)
# type: (...) -> (ProbTrhes, PredProb)
    x_p = minmax_scale(x, feature_range=(1e-5, 1 - 1e-5))  #24 (line in Coconut source)
    p = np.pad(np.unique(x_p), ((1, 1)), constant_values=(0, 1))  #25 (line in Coconut source)
    return p, x_p  #26 (line in Coconut source)

# def _all_thres(x:PredProb, t:ProbThres)->PredThres:
# return np.less_equal.outer(t, x)

#TODO use density (.getnnz()) for sparse via dispatching

def _bool_contract(A,  # type: PredThres  #32 (line in Coconut source)
    B  # type: PredThres  #32 (line in Coconut source)
    ):  #32 (line in Coconut source)
    return (A * B).sum(axis=-1)  #32 (line in Coconut source)

def _TP(actual,  # type: PredThres  #33 (line in Coconut source)
    pred  # type: PredThres  #33 (line in Coconut source)
    ):  #33 (line in Coconut source)
    return _bool_contract(pred, actual)  #33 (line in Coconut source)

def _FP(actual,  # type: PredThres  #34 (line in Coconut source)
    pred  # type: PredThres  #34 (line in Coconut source)
    ):  #34 (line in Coconut source)
    return _bool_contract(pred, ~actual)  #34 (line in Coconut source)

def _FN(actual,  # type: PredThres  #35 (line in Coconut source)
    pred  # type: PredThres  #35 (line in Coconut source)
    ):  #35 (line in Coconut source)
    return _bool_contract(~pred, actual)  #35 (line in Coconut source)

def _TN(actual,  # type: PredThres  #36 (line in Coconut source)
    pred  # type: PredThres  #36 (line in Coconut source)
    ):  #36 (line in Coconut source)
    return _bool_contract(~pred, ~actual)  #36 (line in Coconut source)


@dataclass  #38 (line in Coconut source)
class Contingent(_coconut.object):  #39 (line in Coconut source)
    y_true = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: PredThres  #40 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #40 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #40 (line in Coconut source)
    __annotations__["y_true"] = PredThres  #40 (line in Coconut source)
    y_pred = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: PredThres  #41 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #41 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #41 (line in Coconut source)
    __annotations__["y_pred"] = PredThres  #41 (line in Coconut source)

    weights = None  # type: _coconut.typing.Union[ProbThres, None]  #43 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #43 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #43 (line in Coconut source)
    __annotations__["weights"] = _coconut.typing.Union[ProbThres, None]  #43 (line in Coconut source)

    TP = field(init=False)  # type: ProbThres  #45 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #45 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #45 (line in Coconut source)
    __annotations__["TP"] = ProbThres  #45 (line in Coconut source)
    FP = field(init=False)  # type: ProbThres  #46 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #46 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #46 (line in Coconut source)
    __annotations__["FP"] = ProbThres  #46 (line in Coconut source)
    FN = field(init=False)  # type: ProbThres  #47 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #47 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #47 (line in Coconut source)
    __annotations__["FN"] = ProbThres  #47 (line in Coconut source)
    TN = field(init=False)  # type: ProbThres  #48 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #48 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #48 (line in Coconut source)
    __annotations__["TN"] = ProbThres  #48 (line in Coconut source)


    PP = field(init=False)  # type: ProbThres  #51 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #51 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #51 (line in Coconut source)
    __annotations__["PP"] = ProbThres  #51 (line in Coconut source)
    PN = field(init=False)  # type: ProbThres  #52 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #52 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #52 (line in Coconut source)
    __annotations__["PN"] = ProbThres  #52 (line in Coconut source)
    P = field(init=False)  # type: ProbThres  #53 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #53 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #53 (line in Coconut source)
    __annotations__["P"] = ProbThres  #53 (line in Coconut source)
    N = field(init=False)  # type: ProbThres  #54 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #54 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #54 (line in Coconut source)
    __annotations__["N"] = ProbThres  #54 (line in Coconut source)


    PPV = field(init=False)  # type: ProbThres  #57 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #57 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #57 (line in Coconut source)
    __annotations__["PPV"] = ProbThres  #57 (line in Coconut source)
    NPV = field(init=False)  # type: ProbThres  #58 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #58 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #58 (line in Coconut source)
    __annotations__["NPV"] = ProbThres  #58 (line in Coconut source)
    TPR = field(init=False)  # type: ProbThres  #59 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #59 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #59 (line in Coconut source)
    __annotations__["TPR"] = ProbThres  #59 (line in Coconut source)
    TNR = field(init=False)  # type: ProbThres  #60 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #60 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #60 (line in Coconut source)
    __annotations__["TNR"] = ProbThres  #60 (line in Coconut source)

    def __post_init__(self):  #62 (line in Coconut source)
        self.y_true = np.atleast_2d(self.y_true)  #63 (line in Coconut source)
        self.y_pred = np.atleast_2d(self.y_pred)  #64 (line in Coconut source)
        self.TP = _TP(self.y_true, self.y_pred)  #65 (line in Coconut source)
        self.FP = _FP(self.y_true, self.y_pred)  #66 (line in Coconut source)
        self.FN = _FN(self.y_true, self.y_pred)  #67 (line in Coconut source)
        self.TN = _TN(self.y_true, self.y_pred)  #68 (line in Coconut source)

        self.PP = self.TP + self.FP  #70 (line in Coconut source)
        self.PN = self.FN + self.TN  #71 (line in Coconut source)
        self.P = self.TP + self.FN  #72 (line in Coconut source)
        self.N = self.FP + self.TN  #73 (line in Coconut source)

# self.PPV = np.divide(self.TP, self.PP, out=np.ones_like(self.TP), where=self.PP!=0.)
        self.PPV = np.ma.divide(self.TP, self.PP)  #76 (line in Coconut source)
        self.NPV = np.ma.divide(self.TN, self.PN)  #77 (line in Coconut source)
        self.TPR = np.ma.divide(self.TP, self.P)  #78 (line in Coconut source)
        self.TNR = np.ma.divide(self.TN, self.N)  #79 (line in Coconut source)



    _coconut_typevar_T_0 = _coconut.typing.TypeVar("_coconut_typevar_T_0")  #82 (line in Coconut source)

    @classmethod  #82 (line in Coconut source)
    def from_scalar(cls,  # type: Type[_coconut_typevar_T_0]  #83 (line in Coconut source)
        y_true, x  # type: _coconut.typing.Optional[PredProb]  #83 (line in Coconut source)
        ):  #83 (line in Coconut source)
# type: (...) -> _coconut.typing.Optional[_coconut_typevar_T_0]
# p, x_p = quantile_tf(x)
        if x is None:  #85 (line in Coconut source)
            warnings.warn("`None` value recieved, passing the buck...")  #86 (line in Coconut source)
            return None  #87 (line in Coconut source)
        p, x_p = minmax_tf(x)  #88 (line in Coconut source)
        y_preds = np.less_equal.outer(p, x_p)  #89 (line in Coconut source)

        return cls(y_true, y_preds, weights=p)  #91 (line in Coconut source)




    @property  #95 (line in Coconut source)
    def f_beta(self, beta):  #96 (line in Coconut source)
        return f_beta(beta, self)  #96 (line in Coconut source)


    @property  #98 (line in Coconut source)
    def F(self):  #99 (line in Coconut source)
        return F1(self)  #99 (line in Coconut source)


    @property  #101 (line in Coconut source)
    def recall(self):  #102 (line in Coconut source)
        return recall(self)  #102 (line in Coconut source)


    @property  #104 (line in Coconut source)
    def precision(self):  #105 (line in Coconut source)
        return precision(self)  #105 (line in Coconut source)


    @property  #107 (line in Coconut source)
    def mcc(self):  #108 (line in Coconut source)
        return matthews_corrcoef(self)  #108 (line in Coconut source)


    @property  #110 (line in Coconut source)
    def G(self):  #111 (line in Coconut source)
        return fowlkes_mallows(self)  #111 (line in Coconut source)

# def PPV(Yt:PredThres,Pt:PredThres) = TP/PP
# def NPV(Yt:PredThres,Pt:PredThres) = TN/PN
# def TPR(Yt:PredThres,Pt:PredThres) = TP/
# def TNR(Yt:PredThres,Pt:PredThres) = _bool_contract(~Pt,~Yt)


_coconut_call_set_names(Contingent)  #118 (line in Coconut source)
def recall(Y  # type: Contingent  #118 (line in Coconut source)
    ):  #118 (line in Coconut source)
# type: (...) -> ProbThres
    return Y.TPR.filled(1.)  #118 (line in Coconut source)


def precision(Y  # type: Contingent  #120 (line in Coconut source)
    ):  #120 (line in Coconut source)
# type: (...) -> ProbThres
    return Y.PPV.filled(1.)  #120 (line in Coconut source)


def f_beta(beta,  # type: float  #122 (line in Coconut source)
    Y  # type: Contingent  #122 (line in Coconut source)
    ):  #122 (line in Coconut source)
# type: (...) -> ProbThres
    top = (1 + beta**2) * Y.PPV * Y.TPR  #123 (line in Coconut source)
    bottom = beta**2 * Y.PPV + Y.TPR  #124 (line in Coconut source)

    return np.ma.divide(top, bottom).filled(0.)  #126 (line in Coconut source)


def F1(Y  # type: Contingent  #128 (line in Coconut source)
    ):  #128 (line in Coconut source)
# type: (...) -> ProbThres
    return f_beta(1., Y)  #128 (line in Coconut source)


def matthews_corrcoef(Y  # type: Contingent  #130 (line in Coconut source)
    ):  #130 (line in Coconut source)
# type: (...) -> ProbThres
    _coconut_where_m_0 = np.vstack([Y.TPR, Y.TNR, Y.PPV, Y.NPV])  #133 (line in Coconut source)
    _coconut_where_l_0 = np.sqrt(_coconut_where_m_0).prod(axis=0)  #134 (line in Coconut source)
    _coconut_where_r_0 = np.sqrt(1 - _coconut_where_m_0).prod(axis=0)  #135 (line in Coconut source)

    return (_coconut_where_l_0 - _coconut_where_r_0).filled(0)  #137 (line in Coconut source)

def fowlkes_mallows(Y  # type: Contingent  #137 (line in Coconut source)
    ):  #137 (line in Coconut source)
# type: (...) -> ProbThres
    return np.sqrt(recall(Y) * precision(Y))  #138 (line in Coconut source)


# def precision(y_true, y_pred):
#     TP,FP,TN,FN = _retrieval_square(y_true, p_pred)

# def _wasserstein_gaussian(C1, C2):
#     a = np.trace(C1+C2)
#     sqrtC1 = sqrtm(C1)
#     b = np.trace(sqrtm(sqrtC1@C2@sqrtC1))

#     X = rw.to_array()
#     # print(a,b)
#     return a - 2*b

# @jaxtyped(typechecker=beartype)
# def bhattacharyya(a:PredProb,b:PredProb):
#     """non-metric distance between distributions"""
#     return np.sqrt(a*b).sum(axis=0)


# @jaxtyped(typechecker=beartype)
# def hellinger(a:PredProb,b:PredProb):
#     """distance metric between binary distributions"""
#     return np.sqrt(1-bhattacharyya(a,b))

# @jaxtyped(typechecker=beartype)
# def thres_expect(x_thres:Float[np.ndarray,'t'], score:Float[np.ndarray, 't'])->float:
#     # return 0.5*thres_expect(stats.beta(0.5,0.5),x_thres, score)+0.5*thres_expect(stats.beta(2.5,1.7),x_thres,score)
#     # return thres_expect(stats.beta(2.5,1.7), x_thres,score)
#     return trapezoid(score, x=x_thres)
