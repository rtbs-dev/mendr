#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xd9691288

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
# from sklearn.metrics import precision_recall_curve, fbeta_score
from scipy.stats import ecdf  #3 (line in Coconut source)
from jaxtyping import Bool  #4 (line in Coconut source)
from jaxtyping import Float  #4 (line in Coconut source)
from dataclasses import dataclass  #5 (line in Coconut source)
from dataclasses import field  #5 (line in Coconut source)
from sklearn.preprocessing import minmax_scale  #6 (line in Coconut source)
import warnings  #7 (line in Coconut source)

__all__ = ["Contingent", "recall", "precision", "f_beta", "F1", "matthews_corrcoef", "fowlkes_mallows"]  #9 (line in Coconut source)

PredProb = Float[np.ndarray, 'features']  # type: _coconut.typing.TypeAlias  #19 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #19 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #19 (line in Coconut source)
__annotations__["PredProb"] = _coconut.typing.TypeAlias  #19 (line in Coconut source)
ProbThres = Float[np.ndarray, 'batch']  # type: _coconut.typing.TypeAlias  #20 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #20 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #20 (line in Coconut source)
__annotations__["ProbThres"] = _coconut.typing.TypeAlias  #20 (line in Coconut source)
PredThres = Bool[np.ndarray, 'batch features']  # type: _coconut.typing.TypeAlias  #21 (line in Coconut source)
if "__annotations__" not in _coconut.locals():  #21 (line in Coconut source)
    __annotations__ = {}  # type: ignore  #21 (line in Coconut source)
__annotations__["PredThres"] = _coconut.typing.TypeAlias  #21 (line in Coconut source)


def quantile_tf(x  # type: PredProb  #24 (line in Coconut source)
    ):  #24 (line in Coconut source)
# type: (...) -> (ProbThres, PredProb)
    cdf = ecdf(x).cdf  #25 (line in Coconut source)
    p = (_coconut_complex_partial(np.pad, {1: ((1, 1))}, 2, (), constant_values=(0, 1)))(cdf.probabilities)  #26 (line in Coconut source)
    return p, cdf.evaluate(x)  #27 (line in Coconut source)


def minmax_tf(x  # type: PredProb  #29 (line in Coconut source)
    ):  #29 (line in Coconut source)
# type: (...) -> (ProbThres, PredProb)
    x_p = minmax_scale(x, feature_range=(1e-5, 1 - 1e-5))  #30 (line in Coconut source)
    p = np.pad(np.unique(x_p), ((1, 1)), constant_values=(0, 1))  #31 (line in Coconut source)
    return p, x_p  #32 (line in Coconut source)

# def _all_thres(x:PredProb, t:ProbThres)->PredThres:
# return np.less_equal.outer(t, x)

#TODO use density (.getnnz()) for sparse via dispatching

def _bool_contract(A,  # type: PredThres  #38 (line in Coconut source)
    B  # type: PredThres  #38 (line in Coconut source)
    ):  #38 (line in Coconut source)
    return (A * B).sum(axis=-1)  #38 (line in Coconut source)

def _TP(actual,  # type: PredThres  #39 (line in Coconut source)
    pred  # type: PredThres  #39 (line in Coconut source)
    ):  #39 (line in Coconut source)
    return _bool_contract(pred, actual)  #39 (line in Coconut source)

def _FP(actual,  # type: PredThres  #40 (line in Coconut source)
    pred  # type: PredThres  #40 (line in Coconut source)
    ):  #40 (line in Coconut source)
    return _bool_contract(pred, ~actual)  #40 (line in Coconut source)

def _FN(actual,  # type: PredThres  #41 (line in Coconut source)
    pred  # type: PredThres  #41 (line in Coconut source)
    ):  #41 (line in Coconut source)
    return _bool_contract(~pred, actual)  #41 (line in Coconut source)

def _TN(actual,  # type: PredThres  #42 (line in Coconut source)
    pred  # type: PredThres  #42 (line in Coconut source)
    ):  #42 (line in Coconut source)
    return _bool_contract(~pred, ~actual)  #42 (line in Coconut source)


@dataclass  #44 (line in Coconut source)
class Contingent(_coconut.object):  #45 (line in Coconut source)
    """ dataclass to hold true and (batched) predicted values

    Parameters:
        y_true: True positive and negative binary classifications
        y_pred: Predicted, possible batched (tensor)
        weights: weight(s) for y_pred, useful for expected values of scores

    Properties:
        f_beta: beta-weighted harmonic mean of precision and recall
        F:  alias for f_beta(1)
        recall: a.k.a. true-positive rate
        precision: a.k.a. positive-predictive-value (PPV)
        mcc: Matthew's Correlation Coefficient
        G: Fowlkes-Mallows score (geometric mean of precision and recall)
    """  #60 (line in Coconut source)
    y_true = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: PredThres  #61 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #61 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #61 (line in Coconut source)
    __annotations__["y_true"] = PredThres  #61 (line in Coconut source)
    y_pred = _coconut.typing.cast(_coconut.typing.Any, _coconut.Ellipsis)  # type: PredThres  #62 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #62 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #62 (line in Coconut source)
    __annotations__["y_pred"] = PredThres  #62 (line in Coconut source)

    weights = None  # type: _coconut.typing.Union[ProbThres, None]  #64 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #64 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #64 (line in Coconut source)
    __annotations__["weights"] = _coconut.typing.Union[ProbThres, None]  #64 (line in Coconut source)

    TP = field(init=False)  # type: ProbThres  #66 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #66 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #66 (line in Coconut source)
    __annotations__["TP"] = ProbThres  #66 (line in Coconut source)
    FP = field(init=False)  # type: ProbThres  #67 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #67 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #67 (line in Coconut source)
    __annotations__["FP"] = ProbThres  #67 (line in Coconut source)
    FN = field(init=False)  # type: ProbThres  #68 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #68 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #68 (line in Coconut source)
    __annotations__["FN"] = ProbThres  #68 (line in Coconut source)
    TN = field(init=False)  # type: ProbThres  #69 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #69 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #69 (line in Coconut source)
    __annotations__["TN"] = ProbThres  #69 (line in Coconut source)


    PP = field(init=False)  # type: ProbThres  #72 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #72 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #72 (line in Coconut source)
    __annotations__["PP"] = ProbThres  #72 (line in Coconut source)
    PN = field(init=False)  # type: ProbThres  #73 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #73 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #73 (line in Coconut source)
    __annotations__["PN"] = ProbThres  #73 (line in Coconut source)
    P = field(init=False)  # type: ProbThres  #74 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #74 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #74 (line in Coconut source)
    __annotations__["P"] = ProbThres  #74 (line in Coconut source)
    N = field(init=False)  # type: ProbThres  #75 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #75 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #75 (line in Coconut source)
    __annotations__["N"] = ProbThres  #75 (line in Coconut source)


    PPV = field(init=False)  # type: ProbThres  #78 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #78 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #78 (line in Coconut source)
    __annotations__["PPV"] = ProbThres  #78 (line in Coconut source)
    NPV = field(init=False)  # type: ProbThres  #79 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #79 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #79 (line in Coconut source)
    __annotations__["NPV"] = ProbThres  #79 (line in Coconut source)
    TPR = field(init=False)  # type: ProbThres  #80 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #80 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #80 (line in Coconut source)
    __annotations__["TPR"] = ProbThres  #80 (line in Coconut source)
    TNR = field(init=False)  # type: ProbThres  #81 (line in Coconut source)
    if "__annotations__" not in _coconut.locals():  #81 (line in Coconut source)
        __annotations__ = {}  # type: ignore  #81 (line in Coconut source)
    __annotations__["TNR"] = ProbThres  #81 (line in Coconut source)

    def __post_init__(self):  #83 (line in Coconut source)
        self.y_true = np.atleast_2d(self.y_true)  #84 (line in Coconut source)
        self.y_pred = np.atleast_2d(self.y_pred)  #85 (line in Coconut source)
        self.TP = _TP(self.y_true, self.y_pred)  #86 (line in Coconut source)
        self.FP = _FP(self.y_true, self.y_pred)  #87 (line in Coconut source)
        self.FN = _FN(self.y_true, self.y_pred)  #88 (line in Coconut source)
        self.TN = _TN(self.y_true, self.y_pred)  #89 (line in Coconut source)

        self.PP = self.TP + self.FP  #91 (line in Coconut source)
        self.PN = self.FN + self.TN  #92 (line in Coconut source)
        self.P = self.TP + self.FN  #93 (line in Coconut source)
        self.N = self.FP + self.TN  #94 (line in Coconut source)

# self.PPV = np.divide(self.TP, self.PP, out=np.ones_like(self.TP), where=self.PP!=0.)
        self.PPV = np.ma.divide(self.TP, self.PP)  #97 (line in Coconut source)
        self.NPV = np.ma.divide(self.TN, self.PN)  #98 (line in Coconut source)
        self.TPR = np.ma.divide(self.TP, self.P)  #99 (line in Coconut source)
        self.TNR = np.ma.divide(self.TN, self.N)  #100 (line in Coconut source)



    _coconut_typevar_T_0 = _coconut.typing.TypeVar("_coconut_typevar_T_0")  #103 (line in Coconut source)

    @classmethod  #103 (line in Coconut source)
    def from_scalar(cls,  # type: Type[_coconut_typevar_T_0]  #104 (line in Coconut source)
        y_true, x  # type: _coconut.typing.Optional[PredProb]  #104 (line in Coconut source)
        ):  #104 (line in Coconut source)
# type: (...) -> _coconut.typing.Optional[_coconut_typevar_T_0]
        """ take scalar predictions and generate (batched) Contingent

        by default, x is rescaled to [0,1] and used as the weights parameter
        for the Contingent constructor. Only unique values are needed, since
        the thresholding only changes with each unique prediction value.

        Uses numpy's `less_equal.outer` to accomplish fast, vectorized thresholding
        and enable rapid estimation of batched scores accross all thresholds.


        Parameters:
            y_true: True pos/neg binary vector
            x: scalar weights for relative prediction strength (positive)
        """  #118 (line in Coconut source)
# p, x_p = quantile_tf(x)
        if x is None:  #120 (line in Coconut source)
            warnings.warn("`None` value recieved, passing the buck...")  #121 (line in Coconut source)
            return None  #122 (line in Coconut source)
        p, x_p = minmax_tf(x)  #123 (line in Coconut source)
        y_preds = np.less_equal.outer(p, x_p)  #124 (line in Coconut source)

        return cls(y_true, y_preds, weights=p)  #126 (line in Coconut source)




    @property  #130 (line in Coconut source)
    def f_beta(self, beta):  #131 (line in Coconut source)
        return f_beta(beta, self)  #131 (line in Coconut source)


    @property  #133 (line in Coconut source)
    def F(self):  #134 (line in Coconut source)
        return F1(self)  #134 (line in Coconut source)


    @property  #136 (line in Coconut source)
    def recall(self):  #137 (line in Coconut source)
        return recall(self)  #137 (line in Coconut source)


    @property  #139 (line in Coconut source)
    def precision(self):  #140 (line in Coconut source)
        return precision(self)  #140 (line in Coconut source)


    @property  #142 (line in Coconut source)
    def mcc(self):  #143 (line in Coconut source)
        return matthews_corrcoef(self)  #143 (line in Coconut source)


    @property  #145 (line in Coconut source)
    def G(self):  #146 (line in Coconut source)
        return fowlkes_mallows(self)  #146 (line in Coconut source)

# def PPV(Yt:PredThres,Pt:PredThres) = TP/PP
# def NPV(Yt:PredThres,Pt:PredThres) = TN/PN
# def TPR(Yt:PredThres,Pt:PredThres) = TP/
# def TNR(Yt:PredThres,Pt:PredThres) = _bool_contract(~Pt,~Yt)


_coconut_call_set_names(Contingent)  #153 (line in Coconut source)
def recall(Y  # type: Contingent  #153 (line in Coconut source)
    ):  #153 (line in Coconut source)
# type: (...) -> ProbThres
    """ True Positive Rate
    """  #155 (line in Coconut source)
    return Y.TPR.filled(1.)  #156 (line in Coconut source)



def precision(Y  # type: Contingent  #159 (line in Coconut source)
    ):  #159 (line in Coconut source)
# type: (...) -> ProbThres
    """ Positive Predictive Value
    """  #161 (line in Coconut source)
    return Y.PPV.filled(1.)  #162 (line in Coconut source)



def f_beta(beta,  # type: float  #165 (line in Coconut source)
    Y  # type: Contingent  #165 (line in Coconut source)
    ):  #165 (line in Coconut source)
# type: (...) -> ProbThres
    """F_beta score

    weighted harmonic mean of precision and recall, with beta-times
    more bias for recall.
    """  #170 (line in Coconut source)
    top = (1 + beta**2) * Y.PPV * Y.TPR  #171 (line in Coconut source)
    bottom = beta**2 * Y.PPV + Y.TPR  #172 (line in Coconut source)

    return np.ma.divide(top, bottom).filled(0.)  #174 (line in Coconut source)


def F1(Y  # type: Contingent  #176 (line in Coconut source)
    ):  #176 (line in Coconut source)
# type: (...) -> ProbThres
    """partially applied f_beta with beta=1 (equal/no bias)
    """  #178 (line in Coconut source)
    return f_beta(1., Y)  #179 (line in Coconut source)



def matthews_corrcoef(Y  # type: Contingent  #182 (line in Coconut source)
    ):  #182 (line in Coconut source)
# type: (...) -> ProbThres
    """ Matthew's Correlation Coefficient (MCC)

    Widely considered the most fair/least bias metric for imbalanced
    classification tasks.
    """  #187 (line in Coconut source)
    _coconut_where_m_0 = np.vstack([Y.TPR, Y.TNR, Y.PPV, Y.NPV])  #189 (line in Coconut source)
    _coconut_where_l_0 = np.sqrt(_coconut_where_m_0).prod(axis=0)  #190 (line in Coconut source)
    _coconut_where_r_0 = np.sqrt(1 - _coconut_where_m_0).prod(axis=0)  #191 (line in Coconut source)
# return 1-cdist(Y.y_pred, Y.y_true, "correlation")[:,0]

    return (_coconut_where_l_0 - _coconut_where_r_0).filled(0)  #194 (line in Coconut source)

def fowlkes_mallows(Y  # type: Contingent  #194 (line in Coconut source)
    ):  #194 (line in Coconut source)
# type: (...) -> ProbThres
    """ G, the geometric mean of precision and recall.

    commonly used in unsupervised cases where synthetic test-data
    has been made available (e.g. MENDR, clustering validation, etc.)
    """  #199 (line in Coconut source)
    return np.sqrt(recall(Y) * precision(Y))  #200 (line in Coconut source)


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
