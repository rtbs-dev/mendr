from autoregistry import Registry
import time
from typing import Literal, Annotated, TypedDict
from dvc.api import DVCFileSystem
import re
from pathlib import Path
from beartype import beartype
from beartype.vale import Is
from serde.json import from_json
from dvclive import Live

from affinis import associations as asc
from affinis.utils import _sq
from affinis.proximity import sinkhorn

from mendr.io import SerialRandWalks, SerialSparse
import mendr.metrics as m

import numpy as np

from scipy.integrate import trapezoid
from scipy.sparse import csr_array
from scipy.stats import iqr
from sklearn.covariance import GraphicalLassoCV, graphical_lasso
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

DATA_REPO_FS = DVCFileSystem()
GRAPH_ID_PATT = r"[a-z//]+/([a-z]{2})\w+/(N[0-9]{3}(?:[A-Z]{1}[0-9]+)+).json"
_dataset_paths = DATA_REPO_FS.find("/data", dvc_only=True, detail=False)


def _graph_path_to_ID(fp: str) -> str:
    rgx = re.search(GRAPH_ID_PATT, fp)
    if rgx:
        return str.swapcase(rgx.group(1)) + "-" + rgx.group(2)
    else:
        return ''


# TODO none-aware
_datasets = dict(
    zip(
        map(_graph_path_to_ID, _dataset_paths),
        map(Path, _dataset_paths),
    )
)



# DatasetIDType = Annotated[str, Is[lambda id: id in _datasets]]
DatasetIDType = Literal[tuple(_datasets)]


@beartype
def load_graph(graph_id: DatasetIDType) -> SerialRandWalks:
    # print(graph_id)
    return from_json(SerialRandWalks, DATA_REPO_FS.read_text(_datasets[graph_id]))


_estimators = Registry(prefix="_alg_", hyphen=True, case_sensitive=True)


@_estimators(aliases=["CoOc", 'cooccur-prob'])
def _alg_cooccurrence_probability(X, **kws):
    return _sq(asc.coocur_prob(X, **kws))


@_estimators(aliases=["CS", "cosine"])
def _alg_cosine_similarity(X, **kws):
    return _sq(asc.ochiai(X, **kws))


@_estimators(aliases=["RP"])
def _alg_resource_projection(X, **kws):
    return _sq(asc.resource_project(X, **kws))


@_estimators(aliases=["GL", "glasso"])
@ignore_warnings(category=ConvergenceWarning)
def _alg_graphical_lasso(X, **kws):
    try:
        # return -_sq(graphical_lasso(asc.coocur_prob(X, **kws), 0.05)[1])
        return -_sq(
            GraphicalLassoCV(assume_centered=True)
            .fit(X.toarray())  # TODO memory footprint :(
            .get_precision()
        )
    except FloatingPointError:
        return None
        # m = np.zeros(X.shape[1],X.shape[1])
        # m.fill(np.nan)
        # return _sq(m)


@_estimators(aliases=["HSS"])
def _alg_high_salience_skeleton(X, **kws):
    return _sq(asc.high_salience_skeleton(X, **kws))


@_estimators(aliases=["eOT", "sinkhorn"])
def _alg_entropic_optimal_transport(X, **kws):  # TODO allow generic prior alg, i guess?
    return _sq(sinkhorn(asc.ochiai(X, **kws)))


@_estimators(aliases=["MI"])
def _alg_mutual_information(X, **kws):
    return _sq(asc.mutual_information(X, **kws))


@_estimators(aliases=["TS", "tree-shift", "FP"])
def _alg_forest_pursuit(X, **kws):
    return _sq(asc.forest_pursuit_edge(X, **kws))


@_estimators(aliases=["FPi", "TSi"])
def _alg_forest_pursuit_interactions(X, **kws):
    return _sq(asc.forest_pursuit_interaction(X, **kws))


@_estimators(aliases=["EFM","expected-forest"])
def _alg_expected_forest_max(X, **kws):
    return _sq(asc.expected_forest_maximization(X, **kws))


# EstimatorNameType = Annotated[str, Is[lambda s: s in list(_estimators)]]
EstimatorNameType = Literal[tuple(_estimators)]

_metrics = Registry(prefix="_met_", hyphen=True, case_sensitive=True)


@_metrics(aliases=["F1"])
def _met_exp_f_score(M, **kws):
    if M is None:
        return np.nan
    return trapezoid(m.F1(M), x=M.weights)


@_metrics(aliases=["F-M", "fowlkes-mallows"])
def _met_exp_fowlkes_mallows_score(M, **kws):
    if M is None:
        return np.nan
    return trapezoid(m.fowlkes_mallows(M), x=M.weights)


@_metrics(aliases=["MCC", "matthews-corrcoef"])
def _met_exp_matthews_correllation_coefficient(M, **kws):
    if M is None:
        return np.nan
    return trapezoid(m.matthews_corrcoef(M), x=M.weights)


@_metrics(aliases=["APS", "expected-precision"])
def _met_avg_precision_score(M, **kws):
    if M is None:
        return np.nan
    return np.sum(np.diff(M.recall[::-1], prepend=0) * M.precision[::-1])


# MetricNameType = Annotated[str, Is[lambda s: s in list(_metrics)]]
MetricNameType = Literal[tuple(_metrics)]


def sigfigs(n,sig):
    # return '{:g}'.format(float('{:.{p}g}'.format(n, p=sig)))
    return float('{:.{p}g}'.format(n, p=sig))
    
np_sigfig = np.frompyfunc(sigfigs, 2, 1)

@beartype
def report(
    dataset: DatasetIDType,
    estimator: EstimatorNameType,
    metrics: list[MetricNameType]=['MCC'],
    **est_kwds
):

    exp = load_graph(dataset)
    X = csr_array(exp.activations.to_array().to_scipy_sparse())
    gT = _sq(exp.graph.to_array().todense()).astype(bool)

    node_cts = X.sum(axis=0)
    actv_cts = X.sum(axis=1)

    res=dict()
    res['ID']=dataset
    res['kind']=dataset[:2]
    res['n-edges']=gT.sum()
    res['n-nodes']=exp.graph.shape[0]
    res['n-walks']=exp.jumps.shape[0]
    res['n-jumps']=exp.jumps.shape[1]

    res['med-node-ct'] = np.median(node_cts)
    res['iqr-node-ct'] = iqr(node_cts)
    res['med-actv-ct'] = np.median(actv_cts)
    res['iqr-actv-ct'] = iqr(actv_cts)

    method = dict()
    method['name']=estimator
    # yappi.clear_stats()
    # yappi.start()
    # start=yappi.get_clock_time()
    start = time.time()
    gP = _estimators[estimator](X, **est_kwds)
    end = time.time()
    # end = yappi.get_clock_time()
    # yappi.stop()
    method['seconds']= sigfigs(end - start,5)
    method['estimate'] = np_sigfig(gP, 5).astype(float) if gP else None
    M = m.Contingent.from_scalar(gT, gP)
        
    scores = {met:sigfigs(_metrics[met](M),5) for met in metrics}   

    return res | method | scores
    # return alg(X, *args, **kwds)
    

# def eval_estimator(y_pred, y_true)-> dict:  # -> MetricStuff typeddiexpectedct?
#     # ugh
#     p,r,t = precision_recall_curve(y_true, y_pred, drop_intermediate=True)


def run_experiment(graph_id: DatasetIDType, algs: list[EstimatorNameType]) -> None:
    with Live(f"reports/{graph_id}", report="md", exp_name=graph_id) as live:
        # expected# prelim exp stuff
        exp = load_graph(graph_id)
        y_true = exp.graph.to_array()  # TODO FLATTEN ME!!
        for alg in algs:
            # per-alg-stuff
            pred = estimate_graph(exp.activations, alg)  # TODO handle args/kwargs
            # metrics = eval_estimator(pred, ytrue)
            M = m.Contingent.from_scalar(y_true, pred)

            for metric, val in metrics.items():
                live.log_metric(f"{alg}/{metric}", val)  # nested??
        # post-exp stuff
    ...
