from autoregistry import Registry
import time
from typing import Literal, Annotated, TypedDict
from dvc.api import DVCFileSystem
import re
from pathlib import Path
from beartype import beartype
from beartype.vale import Is
from serde.json import from_json

from affinis import associations as asc
from affinis.associations import _spanning_forests_obs_bootstrap
from affinis.utils import _sq
from affinis.proximity import sinkhorn

from mendr.io import SerialRandWalks#, SerialSparse
import mendr.metrics as m

import numpy as np

from scipy.integrate import trapezoid
from scipy.sparse import csr_array,coo_array
from scipy.stats import iqr
from sklearn.covariance import GraphicalLassoCV#, graphical_lasso
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

DATA_REPO_FS = DVCFileSystem()
GRAPH_ID_PATT = r"[a-z//]+/([a-z]{2})\w+/(N[0-9]{3}(?:[A-Z]{1}[0-9]+)+).json"
_dataset_paths = DATA_REPO_FS.find("/data/sets", dvc_only=True, detail=False)


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



DatasetIDType = Annotated[str, Is[lambda id: id in _datasets]]
# DatasetIDType = Literal[tuple(_datasets)]


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

@_estimators(aliases=["HYP", "hyperbolic-projection"])
def _alg_hyperbolic_project(X, **kws):
    w_hyp = 1/(X.sum(axis=1)-1)
    return _sq(((X.T*w_hyp)@X).toarray()) 

EstimatorNameType = Annotated[str, Is[lambda s: s in list(_estimators)]]
# EstimatorNameType = Literal[tuple(_estimators)]

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
    return m.avg_precision_score(M)


MetricNameType = Annotated[str, Is[lambda s: s in list(_metrics)]]
# MetricNameType = Literal[tuple(_metrics)]


def sigfigs(n,sig):
    # return '{:g}'.format(float('{:.{p}g}'.format(n, p=sig)))
    return float('{:.{p}g}'.format(n, p=sig))
    
np_sigfig = np.frompyfunc(sigfigs, 2, 1)

@beartype
def report(
    dataset: DatasetIDType,
    estimator: EstimatorNameType,
    metrics: list[MetricNameType]=['MCC'],
    preproc: Literal['forest']|None=None,
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
    if preproc=='forest':
        res['name'] = estimator+'-FP'
        E_obs = _spanning_forests_obs_bootstrap(X)
        A_FP = _sq(asc.forest_pursuit_edge(X))
        n1, n2 = np.triu(_sq(A_FP)).nonzero()
        # print(n1.shape)
        e = np.ma.nonzero(A_FP)[0]
        B = coo_array((np.concatenate([A_FP, -A_FP]), (np.concatenate([e,e]),np.concatenate([n1,n2]))), shape=(e.shape[0], X.shape[1]))

        # np.diag((B.T@B).toarray())==np.diag(nx.laplacian_matrix(G).toarray()).round(1)
        X=(E_obs@(np.abs(B)))

    start = time.time()
    gP = _estimators[estimator](X, **est_kwds)
    end = time.time()
    # end = yappi.get_clock_time()
    # yappi.stop()
    method['seconds']= sigfigs(end - start,5)
    method['estimate'] = np_sigfig(gP, 5).astype(float) if gP is not None else None
    M = m.Contingent.from_scalar(gT, gP)
        
    scores = {met:sigfigs(_metrics[met](M),5) for met in metrics}   

    return res | method | scores
    # return alg(X, *args, **kwds)
   
