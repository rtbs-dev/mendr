#!/usr/bin/env python3
from autoregistry import Registry
from affinis import associations
from typing import Literal, Annotated, TypedDict
from dvc.api import DVCFileSystem
import re
from pathlib import Path
from beartype import beartype
from beartype.door import Is
from serde import from_json
from dvclive import Live
from .io import SerialRandWalks, SerialSparse
from sklearn.metrics import precision_recall_curve

DATA_REPO_FS = DVCFileSystem()
GRAPH_ID_PATT = '[a-z//]+(N[0-9]{3}(?:[A-Z]{1}[0-9]+)+).json'
_estimators = Registry(associations, hyphen=True)
_dataset_paths = DATA_REPO_FS.find('data', dvc_only=True, detail=False)
#TODO none-aware
_datasets = dict(zip(map(lambda s: re.search(GRAPH_ID_PATT,s).group(1), _dataset_paths), map(Path, _dataset_paths)))

#TODO make a better estimator registry with aliases and sinkhorn..ochiai, etc. 
EstimatorNameType = Annotated[str, Is[lambda s: s in list(_estimators)]]
DatasetIDType = Annotated[str, Is[_datasets.has_key]]

class MetricsType(TypedDict): 
    hellinger: float
    f_beta: float
    mcorr: float
    



# ValidEstimators =
@beartype
def load_graph(graph_id:DatasetIDType)->SerialRandWalks:
    # print(graph_id)
    return from_json(DATA_REPO_FS.read(_datasets[graph_id]))

@beartype
def estimate_graph(rw:SerialSparse,estimator:EstimatorNameType, *args, pseudocts='min-connect', **kwds):
    X = rw.to_array()
    # A = experiment.graph.to_arrayy()
    alg = _estimators[estimator]
    return alg(X, *args, pseudocts=pseudocts, **kwds)





def eval_estimator(y_pred, y_true)-> dict:  # -> MetricStuff typeddiexpectedct?
    # ugh 
    p,r,t = precision_recall_curve(y_true, y_pred, drop_intermediate=True)
    

def run_experiment(graph_id:DatasetIDType, algs:list[EstimatorNameType])->None:
    with Live(f'reports/{graph_id}', report='md', exp_name=graph_id) as live:
        expected# prelim exp stuff
        exp = load_graph(graph_id)
        ytrue = exp.graph.to_array()  #TODO FLATTEN ME!!
        for alg in algs:
            # per-alg-stuff
            pred = estimate_graph(exp.activations, alg) #TODO handle args/kwargs
            metrics = eval_estimator(pred, ytrue)
            for (metric, val) in metrics.items():
                live.log_metric(f'{alg}/{metric}', val) # nested??
        # post-exp stuff
    ...
