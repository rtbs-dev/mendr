from .generate import RandGraphType, graph_gen, walk_randomly
from .io import SerialSparse, SerialRandWalks
from .one_hot import rw_jumps_to_coords
from .experiments import DatasetIDType, EstimatorNameType, MetricNameType, load_graph, _datasets, report, _metrics

from tqdm import tqdm

import numpy as np
from serde.json import to_json
from cyclopts import App
from scipy.stats import iqr
# from beartype import beartype
import csrgraph as cg
from typing import Literal

app = App()
app.command(mendr_sim := App(name="sim"))
# app.command(mendr_test := App(name="test"))


@mendr_sim.command
def random_graph(kind: RandGraphType, size: int, seed: int | None = None):
    """Generate a random graph and send a json representation to stdout."""
    RNG = np.random.default_rng(seed)
    g = graph_gen(kind, size, rng=RNG)
    print(to_json(SerialSparse.from_array(g)))


@mendr_sim.command
def random_graph_walks(
    kind: RandGraphType,
    size: int,
    n_walks: int | None = None,
    n_jumps: int | None = None,
    seed: int | None = None,
):
    """Generate a random graph and sample random walks on it.

    Send both as json to stdout.
    """
    RNG = np.random.default_rng(seed)

    g = cg.csrgraph(graph_gen(kind, size, rng=RNG))
    rw = walk_randomly(g, n_jumps, n_walks, rng=RNG)
    activations = rw_jumps_to_coords(rw, num_nodes=size)

    # print(to_json([SerialSparse.from_array(g), SerialSparse.from_array(activations)]))
    experiment = SerialRandWalks(
        SerialSparse.from_array(g), rw, SerialSparse.from_array(activations)
    )
    print(to_json(experiment))


DEFAULT_ALGS = ['FP','FPi','CoOc','CS','MI','eOT','HSS', 'GL', 'RP']
DEFAULT_METRICS = ['F1','F-M','MCC','APS']

@app.command
def recovery_test(
    method: EstimatorNameType,
    datasets: list[DatasetIDType]=list(_datasets),
    metrics: list[MetricNameType]=DEFAULT_METRICS,
    preprocess: Literal["forest"]|None=None,
    **alg_kws: dict|None
):
    """Run an algorithm through the MENDR datasets
    
    Send result report for each dataset as JSONL to stdout    
    """
    progress = tqdm(datasets)
    for dataset in progress: 
        progress.set_description(f"Graph Dataset {dataset}")
        print(to_json(report(dataset, method,metrics,preproc=preprocess,**alg_kws )))
    

# def main():
#     app()


if __name__ == "__main__":
    app()
