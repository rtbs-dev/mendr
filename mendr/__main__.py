from .generate import RandGraphType, graph_gen, walk_randomly
from .io import SerialSparse, SerialRandWalks
from .one_hot import rw_jumps_to_coords
from .experiments import DatasetIDType, EstimatorNameType, MetricNameType, load_graph, _datasets

import numpy as np
from serde.json import to_json
from cyclopts import App

# from beartype import beartype
import csrgraph as cg

app = App()
app.command(mendr_sim := App(name="sim"))
app.command(mendr_test := App(name="test"))


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

@test.command
def recovery_algorithm(
    alg: EstimatorNameType,
    datasets: list[DatasetIDType]=_datasets
):
    ...

# def main():
#     app()


if __name__ == "__main__":
    app()
