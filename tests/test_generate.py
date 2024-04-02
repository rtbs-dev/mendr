#!/usr/bin/env python3
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from serde.json import to_json, from_json
import sparse
from nist_gmd.one_hot import rw_jumps_to_coords
from nist_gmd.io import SerialSparse


make_jumps = hnp.arrays(
    int,
    hnp.array_shapes(max_dims=2, min_dims=2, min_side=5),
    elements=st.integers(min_value=0, max_value=1000),
)


@given(make_jumps)
def test_one_hot(rw):
    # print(rw.shape[0])
    rw_jumps_to_coords(rw)


# make_coords = hnp.arrays(
#     int,
#     (2, st.integers(min_value=1, max_value=100)),
#     elements = st.integers(min_value=0, max_value=100)
#
# rng = np.random.default_rng()


@given(hnp.array_shapes(max_dims=2, min_dims=2, min_side=10, max_side=100))
@settings(deadline=None)
def test_serialize_deserialize(shape):
    # s = sparse.COO(idx, data=1)
    s = sparse.random(shape)  # TODO COO.data is getting ignored intentionally
    s_io = SerialSparse.from_array(s)
    s_oi = from_json(SerialSparse, to_json(s_io)).to_array()

    assert np.allclose(s.coords, s_oi.coords)
    # assert np.allclose(s.data, s_oi.data)
