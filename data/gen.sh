#!/usr/bin/env sh

~/.cargo/bin/rust-parallel -p -s \
    -r '(?P<kind>.*) (?P<size>.*) (?P<seed>.*)' \
    "gmd-nist random-graph-walks {kind} {size} --seed {seed}
              > data/{kind}/N{size}S{seed}.json" \
    ::: tree block \
    ::: 010 030 100 \
    ::: {01..10}
