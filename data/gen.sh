#!/usr/bin/env sh

~/.cargo/bin/rust-parallel -p -s \
    -r '(?P<kind>.*) (?P<size>.*) (?P<seed>.*)' \
    "mendr-sim random-graph-walks {kind} {size} --seed {seed} > data/{kind}/N{size}S{seed}.json" \
    ::: tree block scalefree\
    ::: 010 030 100 300\
    ::: {01..30}
