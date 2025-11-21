# `mendr` CLI Tools
> you don't really want to leave the terminal, do you? 


```sh
$ mendr -h
Usage: mendr COMMAND

╭─ Commands ───────────────────────────────────────╮
│ recovery-test  Run an algorithm through the      │
│                MENDR datasets                    │
│ sim            Randomly generate a grame and     │
│                simulate a random-walk            │
│ --help,-h      Display this message and exit.    │
│ --version      Display application version.      │
╰──────────────────────────────────────────────────╯
```



## `sim`
MENDR comes with a CLI tool for consistent "problem" generation. 
This let's the community quickly create their own problems, with settings that are useful for their setting, while still doing so in a consistent way. 

```sh
$ mendr sim
Usage: mendr sim COMMAND

╭─ Commands ────────────────────────────────────────╮
│ random-graph        Generate a random graph and   │
│                     send a json representation to │
│                     stdout.                       │
│ random-graph-walks  Generate a random graph and   │
│                     sample random walks on it.    │
│ --help,-h           Display this message and      │
│                     exit.                         │
│ --version           Display application version.  │
╰───────────────────────────────────────────────────╯
```

The random graph generation tool is type-checked at runtime using Cyclopts and Beartype, so you never have to worry about miss-specified parameters (you'll be informed immediately!)

### `random-graph` 

```sh
$ mendr sim random-graph  -h
Usage: mendr-sim random-graph [ARGS] [OPTIONS]

Generate a random graph and send a json 
representation to stdout.

╭─ Parameters ──────────────────────────────────────╮
│ *  KIND,--kind  [choices: tree,block,scalefree]   │
│                 [required]                        │
│ *  SIZE,--size  [required]                        │
│    SEED,--seed                                    │
╰───────────────────────────────────────────────────╯
```

### `random-graph-walks`

Similarly, you can create a graph and a set of random-walks on that graph in one shot: 

```sh
$ mendr-sim random-graph-walks  -h
Usage: mendr-sim random-graph-walks [ARGS] [OPTIONS]

Generate a random graph and sample random walks on 
it.

Send both as json to stdout.

╭─ Parameters ──────────────────────────────────────╮
│ *  KIND,--kind        [choices:                   │
│                       tree,block,scalefree]       │
│                       [required]                  │
│ *  SIZE,--size        [required]                  │
│    N-WALKS,--n-walks                              │
│    N-JUMPS,--n-jumps                              │
│    SEED,--seed                                    │
╰───────────────────────────────────────────────────╯
```

```{note}
Like with the core MENDR dataset, leaving the optional `N-WALKS` and `N-JUMPS` args blank will use a random number of walks to sample, distributed as 
$\textrm{NegBinomial}(2, \frac{1}{n})+10$
where $n$ is the passed `SIZE` parameter. 
Similarly, `N-JUMPS` would be sampled from 
$\textrm{Geometric}(\frac{1}{n})+5$
which controls the number of jumps all random walkers will perform while activating nodes. 
```

## `recovery-test`

Automates the benchmarking of a MENDR algorithm against any number of the datasets provided.
Useful for CI/CD and reproducibility. 

```sh
mendr recovery-test -h
Usage: mendr recovery-test [ARGS] [OPTIONS]

Run an algorithm through the MENDR datasets

Send result report for each dataset as JSONL to stdout

╭─ Parameters ────────────────────────────────────────────────────╮
│ *  METHOD,--method                       [required]             │
│    DATASETS,--datasets,--empty-datasets  [default:              │
│                                          ['BL-N010S01',         │
│                                          'BL-N010S02',          │
│                                             ...                 │
│                                          'TR-N300S29',          │
│                                          'TR-N300S30']]         │
│    METRICS,--metrics,--empty-metrics     [default: ['F1',       │
│                                          'F-M', 'MCC', 'APS',   │
│                                          'MCC-max']]            │
│    PREPROCESS,--preprocess               [choices: forest]      │
│ *  --alg-kws                             [required]             │
╰─────────────────────────────────────────────────────────────────╯

```
