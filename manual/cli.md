# `mendr` CLI Tools
> you don't really want to leave the terminal, do you? 

## `mendr-sim`
MENDR comes with a CLI tool for consistent "problem" generation. 
This let's the community quickly create their own problems, with settings that are useful for their setting, while still doing so in a consistent way. 

```sh
󱞩 mendr-sim
Usage: mendr-sim COMMAND

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

```sh
󱞩 mendr-sim random-graph  -h
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

Similarly, you can create a graph and a set of random-walks on that graph in one shot: 

```sh
󱞩 mendr-sim random-graph-walks  -h
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

