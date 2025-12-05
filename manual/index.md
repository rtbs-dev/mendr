# MENDR
> **M**easurement **E**rror in **N**etwork **D**iffusion **D**atasets for (graph) **R**econstruction

MENDR is a benchmarking data and results registry, along with tooling for accurate (de)serialization and validation of each challenge problem. 

## Quick-links

```{tableofcontents}
```

## Leaderboard
Current performance leaderboard for various methods on the MENDR challenge datasets, as originally reported in {cite}`sexton2025measuring`.

![table](https://dissertation.rtbs.dev/content/part2/2-05-forest-pursuit.html#tbl-fp)

## What does it help with?

Recovering graph structure from binary node activations is a difficult problem, especially when that data is generated from diffusion-like processes (say, random-walks). 

Because in situ graph recovery rarely has ground-truth (similar to clustering and outlier detection... these are treated as _unsupervised_), verification of algorithm performance is usually impossible. 
Validation, while possible through generation of synthetic datasets, is rarely performed. 

MENDR tries to provide the latter: _algorithm validation_ for network reconstruction from random walks. 

## What's inside?

### Datasets
Each MENDR "problem" consists of a randomly generated network, and a unique set of random-walks for each. 
The problems are given an ID based on their generation parameters. 
For more information, see the [`mendr` CLI page](cli.md)

### Data VCS 
TO facilitate community a


