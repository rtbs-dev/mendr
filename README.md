# MENDR
> **M**easurement **E**rror in **N**etwork **D**iffusion & **R**econstruction

MENDR is a benchmarking data and results registry, along with tooling for accurate (de)serialization and validation of each challenge problem. 

## Installing

`mendr` is currently un-published. 
Reference installations can be achieved for development purposes with 

```
pip install git+https://github.com/usnistgov/mendr.git
```


# Dataset Overview
The MENDR dataset contains several thousand challenge datasets, where the goal is to recover a graph structure from a set of random-walk visitations.
If a node is considered "active" when a random walk has visited it, we can create a binary node activation vector from the set of visited nodes.
Then, running many random walks on a graph will produce a binary node "activation" matrix.

Every graph+activation pair has a unique ID in MENDR.
The id will start with its code:

BL
: block network

TR
: Tree network

SC
: Scale-free (Barabasi-Albert)

This is followed by a code containing the number of nodes and the seed that generated the random sample, e.g. `BL-N030S01`.

Other parameters are sampled randomly for each combination of the above, as detailed in the folowing table:
 

|parameters             | values                           |
| :-                    |:-:                               | 
| random graph **kind** | Tree, Block, BA$(m\in\{1,2\}$)                |
| network **$n$-nodes** | 10,30,100,300                    |
| random **walks**      | 1 sample $m\sim\text{NegBinomial}(2,\tfrac{1}{n})+10$|
| random walk **jumps** | 1 sample $j\sim\text{Geometric}(\tfrac{1}{n})+5$     | 
| random walk **root**  | 1 sample $n_0 \sim \text{Multinomial}(\textbf{n},1)$|
| random **seed**       |  1, 2, ... ,  30                 |

: Experiment Settings (`MENDR` Dataset) {#tbl-mendr}


