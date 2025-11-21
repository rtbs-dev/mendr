# Dataset Overview

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


