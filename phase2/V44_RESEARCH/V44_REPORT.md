# V44-A GLOBAL GEOMETRY REPORT

## Objective
Test whether global layout features and candidate constellation geometry can disambiguate the true placement from a false periodic replica. 

## Dataset
53 eligible Group A pairs (V25 Top-1 is WRONG, but GT is correctly present in the Top-20 pool).

## Results: Pairwise GT vs WRONG
We define a "win" as the GT candidate scoring strictly higher (or lower, depending on the natural direction) than the False Top-1.

| Feature | GT Win Rate | Direction for GT Win | Ties |
|---|---|---|---|
| center_distance | **64.2%** | LESS | 0.0% |
| min_boundary_distance | **62.3%** | GREATER | 1.9% |
| constellation_asymmetry | 58.5% | GREATER | 1.9% |
| 3rd_neighbor_distance | 58.5% | GREATER | 17.0% |
| 2nd_neighbor_distance | 58.5% | GREATER | 18.9% |
| 
eighbor_count_50 | 50.9% | LESS | 17.0% |

## Verdict
The GT win rates are hovering around **58–64%**, which places us squarely in the >50% (interesting) category, just barely missing the >65% (STOP EVERYTHING) hard gate for center_distance. 

The global geometry *is* providing new information. The true candidate is generally closer to the center, further from the absolute image boundaries, and sits in a slightly more asymmetric local candidate constellation than the false periodic replica. 
