### Phase A Step D, the gap column corrected

The old column 'gap larger than the band?' compared the oracle-minus-script gap with the +/-1-voxel band on the oracle's ABSOLUTE LEVEL. A paired difference does not inherit a level's uncertainty (the same mask shift moves both arms), so that column answered 'no' almost everywhere and contradicted the narrative drawn from the next column. It is replaced by: does the gap keep its sign and its order across the nominal, eroded and dilated masks?

Does the paired gap (S' planned for T, scored at T, minus S at T; plain coverage) keep its sign and its order (max/min of the three magnitudes <= 3) across the nominal, eroded and dilated masks?

#### low k (1.2e-4 /s)

| T | gap, nominal | gap, eroded | gap, dilated | keeps sign and order? | why |
|---|---|---|---|---|---|
| 1 | +0.076 | +0.035 | +0.090 | **yes** | keeps its sign and its order |
| 10 | +0.137 | +0.085 | +0.131 | **yes** | keeps its sign and its order |
| 100 | +0.179 | +0.150 | +0.132 | **yes** | keeps its sign and its order |
| 1000 | +0.204 | +0.239 | +0.132 | **yes** | keeps its sign and its order |
| 10000 | +0.120 | +0.217 | +0.062 | **no** | changes order (max/min = 3.5 > 3) |
| 100000 | +0.019 | +0.055 | +0.010 | **no** | changes order (max/min = 5.5 > 3) |
| 1e+06 | +0.003 | +0.007 | +0.001 | **no** | changes order (max/min = 4.4 > 3) |

#### high k (6.0e-4 /s)

| T | gap, nominal | gap, eroded | gap, dilated | keeps sign and order? | why |
|---|---|---|---|---|---|
| 1 | +0.243 | +0.227 | +0.211 | **yes** | keeps its sign and its order |
| 10 | +0.283 | +0.288 | +0.214 | **yes** | keeps its sign and its order |
| 100 | +0.238 | +0.292 | +0.152 | **yes** | keeps its sign and its order |
| 1000 | +0.137 | +0.246 | +0.075 | **no** | changes order (max/min = 3.3 > 3) |
| 10000 | +0.037 | +0.100 | +0.019 | **no** | changes order (max/min = 5.4 > 3) |
| 100000 | +0.007 | +0.020 | +0.004 | **no** | changes order (max/min = 5.2 > 3) |
| 1e+06 | +0.001 | +0.003 | +0.001 | **no** | not resolved (all three gaps below the floor) |
