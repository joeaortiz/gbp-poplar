# Poplar Implementation of Bundle Adjustment using Gaussian Belief Propagation on Graphcore's IPU

Implementation of CVPR 2020 paper: [Bundle Adjustment on a Graph Processor](https://arxiv.org/abs/2003.03134)

Find the corresponding python implementation [here](https://github.com/joeaortiz/gbp).

## Running Bundle Adjustment

```
cd ba
make
./ba --bal_file ../sequences/fr1xyz.txt
```

For more options

 ```
 ./ba --help
 ```
