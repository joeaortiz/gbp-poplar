# Poplar Implementation of Bundle Adjustment using Gaussian Belief Propagation on Graphcore's IPU

Implementation of CVPR 2020 paper: [Bundle Adjustment on a Graph Processor](https://arxiv.org/abs/2003.03134)

Find the Poplar SDK documentation [here](https://www.graphcore.ai/developer-documentation).

Find the corresponding python implementation [here](https://github.com/joeaortiz/gbp).

## Running Bundle Adjustment

```
cd ba
make ba
./ba --bal_file ../sequences/fr1xyz.txt
```

For more options

 ```
 ./ba --help
 ```

## Running SLAM

```
cd ba
make slam
./slam --bal_file ../sequences/fr2robot2.txt 
```

### Citation

If you find our work useful in your research, please consider citing:

```
@InProceedings{OrtizCVPR2020,
author = {Ortiz, Joseph and Pupilli, Mark and Leutenegger, Stefan and Davison, Andrew J.},
title = {Bundle Adjustment on a Graph Processor},
booktitle = {{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
year = {2020}
}
```
