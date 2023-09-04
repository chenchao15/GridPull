<p align="center">
  <h1 align="center">GridPull: Towards Scalability in Learning Implicit Representations from 3D Point Clouds</h1>
  <p align="center">
    <a href="https://chenchao15.github.io/"><strong>Chao Chen</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>


  </p>
  <h2 align="center">ICCV 2023</h2>
  <h3 align="center"><a href="https://arxiv.org/pdf/2308.13175.pdf">Paper</a> | <a href="https://chenchao15.github.io/GridPull">Project Page</a></h3>
  <div align="center"></div>

</p>

## Citation

If you find our code or paper useful, please consider citing

    @inproceedings{chao2023gridpull,
        title = {GridPull: Towards Scalability in Learning Implicit Representations from 3D Point Clouds},
        author = {Chao Chen and Yu-Shen Liu and Zhizhong Han},
        booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
        year = {2023}
    }

## Overview
<p align="center">
  <img src="figs/overviews.png" width="780" />
</p>


We propose GridPull to speed up the learning of implicit function from large scale point clouds. GridPull does not require learned priors or point normal, and directly infers a distance field from a point cloud without using any neural components. We infer the distance field on grids near the surface, which reduces the number of grids we need to infer. Our loss function encourages continuous distances and consistent gradients in the field, which makes up the lack of continuousness brought by neural networks.

## Demo Results
### ShapeNet
<p align="center">
  <img src="figs/shapenet.png" width="780" />
</p>


### FAMOUS
<p align="center">
  <img src="figs/famous.png" width="780" />
</p>

### SRB

<p align="center">
  <img src="figs/srb.png" width="760" />
</p>

### SceneNet

<p align="center">
  <img src="figs/scenenet.png" width="760" />
</p>

### SceneNet

<p align="center">
  <img src="figs/scenenet.png" width="760" />
</p>

### 3DFront

<p align="center">
  <img src="figs/3dfront.png" width="760" />
</p>



### mattport3d

<p align="center">
  <img src="figs/mattport3d.png" width="760" />
</p>

## Installation

Coming soon

## Dataset
## Train
## Test
