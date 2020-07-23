# PPRGo (PyTorch)

This repository provides a PyTorch implementation of PPRGo for a single machine. You can find the [original TensorFlow 1 implementation in another repository](https://github.com/TUM-DAML/pprgo). PPRGo is a fast GNN able to scale to massive graphs in both single-machine and distributed setups. It was proposed in our paper

**[Scaling Graph Neural Networks with Approximate PageRank](https://www.daml.in.tum.de/pprgo)**   
by Aleksandar Bojchevski*, Johannes Klicpera*, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann  
Published at ACM SIGKDD 2020.

## Demonstration
To see for yourself how fast PPRGo runs even on a large dataset we've set up a [Google Colab notebook](https://colab.research.google.com/drive/1nw3MIpXPK_n6IZvKcLxkOuOi9_1i6IzA?usp=sharing), which trains and generates predictions for the Reddit dataset, as described in the paper.

## Installation
You can install the repository using `pip install -e .`. Since CUDA 10.0 includes a bug that affects PPRGo we strongly recommend using e.g. 10.1.

## Run the code
This repository contains a demo notebook for running training and inference (`demo.ipynb`) and a script for running the model on a cluster with [SEML](https://github.com/TUM-DAML/seml) (`run_seml.py`).

## Contact
Please contact a.bojchevski@in.tum.de or klicpera@in.tum.de if you have any questions.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{bojchevski2020pprgo,
  title={Scaling Graph Neural Networks with Approximate PageRank},
  author={Bojchevski, Aleksandar and Klicpera, Johannes and Perozzi, Bryan and Kapoor, Amol and Blais, Martin and R{\'o}zemberczki, Benedek and Lukasik, Michal and G{\"u}nnemann, Stephan},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2020},
  publisher = {ACM},
  address = {New York, NY, USA},
}
```
