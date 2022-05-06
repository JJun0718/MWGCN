
# MWGCN

This repository is the source code of paper: "MWGCN: predicting traditional Chinese medicine formula efficacy with graph convolutional network combining multilayer feature fusion and network weighting"

## Folders and files:

`/data` contains the raw data file used in this paper. The three excel files 'formulae.xlsx', 'herb_dosage.xlsx' and 'herb_trait.xlsx' in the folder '/data/origin' are part of the raw data used in the paper.

`/Data_Process` contains the data partitioning and graph construction.

`/result` contains the output results

`/result_plt` contains visualizations of results

`/utils` contains the extraction of F-H information (such as adjacency matrix, node features)

train.py is a training file for MWGCN

## Require

Python 3.6

Pytorch >= 1.6.0


## Preparing your own data

`/Data_Process/standardfile.py` is an example for preparing your own data.

## Train and evaluate

1. (i) cd Data_Process (ii) Run `python standardfile.py` (iii) Run `python herb_herb_graph.py` (iv) Run `python formula_herb_graph.py`

2. (i) cd .. (ii) Run `python train.py`