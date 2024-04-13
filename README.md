# Reproduction: Self-Supervised Decoufounding Against Spatio-Temporal Shifts: Theory and Modeling
================================================

## Introduction

This repository aims for reproducing the paper:

`Ji, Jiahao, et al. "Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling." arXiv preprint arXiv:2311.12472 (2023).`

The process of reproduction includes:

* **Data Preprocessing**: Since the data provided by the original repository is not enough for reproducing the OOD case in different scenario, this repository provides a data preprocess script for generating data needed.
* **Ablation Study**: Six variants of ablation study mentioned in the paper is conducted.

## Quick Start

### Data Preprocessing

For data preprocessing, simply run the script:
```
python data_preprocess.py
```
the output is composed of the following files : 
```
|----NYCBike1\
|   |----train.npz          # training data
|   |----adj_mx.npz         # predefined graph adjacency matrix
|   |----total_test.npz     # test data (total)
|   |----workday_test.npz   # test data (workday)
|   |----holiday_test.npz   # test data (holiday)
|   |----val.npz            # validation data
```
For each of the `npz` file, several components are included:

* `x`: input data, which is a 4D tensor which indicates: `sample size`, `window size`, `num of nodes`, `flow features`.
* `y`: output data, which is a 4D tensor which indicates: `sample size`, `predicted length`, `num of nodes`, `flow features`.
* `time_label`: indicating the temporal type for corresponding output data. The label is ranging from 0 to 47, where 0-23 indicates hours in workdays, whereas 24-47 indicates hours in holidays
