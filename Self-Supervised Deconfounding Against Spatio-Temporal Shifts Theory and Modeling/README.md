# Reproduction

## Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling

### Introduction

The following content aims for reproducing the paper:

`Ji, Jiahao, et al. "Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling." arXiv preprint arXiv:2311.12472 (2023).`

The process of reproduction includes:

* **Data Preprocessing**: Since the data provided by the original repository is not enough for reproducing the OOD case in different scenario, this repository provides a data preprocess script for generating data needed.
* **OOD evaluation**: spatial OOD evaluation and temporal OOD evaluation are conducted to discover the model's generalization ability.
* **Ablation Study**: Six variants of ablation study mentioned in the paper is conducted.
 
### Quick Start

*Note*: all the commands are run under `STEVE` directory.

#### Data Preprocessing

For data preprocessing, download the raw data then simply run the script:
```
python data_preprocess.py
```
*Input*: raw `NYCBike1` data file, which could be downloaded from [here](https://github.com/topazape/ST-ResNet/tree/main/datasets/BikeNYC). For alignment with the script, just place the data under `STEVE` directory.

*Output*:
```
|----NYCBike1\
|   |----train.npz                  # training data
|   |----val.npz                    # validation data
|   |----test.npz                   # test data
|   |----total_test_dataloader.pkl  # test dataloader (total)
|   |----workday_test_dataloder.pkl # temporal OOD data (workday)
|   |----holiday_test.pkl           # temporal OOD data (holiday)
|   |----cluste_labels.npy          # recording the spatial clusters for spatial OOD forecasting

```
For each of the `npz` file, several components are included:

* **x**: input data, which is a 4D tensor which indicates: `sample size`, `window size`, `num of nodes`, `flow features`.
* **y**: output data, which is a 4D tensor which indicates: `sample size`, `predicted length`, `num of nodes`, `flow features`.
* **time_label**: indicating the temporal type for corresponding output data. The label is ranging from 0 to 47, where 0-23 indicates hours in workdays, whereas 24-47 indicates hours in holidays.
* **c**: indicating the traffic load capacity in self-supervised task 3 in the paper, having the same dimension with `y`.

For each of the `pkl` file, two components are included:

* `dataloader`: used in test phase for generating different types of OOD data.
* `scaler`: used in test phase for data inverse transformation. 

#### OOD evaluation:
After doing data preprocess, data needed for temporal OOD forecasting and spatial OOD forecasting is obatained. Before evaluation, please train the whole model with full training data. Just set the configuration of `mode` as `train` and other modules like `cd`, `gr`, `sl`, `ti`, `tl` to be `true` from `configs/NYCBike1.yaml`, then simply run the script:
```
python run.py
```
After training, just easily configure `mode` as `test` and run the above script again, the output is saved in `experiments/NYCBike1/results.npz` file, which includes:
* **Temporal OOD Forecasting**: prediction and ground truth of the workday data and holiday data.
* **Spatial OOD Forecasting**: prediction and ground truth of respective spatial clusters data.
* **OOD Evaluation Result**: `mae` and `mape` of all the aspects mentioned above.

#### Ablation Study:
When doing ablation study, please configure the module needed to be verified in configuration file `configs/NYCBike1.yaml`. Every configuration is a boolean type with `True` for implementation and `False` for ignorant.

There are six variants of ablation study mentioned in the paper:
* **cd**: Contextual Disentanglement Module
* **gr**: Gradient Reversal Layer Implementation
* **idp**: Independent Requirement of DCA
* **sl**: Spatial Location Classification Self-supervised Task
* **ti**: Temporal Index Identification Self-supervised Task
* **tl**: Traffic Load Prediction Self-supervised Task

When doing ablation study for a specific variant, just set the corresponding configuration as `False` and train the model. Note that when doing `idp`, you need to set both `cd` and `gr` as `False`.

### Result

#### OOD Evaluation
* Temporal OOD Forecasting

    | Metric | Workday | Holiday | Total |
    |--------|---------|---------|-------|
    | MAE    |  Value  |  Value  | Value |
    | MAPE   |  Value  |  Value  | Value |

* Spatial OOD Forecasting

    | Metric | c0 | c1 | c2 | Total |
    |--------|---------|---------|-------|------|
    | MAE    |  Value  |  Value  | Value | Value |
    | MAPE   |  Value  |  Value  | Value | Value |

#### Ablation Study
* Temporal OOD Forecasting

    | Metric | STEVE | *w/o cd* | *w/o gr* | *w/o idp* | *w/o sl* | *w/0 ti* | *w/o tl*|
    |--------|---------|---------|---------|-----------|----------|----------|---------|
    | MAE    |  Value  |  Value  | Value |Value |Value |Value |Value |
    | MAPE   |  Value  |  Value  | Value |Value |Value |Value |Value |


* Spatial OOD Forecasting

    | Metric | STEVE | *w/o cd* | *w/o gr* | *w/o idp* | *w/o sl* | *w/0 ti* | *w/o tl*|
    |--------|---------|---------|---------|-----------|----------|----------|---------|
    | MAE    |  Value  |  Value  | Value |Value |Value |Value |Value |
    | MAPE   |  Value  |  Value  | Value |Value |Value |Value |Value |