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
<table cellpadding=0 cellspacing=0 width=483 style='border-collapse:
 collapse;table-layout:fixed;width:364pt'>
 
 <tr height=19 style='height:14.0pt'>
  <td rowspan=2 height=38 class=xl66 width=69 style='height:28.0pt;width:52pt' align=center><strong>Metric</strong></td>
  <td rowspan=2 class=xl66 width=69 style='width:52pt' align=center><strong>Total</strong></td>
  <td colspan=2 class=xl65 width=138 style='width:104pt' align=center><strong>Temporal</strong></td>
  <td colspan=3 class=xl65 width=207 style='width:156pt' align=center><strong>Spatial</strong></td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td height=19 style='height:14.0pt' align=center>Workday</td>
  <td align=center>Holiday</td>
  <td align=center>c0</td>
  <td align=center>c1</td>
  <td align=center>c2</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td height=19 style='height:14.0pt' align=center>MAE</td>
  <td class=xl67 align=center>4.80</td>
  <td align=center>4.76</td>
  <td align=center>4.78</td>
  <td align=center>2.72</td>
  <td align=center>6.48</td>
  <td align=center>4.57</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td height=19 style='height:14.0pt' align=center>MAPE</td>
  <td align=center>22.64</td>
  <td align=center>21.23</td>
  <td class=xl67 align=center>23.40</td>
  <td align=center>22.82</td>
  <td align=center>21.73</td>
  <td align=center>23.14</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
 </tr>
 <![endif]>
</table>

#### Ablation Study

<table border=0 cellpadding=0 cellspacing=0>
 <col width=69 span=16 style='width:52pt'>
 <tr height=19 style='height:14.0pt'>
  <td colspan=2 height=19 class=xl66 width=138 style='height:14.0pt;width:104pt' align=center><Strong>Ablation</strong></td>
  <td colspan=2 class=xl66 width=138 style='width:104pt' align=center><strong>STEVE</strong></td>
  <td colspan=2 class=xl66 width=138 style='width:104pt' align=center><strong>w/o cd</strong></td>
  <td colspan=2 class=xl66 width=138 style='width:104pt' align=center><strong>w/o gr</strong><span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td colspan=2 class=xl66 width=138 style='width:104pt' align=center><strong>w/o idp</strong></td>
  <td colspan=2 class=xl66 width=138 style='width:104pt' align=center><strong>w/o sl</strong></td>
  <td colspan=2 class=xl66 width=138 style='width:104pt' align=center><strong>w/o ti</strong></td>
  <td colspan=2 class=xl66 width=138 style='width:104pt' align=center><strong>w/o tl</strong></td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td colspan=2 height=19 class=xl66 style='height:14.0pt' align=center><strong>Metric</strong></td>
  <td class=xl66 align=center>MAE</td>
  <td class=xl66 align=center>MAPE</td>
  <td class=xl66 align=center>MAE</td>
  <td class=xl66 align=center>MAPE</td>
  <td class=xl66 align=center>MAE</td>
  <td class=xl66 align=center>MAPE</td>
  <td class=xl66 align=center>MAE</td>
  <td class=xl66 align=center>MAPE</td>
  <td class=xl66 align=center>MAE</td>
  <td class=xl66 align=center>MAPE</td>
  <td class=xl66 align=center>MAE</td>
  <td class=xl66 align=center>MAPE</td>
  <td class=xl66 align=center>MAE</td>
  <td class=xl66 align=center>MAPE</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td rowspan=2 height=38 class=xl66 style='height:28.0pt' align=center><strong>Temporal</strong></td>
  <td class=xl66 align=center>Workday</td>
  <td class=xl66 align=center>4.76</td>
  <td class=xl66 align=center>21.23</td>
  <td class=xl66 align=center>4.64</td>
  <td class=xl66 align=center>20.34</td>
  <td class=xl66 align=center>4.57</td>
  <td class=xl66 align=center>20.48</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
  <td class=xl68 align=center>4.80</td>
  <td class=xl66 align=center>21.33</td>
  <td class=xl68 align=center>4.70</td>
  <td class=xl66 align=center>20.89</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td height=19 class=xl66 style='height:14.0pt' align=center>Holiday</td>
  <td class=xl66 align=center>4.78</td>
  <td class=xl68 align=center>23.40</td>
  <td class=xl66 align=center>4.97</td>
  <td class=xl66 align=center>22.74</td>
  <td class=xl66 align=center>4.51</td>
  <td class=xl66 align=center>22.26</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>4.66</td>
  <td class=xl66 align=center>22.85</td>
  <td class=xl66 align=center>4.72</td>
  <td class=xl66 align=center>22.87</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td rowspan=3 height=57 class=xl66 style='height:42.0pt' align=center><strong>Spatial</strong></td>
  <td class=xl66 align=center>c0</td>
  <td class=xl66 align=center>2.72</td>
  <td class=xl66 align=center>22.82</td>
  <td class=xl66 align=center>2.84</td>
  <td class=xl66 align=center>23.47</td>
  <td class=xl66 align=center>2.67</td>
  <td class=xl66 align=center>22.55</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>2.74</td>
  <td class=xl68 align=center>22.80</td>
  <td class=xl66 align=center>2.74</td>
  <td class=xl66 align=center>22.72</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td height=19 class=xl66 style='height:14.0pt' align=center>c1</td>
  <td class=xl66 align=center>6.48</td>
  <td class=xl66 align=center>21.73</td>
  <td class=xl66 align=center>6.36</td>
  <td class=xl66 align=center>20.88</td>
  <td class=xl66 align=center>6.08</td>
  <td class=xl66 align=center>20.05</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>6.53</td>
  <td class=xl66 align=center>21.82</td>
  <td class=xl66 align=center>6.37</td>
  <td class=xl68 align=center>21.00</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td height=19 class=xl66 style='height:14.0pt' align=center>c2</td>
  <td class=xl66 align=center>4.57</td>
  <td class=xl66 align=center>23.14</td>
  <td class=xl68 align=center>4.60</td>
  <td class=xl66 align=center>23.61</td>
  <td class=xl66 align=center>4.38</td>
  <td class=xl66 align=center>22.08</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>4.59</td>
  <td class=xl66 align=center>23.07</td>
  <td class=xl66 align=center>4.56</td>
  <td class=xl66 align=center>22.88</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
 </tr>
 <tr height=19 style='height:14.0pt'>
  <td colspan=2 height=19 class=xl65 style='height:14.0pt' align=center><strong>Total</strong></td>
  <td class=xl69 align=center>4.80</td>
  <td class=xl65 align=center>22.64</td>
  <td class=xl66 align=center>4.81</td>
  <td class=xl66 align=center>22.72</td>
  <td class=xl66 align=center>4.57</td>
  <td class=xl66 align=center>21.54</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>4.82</td>
  <td class=xl66 align=center>22.63</td>
  <td class=xl66 align=center>4.76</td>
  <td class=xl66 align=center>22.26</td>
  <td class=xl66 align=center>0</td>
  <td class=xl66 align=center>0</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
  <td width=69 style='width:52pt'></td>
 </tr>
 <![endif]>
</table>