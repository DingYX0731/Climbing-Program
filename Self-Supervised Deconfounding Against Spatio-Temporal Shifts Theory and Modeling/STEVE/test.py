import os
from datetime import datetime

import warnings

from lib.metrics import test_metrics

warnings.filterwarnings('ignore')

import torch
import pickle

from lib.utils import get_project_path

from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, get_log_dir,
)

from lib.dataloader import get_dataloader
from lib.logger import get_logger, PD_Stats
from lib.utils import dwa
import numpy as np
from models.our_model import STEVE

from munch import DefaultMunch


def text2args(text):
    args_dict={}
    temp=text.split(", ")
    for s in temp:
        key,value=s.split("=")
        if '\'' in value:
            args_dict[key] = value.replace('\'','')
        elif '.' in value:
            args_dict[key] =float(value)
        elif 'False' in value:
            args_dict[key] = False
        elif 'True' in value:
            args_dict[key] =True
        else:
            args_dict[key] = int(value)
    args=DefaultMunch.fromDict(args_dict)
    return args


def test(model, dataloader, scaler):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to('cuda:0')
            target = target.to('cuda:0')
            repr1, repr2 = model(data)
            # c_hat=model.predict_con(data)
            predict_output = model.predict(repr1, repr2, data)
            target = target.squeeze(1)
            # invariant_pred.append(invariant)
            # variant_pred.append(variant)
            y_true.append(target)
            y_pred.append(predict_output)
            # y_pred.append(pred_output)
    # invariant_pred = scaler.inverse_transform(torch.cat(invariant_pred, dim=0)).cpu()
    # variant_pred = scaler.inverse_transform(torch.cat(variant_pred, dim=0)).cpu()
    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0)).cpu()
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0)).cpu()

    return y_true, y_pred


def make_one_hot(labels, classes):
    # labels=labels.to('cuda:1')
    labels = labels.unsqueeze(dim=-1)
    one_hot = torch.FloatTensor(labels.size()[0], classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def create_c (y, cluster_labels):
    print(y.shape)
    y_test_c0 = torch.tensor([])
    y_test_c1 = torch.tensor([])
    y_test_c2 = torch.tensor([])
    temp_y = y.permute(2, 0, 1, 3)
    for i in range(cluster_labels.shape[0]):
        label = cluster_labels[i]
        if label == 0:
            y_test_c0 = torch.cat((y_test_c0, temp_y[i].unsqueeze(0)), dim=0)
        elif label == 1:
            y_test_c1 = torch.cat((y_test_c1, temp_y[i].unsqueeze(0)), dim=0)
        else:
            y_test_c2 = torch.cat((y_test_c2, temp_y[i].unsqueeze(0)), dim=0)

    y_test_c0 = y_test_c0.permute(1, 2, 0, 3)
    y_test_c1 = y_test_c1.permute(1, 2, 0, 3)
    y_test_c2 = y_test_c2.permute(1, 2, 0, 3)

    return y_test_c0, y_test_c1, y_test_c2



def main(args):
    
    A = load_graph(args.graph_file, device=args.device)  

    init_seed(args.seed)

    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )

    with open("workday_test_dataloader.pkl", "rb") as f:
        workday_set = pickle.load(f)
    
    with open("holiday_test_dataloader.pkl", "rb") as f:
        holiday_set = pickle.load(f)

    with open("total_test_dataloader.pkl", "rb") as f:
        total_set = pickle.load(f)

    log_dir = 'experiments/NYCBike1/mydata/20240412-230858'
    model = STEVE(args=args, adj=A, in_channels=args.d_input, embed_size=args.d_model,
                T_dim=args.input_length, output_T_dim=1, output_dim=args.d_output,device=args.device).to(args.device)
    
    best_path=os.path.join(log_dir,'best_model.pth')
    print('load model from {}.'.format(best_path))
    state_dict = torch.load(
                best_path,
                map_location=torch.device(args.device)
            )
    model.load_state_dict(state_dict['model'])
    print('load model successfully.')

    test_result = {}

    y_true, y_pred = test(model, total_set['dataloader'], total_set['scaler'])
    mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
    test_result['mae'] = mae
    test_result['mape'] = mape

    y_true_workday, y_pred_workday = test(model, workday_set['dataloader'], workday_set['scaler'])
    y_true_holiday, y_pred_holiday = test(model, holiday_set['dataloader'], holiday_set['scaler'])
    mae_workday, mape_workday = test_metrics(y_pred_workday[..., 0], y_true_workday[..., 0])
    mae_holiday, mape_holiday = test_metrics(y_pred_holiday[..., 0], y_true_holiday[..., 0])
    test_result['mae_workday'] = mae_workday
    test_result['mape_workday'] = mape_workday
    test_result['mae_holiday'] = mae_holiday
    test_result['mape_holiday'] = mape_holiday

    y_true, y_pred = test(model, total_set['dataloader'], total_set['scaler'])
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    cluster_labels = np.load("cluster_labels.npy")
    y_pred_c0, y_pred_c1, y_pred_c2 = create_c(y_pred.unsqueeze(1), cluster_labels)
    y_true_c0, y_true_c1, y_true_c2 = create_c(y_true.unsqueeze(1), cluster_labels)
    mae_c0, mape_c0 = test_metrics(y_pred_c0[..., 0], y_true_c0[..., 0])
    mae_c1, mape_c1 = test_metrics(y_pred_c1[..., 0], y_true_c1[..., 0])
    mae_c2, mape_c2 = test_metrics(y_pred_c2[..., 0], y_true_c2[..., 0])
    test_result['mae_c0'] = mae_c0
    test_result['mape_c0'] = mape_c0
    test_result['mae_c1'] = mae_c1
    test_result['mape_c1'] = mape_c1
    test_result['mae_c2'] = mae_c2
    test_result['mape_c2'] = mape_c2

    result_path=os.path.join(log_dir,'result.npz')
    print('save result in {}.'.format(result_path))
    np.savez(result_path,
            y_true_workday=y_true_workday,y_pred_workday=y_pred_workday,
            y_true_holiday=y_true_holiday,y_pred_holiday=y_pred_holiday,
            y_true=y_true,y_true_c0=y_true_c0,y_true_c1=y_true_c1,y_true_c2=y_true_c2,
            y_pred=y_pred,y_pred_c0=y_pred_c0,y_pred_c1=y_pred_c1,y_pred_c2=y_pred_c2,
            test_result=test_result)
        
        
        


    

    














