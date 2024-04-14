import os
import sys
import h5py
import math
import torch
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from lib.dataloader import normalize_data

def create_folder (folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder {folder_name} created")
    else:
        print(f"Folder {folder_name} already exists")

def split_x_y_tensor (tensor):
    x_tensor = None
    y_tensor = None
    for i in range(tensor.size(0)-77):
        window_list = [
            tensor[i:i+5],
            tensor[i+24:i+29],
            tensor[i+48:i+53],
            tensor[i+72:i+76]
        ]
        if x_tensor is None:
            x_tensor = torch.cat(window_list, dim=0)
            x_tensor = x_tensor.unsqueeze(0)
        else:
            window_list = torch.cat(window_list, dim=0).unsqueeze(0)
            x_tensor = torch.cat((x_tensor, window_list), dim=0)

        if y_tensor is None:
            y_tensor = tensor[i+77].unsqueeze(0)
        else:
            y_tensor = torch.cat((y_tensor, tensor[i+77].unsqueeze(0)), dim=0)

    x_tensor = x_tensor.permute(0, 1, 3, 2)
    y_tensor = y_tensor.permute(0, 2, 1).unsqueeze(1)
    return x_tensor, y_tensor


def dayhour_to_timelabel(day, hour):
    if day < 5: # workday
        time_label = hour
    else: # holiday
        time_label = hour + 24
    return time_label

def move_forward_1h (hour, day):
    hour += 1
    if hour == 24:
        day += 1
        hour = 0
        if day == 7:
            day = 0
    return hour, day

def create_time_label (hour, day, size):
    time_label = torch.zeros(size)
    for i in range(time_label.size(0)):
        time_label[i] = dayhour_to_timelabel(day, hour)
        hour, day = move_forward_1h(hour, day)
    return time_label

def create_traffic_load (y_tensor):
    max_CP_inflow = {}
    max_CP_outflow = {}
    c = torch.empty_like(y_tensor)

    for i in range(y_tensor.size(0)):
        for node in range(y_tensor.size(2)):
            inflow = y_tensor[i][0][node][0]
            outflow = y_tensor[i][0][node][1]
            if node not in max_CP_inflow:
                max_CP_inflow[node] = inflow
            if node not in max_CP_outflow:
                max_CP_outflow[node] = outflow
            if inflow > max_CP_inflow[node]:
                max_CP_inflow[node] = inflow
            if outflow > max_CP_outflow[node]:
                max_CP_outflow[node] = outflow
            if max_CP_inflow[node] == 0:
                c[i][0][node][0] = 0
            else:
                c[i][0][node][0] = math.ceil(5 * inflow / max_CP_inflow[node])
            if max_CP_outflow[node] == 0:
                c[i][0][node][1] = 0
            else:
                c[i][0][node][1] = math.ceil(5 * outflow / max_CP_outflow[node])
    return c

def split_train_val_test_sets (tensor):
    # shuffle the data
    torch.manual_seed(0)
    indices = torch.randperm(x_tensor.size(0))
    tensor = tensor[indices]

    # split the data
    train_size = int(0.7 * tensor.size(0))
    val_size = int(0.2 * tensor.size(0))

    tensor_train = tensor[:train_size]

    tensor_val = tensor[train_size:train_size+val_size]

    tensor_test = tensor[train_size+val_size:]

    return tensor_train, tensor_val, tensor_test

def create_cluster_labels (tensor):
    spatial = tensor.permute(2, 0, 1).numpy()
    scaler = StandardScaler()
    spatial = scaler.fit_transform(spatial.reshape(-1, spatial.shape[-1])).reshape(spatial.shape)
    spatial_mean = np.mean(spatial, axis=1)
    spatial_median = np.median(spatial, axis=1)
    spatial = np.concatenate((spatial_mean, spatial_median), axis=1)
    n_clusters= range(2, 6)
    best_score = -1
    best_cluster = None
    for n in n_clusters:
        cluster = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cluster_labels = cluster.fit_predict(spatial)

        silhouette_avg = silhouette_score(spatial, cluster_labels)

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_cluster = cluster
    
    return best_cluster.fit_predict(spatial)

def split_work_holiday (tensor, day, hour): # tensor of size (4315, 19, 2, 128)
    workday_tensor = None
    holiday_tensor = None
    for i in range(tensor.size(0)):
        if day <= 4:
            if workday_tensor is None:
                workday_tensor = tensor[i].unsqueeze(0)
                hour, day = move_forward_1h(hour, day)
            else:
                workday_tensor = torch.cat((workday_tensor, tensor[i].unsqueeze(0)), dim=0)
                hour, day = move_forward_1h(hour, day)
        else:
            if holiday_tensor is None:
                holiday_tensor = tensor[i].unsqueeze(0)
                hour, day = move_forward_1h(hour, day)
            else:
                holiday_tensor = torch.cat((holiday_tensor, tensor[i].unsqueeze(0)), dim=0)
                hour, day = move_forward_1h(hour, day)
    
    return workday_tensor, holiday_tensor

def create_temporal_dataloader (x_temporal_tensor, y_temporal_tensor):
    temporal_scaler = normalize_data(x_temporal_tensor, 'Standard')
    x_temporal_tensor = temporal_scaler.transform(x_temporal_tensor).to(torch.float)
    y_temporal_tensor = temporal_scaler.transform(y_temporal_tensor).to(torch.float)

    temporal_dataset = torch.utils.data.TensorDataset(x_temporal_tensor, y_temporal_tensor)

    total_size = len(temporal_dataset)
    
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    train, val, test = torch.utils.data.random_split(temporal_dataset, [train_size, val_size, test_size])

    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=False,
        drop_last=True
    )

    temporal_set = {}
    temporal_set['dataloader'] = test_dataloader
    temporal_set['scaler'] = temporal_scaler

    return temporal_set

"""
Below is the main code for data preprocessing
"""

folder_name = 'NYCBike1'

if not os.path.exists('NYCBike1.h5'):
    print("NYCBike1.h5 not found. Please download the dataset ")
    sys.exit(1)

create_folder(folder_name=folder_name)

data = h5py.File('NYCBike1.h5', 'r')
print(f"Data Keys: {data.keys()}")
data_tensor = torch.tensor(data['data'])
print(f"Data Tensor Shape: {data_tensor.shape}")
tensor = torch.reshape(data_tensor, (data_tensor.shape[0], data_tensor.shape[1], 128))

x_tensor, y_tensor = split_x_y_tensor(tensor)

day=4
hour=4

time_label = create_time_label(hour, day, x_tensor.size(0))

c = create_traffic_load(y_tensor)

x_train, x_val, x_test = split_train_val_test_sets(x_tensor)
y_train, y_val, y_test = split_train_val_test_sets(y_tensor)
time_label_train, time_label_val, time_label_test = split_train_val_test_sets(time_label)
c_train, c_val, c_test = split_train_val_test_sets(c)
np.savez(os.path.join(folder_name, 'train.npz'), x=x_train, y=y_train, time_label=time_label_train, c=c_train)
np.savez(os.path.join(folder_name, 'val.npz'), x=x_val, y=y_val, time_label=time_label_val, c=c_val)
np.savez(os.path.join(folder_name, 'test.npz'), x=x_test, y=y_test, time_label=time_label_test, c=c_test)

# K-Means for spatial OOD evaluation
cluster_labels = create_cluster_labels(tensor)
np.save(os.path.join(folder_name, "cluster_labels.npy"), cluster_labels)

x_workday_tensor, x_holiday_tensor = split_work_holiday(x_tensor, day, hour)
y_workday_tensor, y_holiday_tensor = split_work_holiday(y_tensor, day, hour)

total_set = create_temporal_dataloader(x_tensor, y_tensor)
workday_set = create_temporal_dataloader(x_workday_tensor, y_workday_tensor)
holiday_set = create_temporal_dataloader(x_holiday_tensor, y_holiday_tensor)

with open(os.path.join(folder_name, "workday_test_dataloader.pkl"), "wb") as f:
    pickle.dump(workday_set, f)
print(f"workday_test_dataloader.pkl saved")

with open(os.path.join(folder_name, "holiday_test_dataloader.pkl"), "wb") as f:
    pickle.dump(holiday_set, f)
print(f"holiday_test_dataloader.pkl saved")

with open(os.path.join(folder_name, "total_test_dataloader.pkl"), "wb") as f:
    pickle.dump(total_set, f)
print(f"total_test_dataloader.pkl saved")