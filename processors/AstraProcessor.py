import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dtaidistance import dtw
from processors.BaseProcessor import BaseProcessor
from data import dataset_confs
import random

class AstraProcessor(BaseProcessor):
    def __init__(self, dataset_name, use_encoding, use_bucketing, num_nearest_neighbors, distance_metric):
        self.dataset_name = dataset_name
        self.use_encoding = use_encoding
        self.use_bucketing = use_bucketing
        self.num_nearest_neighbors = num_nearest_neighbors
        if distance_metric == "dtw":
            self.distance_metric = self.dtw_distance
        elif distance_metric == "euclidean_expotential":
            self.distance_metric = self.euclidean_expotential
        elif distance_metric == "cosine_expotential":
            self.distance_metric = self.cosine_expotential
        else: self.distance_metric = distance_metric
        
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        # self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.work_day_col = dataset_confs.work_day_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.target_col = dataset_confs.target_col[self.dataset_name]
        # self.label_col = dataset_confs.label_col[self.dataset_name]
        # self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        # self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        # self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        
        self.sorting_cols = [self.case_id_col, self.work_day_col]

    def load_data(self, file_path):
        return pd.read_csv(file_path)


    def dtw_distance(self, x, y):
        return dtw.distance(x, y)
    
    def euclidean_expotential(self, x, y):
        return np.exp(-np.linalg.norm(x-y))
    
    def cosine_expotential(self, x, y):
        return np.exp(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    
    def split_data(self, data, train_ratio=0.70, split="temporal", leave_out=[], seed=22):  
    # split into train and test using temporal split

        data[self.work_day_col] = data[self.work_day_col].astype(int)
        print(data[self.work_day_col].dtype)
        grouped = data.groupby(self.case_id_col)
        # start_timestamps = grouped[timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]

        elif split == "temporal sim":
            # Calculate the number of elements to select
            num_elements = int(len(data[self.case_id_col].unique()) * train_ratio)
            train_ids = random.sample(list(data[self.case_id_col].unique()), num_elements)

        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
            train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]

        elif split == "loo":
            try:
                leave_id = str(leave_out[0])
                all_ids = list(data[self.case_id_col].unique())
                train_ids = all_ids
                train_ids.remove(leave_id)

            except IndexError:
                print("Leave out list is empty.")

        elif split == "lmo":
            try:
                all_ids = list(data[self.case_id_col].unique())
                train_ids = [id for id in all_ids if id not in leave_out]
                print(f"Train ids: {train_ids}")

            except IndexError:
                print("Leave out list is empty.")
                
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.work_day_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.work_day_col, ascending=True, kind='mergesort')

        return (train, test)

    def train_nn_model(self, X):
        if isinstance(self.distance_metric, str):
            nn_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric=self.distance_metric)
        elif callable(self.distance_metric):
            nn_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric=lambda x, y: self.distance_metric(x, y))
        else:
            raise ValueError("distance_metric should be either a string or a callable function")
        
        nn_model.fit(X)
        return nn_model

    def find_nearest_neighbors(self, model, sample_point):
        distances, indices = model.kneighbors(sample_point)
        return distances, indices