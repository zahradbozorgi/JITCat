import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dtaidistance import dtw
from processors.BaseProcessor import BaseProcessor
from data import dataset_confs
import random

class BPICProcessor(BaseProcessor):
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
        self.cat_types = {}
        
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.target_col = dataset_confs.target_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        
        self.sorting_cols = [self.case_id_col, self.timestamp_col]

    def split_data_strict(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
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
    
    def infer_categories(self, df):
        for col in self.dynamic_cat_cols + self.static_cat_cols:
            unique_values = df[col].unique()
            self.cat_types[col] = CategoricalDtype(categories=unique_values, ordered=True)

    def infer_categories_encoding(self, df):
        self.object_columns = df.select_dtypes(include=['object']).columns
        for col in self.object_columns:
            unique_values = df[col].unique()
            self.cat_types[col] = CategoricalDtype(categories=unique_values, ordered=True)

    def encode_categorical_columns(self, X):
        df = X.copy()
        for col in self.dynamic_cat_cols + self.static_cat_cols:
            if col in self.cat_types:
                df[col] = df[col].astype(self.cat_types[col]).cat.codes
        return df
    
    def encode_categorical_columns_encoding(self, X):
        df = X.copy()
        for col in self.object_columns:
            if col in self.cat_types:
                df[col] = df[col].astype(self.cat_types[col]).cat.codes
        return df
    
    def train_nn_model_bpic(self, df):
        # Infer categories from the data
        self.infer_categories(df)
        # Encode categorical columns
        X = self.encode_categorical_columns(df)
        cat_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric="hamming")

        if isinstance(self.distance_metric, str):
            num_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric=self.distance_metric)
        elif callable(self.distance_metric):
            num_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric=lambda x, y: self.distance_metric(x, y))
        else:
            raise ValueError("distance_metric should be either a string or a callable function")
        
        num_model.fit(X[self.dynamic_num_cols+self.static_num_cols])
        cat_model.fit(X[self.dynamic_cat_cols+self.static_cat_cols])

        return num_model, cat_model
    
    def train_nn_model_bpic_encoding(self, df):
        # Infer categories from the data
        self.infer_categories_encoding(df)
        # Encode categorical columns
        X = self.encode_categorical_columns_encoding(df)
        cat_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric="hamming")

        if isinstance(self.distance_metric, str):
            num_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric=self.distance_metric)
        elif callable(self.distance_metric):
            num_model = NearestNeighbors(n_neighbors=self.num_nearest_neighbors, metric=lambda x, y: self.distance_metric(x, y))
        else:
            raise ValueError("distance_metric should be either a string or a callable function")
        
        self.numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        num_model.fit(X[self.numeric_columns])
        cat_model.fit(X[self.object_columns])

        return num_model, cat_model

    def find_nearest_neighbors(self, cat_model, num_model, sample_point):
        sample_point = self.encode_categorical_columns(sample_point)
        cat_distances, cat_indices = cat_model.kneighbors(sample_point[self.dynamic_cat_cols+self.static_cat_cols])
        
        # Find nearest neighbors for numerical features
        num_distances, num_indices = num_model.kneighbors(sample_point[self.dynamic_num_cols+self.static_num_cols])

        # Normalize distances
        cat_distances_normalized = cat_distances / np.max(cat_distances)
        num_distances_normalized = num_distances / np.max(num_distances)
        
        # Combine distances
        combined_distances = cat_distances_normalized + num_distances_normalized
        # Find indices of the nearest neighbors based on combined distance
        combined_indices = np.argsort(combined_distances, axis=1)
        
        # Return combined distances and indices
        return combined_distances, combined_indices
        

    def find_nearest_neighbors_encoding(self, cat_model, num_model, sample_point):
        sample_point = self.encode_categorical_columns_encoding(sample_point)
        cat_distances, cat_indices = cat_model.kneighbors(sample_point[self.object_columns])
        
        # Find nearest neighbors for numerical features
        
        num_distances, num_indices = num_model.kneighbors(sample_point[self.numeric_columns])

        # Normalize distances
        cat_distances_normalized = cat_distances / np.max(cat_distances)
        num_distances_normalized = num_distances / np.max(num_distances)
        
        # Combine distances
        combined_distances = cat_distances_normalized + num_distances_normalized
        # Find indices of the nearest neighbors based on combined distance
        combined_indices = np.argsort(combined_distances, axis=1)
        
        # Return combined distances and indices
        return combined_distances, combined_indices