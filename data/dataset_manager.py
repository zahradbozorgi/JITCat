import numpy as np
import pandas as pd

import os
from pathlib import Path
import glob
ROOT_DIR = Path().cwd()
while not ROOT_DIR.joinpath("data").exists():
    ROOT_DIR = ROOT_DIR.parent
os.chdir(ROOT_DIR)

import dataset_confs

class DatasetManager:

    def __init__(self, dataset_name, file_type):
        self.dataset_name = dataset_name
        self.file_type = file_type

        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.amino_acid_cols = dataset_confs.amino_acid_cols[self.dataset_name]
        self.cols_to_drop = dataset_confs.cols_to_drop[self.dataset_name]
        self.cols_to_drop_potentially = dataset_confs.cols_to_drop_potentially[self.dataset_name]

    def load_data(self):
        # Implement your data loading logic here
        if self.dataset_name=="AMBR":
            df = self.load_ambr(datatype=self.file_type)
        elif self.dataset_name=="5L":
            df = self.load_5L(datatype=self.file_type)
        elif self.dataset_name=="Astrazeneca":
            df = self.load_astra(datatype=self.file_type)
        else:
            raise ValueError("Invalid dataname. Valid datanames are AMBR, 5L, and Astrazeneca.")
        
        # if datatype=="csv":
        #     df = pd.read_csv(str(ROOT_DIR) + f"/data/processed/{dataname}.csv")
        # if datatype=="pickle":
        #     df = pd.read_pickle(f"data/processed/{dataname}.pkl")

        return df

    def load_ambr(self, datatype="csv"):
        ambr_path = Path("./data/Original/AMBR_ELN/AMBR")
        ambr_files = glob.glob("./data/Original/AMBR_ELN/AMBR/*.xlsx*")
        print(len(ambr_files))

        first_file = pd.ExcelFile(ambr_files[0])
        first = first_file.parse("combined", header=0)
        self.first_timestamp = first[first['working day']==0]['Date & Time'].unique()[0]

        self.cols_dict = {}
        list_df = []
        for selected_file in ambr_files:
            print(selected_file)
            ambr_file = pd.ExcelFile(selected_file)

            df = ambr_file.parse("combined", header=0)
            df.drop(self.amino_acid_cols, axis=1, inplace=True, errors='ignore')

            df[self.case_id_col] = df['experiment'].astype(str) + '_' + df['reactor id'].astype(str)
            experiment = str(df['experiment'].unique())
            df.drop(["experiment", 'reactor id'], axis=1, inplace=True)
            df.drop(self.cols_to_drop, axis=1, inplace=True, errors='ignore')
            df.drop(self.cols_to_drop_potentially, axis=1, inplace=True, errors='ignore')
            list_df.append(df)

            self.cols_dict[experiment] = df.columns

        df = pd.concat(list_df, axis=0, ignore_index=True)
        return df

    def load_5L(datatype="csv"):
        return pd.DataFrame()

    def load_astra(datatype="csv"):
        return pd.DataFrame()
    
    def preprocess_ambr(self, df):
        print("hello")
        lists = self.cols_dict.values()
        common_cols = set.intersection(*map(set, lists))
        df = df[list(common_cols)]

        df.drop(['base media', 'feed media'], axis=1, inplace=True)
        df = df[df['working day']>=0]
        df.drop_duplicates(subset=["UID", 'working day'], keep="last")

        df = df.groupby('UID').apply(self.impute_with_median).reset_index()
        df.drop(['level_1', 'eft'], axis=1, inplace=True)

        df[self.timestamp_col] = df['working day'].apply(lambda x: self.first_timestamp + pd.Timedelta(days=x))
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], dayfirst=True)
        df = df.sort_values(by=[self.case_id_col, self.timestamp_col], ascending=True)

        df[self.timestamp_col] = df[self.timestamp_col].dt.date

        return df

    def impute_with_median(self, group):
        return group.fillna(group.median())

# dataset_manager = DatasetManager("AMBR", "csv")
# data = dataset_manager.load_data()
# data_processed = dataset_manager.preprocess_ambr(data)
