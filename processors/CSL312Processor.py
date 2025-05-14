import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dtaidistance import dtw
from processors.BaseProcessor import BaseProcessor
from data import dataset_confs
import random


class CSL312Processor(BaseProcessor):
    def __init__(self, dataset_name, use_encoding, use_bucketing, num_nearest_neighbors, distance_metric):
        super().__init__(dataset_name, use_encoding, use_bucketing, num_nearest_neighbors, distance_metric)