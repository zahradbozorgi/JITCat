#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from pathlib import Path
import os
import sys
import time
import itertools
import argparse


# In[2]:

overall_start_time = time.time()

from catboost import CatBoostClassifier
import sklearn 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.frozen import FrozenEstimator
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# import math
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.model_selection import ParameterGrid
# from joblib import Parallel, delayed
# import pickle


# In[3]:


ROOT_DIR = Path().cwd()
while not ROOT_DIR.joinpath("data").exists():
    ROOT_DIR = ROOT_DIR.parent
os.chdir(ROOT_DIR)
# Add the root directory to sys.path
sys.path.append(str(ROOT_DIR))


# In[4]:


from processors import processor_factory
from data.DatasetManager import DatasetManager
import data.EncoderFactory as EncoderFactory


# In[5]:


import category_encoders as ce


# In[6]:


import warnings
warnings.filterwarnings('ignore')


# In[7]:


def get_first_n_cases(df, n, dataset_manager):
    earliest_timestamps = df.groupby(dataset_manager.case_id_col)[dataset_manager.timestamp_col].min()
    sorted_cases = earliest_timestamps.sort_values().index[:n]
    return df[df[dataset_manager.case_id_col].isin(sorted_cases)]


# In[8]:


def return_last_row(group):
    max_event_row = group.loc[group['event_nr'].idxmax()]
    return max_event_row


# In[9]:


def find_threshold(proba_values, true_values):
    list_acc = []
    thresholds = np.arange(0, 1.0, 0.05)
    true_values = true_values.map({'regular': True, 'deviant': False})


    for threshold in thresholds:
        preds_thr = proba_values > threshold
        acc= accuracy_score(true_values, preds_thr)
        # acc = np.mean(true_values == preds_thr)
        list_acc.append(acc)

        #print(f"Threshold: {threshold}, Accuracy: {acc}")

    # plt.plot(thresholds, list_acc)


    best_threshold = thresholds[np.where(list_acc==np.max(list_acc))]
    best_accuracy = np.max(list_acc)

    # print(f"Best Threshold: {best_threshold}, Best Accuracy: {best_accuracy}")
    return best_threshold, best_accuracy


# In[10]:


def calculate_moving_avg_f1(df, true_col, pred_col, dataset_manager, window_size=5):
    # Sort the DataFrame by the timestamp column
    df = df.sort_values([dataset_manager.timestamp_col], ascending=True, kind='mergesort')

    # Convert columns to numpy arrays for faster operations
    true_values = df[true_col].to_numpy()
    predicted_values = df[pred_col].to_numpy()

    # Preallocate arrays for results
    num_rows = np.arange(2, len(true_values) + 1)  # Start from 2
    f1_list = np.zeros(len(num_rows))

    # Compute F1 scores incrementally
    for i in range(2, len(true_values) + 1):
        f1_list[i - 2] = f1_score(true_values[:i], predicted_values[:i], average='weighted')

    # Create a DataFrame for results
    f1_df = pd.DataFrame({'num_rows': num_rows, 'f1': f1_list})

    # Compute the moving average of F1 scores
    f1_df['moving_avg_f1'] = f1_df['f1'].rolling(window=window_size, min_periods=1).mean()

    return f1_df


# In[11]:


def calculate_moving_avg_acc_fast(df, true_col, pred_col, dataset_manager, window_size=5):

    df.sort_values([dataset_manager.timestamp_col], ascending=True, kind='mergesort', inplace=True)
    # Convert columns to numpy arrays
    true_values = df[true_col].to_numpy()
    predicted_values = df[pred_col].to_numpy()


    # Calculate cumulative accuracy (vectorized)
    cumulative_correct = np.cumsum(true_values == predicted_values)
    num_rows = np.arange(1, len(true_values) + 1)
    cumulative_accuracy = cumulative_correct / num_rows

    # Create DataFrame for results
    mae_df = pd.DataFrame({
        'num_rows': num_rows[1:],  # Start from 2
        'mae': cumulative_accuracy[1:]  # Start from 2
    })

    # Calculate moving average of MAE
    mae_df['moving_avg_mae'] = mae_df['mae'].rolling(window=window_size).mean()

    return mae_df


# In[12]:


def save_results_to_csv(dataset_name, proposed_metrics, baseline_metrics, file_path='results_metrics_HP_tuning.csv'):
    """
    Save results metrics to a CSV file.

    Parameters:
        dataset_name (str): Name of the dataset.
        proposed_metrics (dict): Metrics for the proposed method (keys: accuracy, f1_score, auc, precision, recall).
        baseline_metrics (dict): Metrics for the baseline method (keys: accuracy, f1_score, auc, precision, recall).
        file_path (str): Path to the results CSV file.
    """
    # Define the columns for the CSV file
    columns = [
        'dataset_name',
        'proposed_accuracy', 'proposed_f1_score', 'proposed_auc', 'proposed_precision', 'proposed_recall',
        'baseline_accuracy', 'baseline_f1_score', 'baseline_auc', 'baseline_precision', 'baseline_recall'
    ]

    # Create a DataFrame for the new results
    new_data = {
        'dataset_name': dataset_name,
        'proposed_accuracy': proposed_metrics['accuracy'],
        'proposed_f1_score': proposed_metrics['f1_score'],
        'proposed_auc': proposed_metrics['auc'],
        'proposed_precision': proposed_metrics['precision'],
        'proposed_recall': proposed_metrics['recall'],
        'baseline_accuracy': baseline_metrics['accuracy'],
        'baseline_f1_score': baseline_metrics['f1_score'],
        'baseline_auc': baseline_metrics['auc'],
        'baseline_precision': baseline_metrics['precision'],
        'baseline_recall': baseline_metrics['recall']
    }

    new_row = pd.DataFrame([new_data])

    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, create it with the appropriate headers
        new_row.to_csv(file_path, index=False, columns=columns)
    else:
        # If the file exists, append the new row
        existing_data = pd.read_csv(file_path)
        # Check if the dataset already exists in the file
        if dataset_name in existing_data['dataset_name'].values:
            print(f"Dataset '{dataset_name}' already exists in the results file. Updating the row.", flush=True)

            # Remove the existing row with the same dataset_name
            existing_data = existing_data[existing_data['dataset_name'] != dataset_name]

            # Append the new row to the DataFrame
            updated_data = pd.concat([existing_data, new_row], ignore_index=True)

            # Overwrite the file with the updated DataFrame
            updated_data.to_csv(file_path, mode='w', index=False, columns=columns)
        else:
            new_row.to_csv(file_path, mode='a', index=False, header=False, columns=columns)


# In[13]:


def calculate_metrics(results_df, dataset_manager):
    """
    Calculate metrics from a results DataFrame.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing 'true_labels', 'predicted_labels', and 'predicted_probs'.

    Returns:
        dict: A dictionary containing accuracy, f1_score, auc, precision, and recall.
    """
    true_labels = results_df[dataset_manager.label_col]
    predicted_labels = results_df['predicted_value']
    predicted_probs = results_df['proba_of_regular']

    metrics = {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'f1_score': f1_score(true_labels, predicted_labels, average='weighted'),
        'auc': roc_auc_score(true_labels, predicted_probs),
        'precision': precision_score(true_labels, predicted_labels, average='weighted'),
        'recall': recall_score(true_labels, predicted_labels, average='weighted')
    }
    return metrics


# In[14]:


# dataset_name = 'bpic2017_accepted'
# test_mode = True

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process dataset name and test mode as arguments.")
parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to process.")
parser.add_argument('--test_mode', type=bool, default=False, help="Run in test mode (True or False).")
args = parser.parse_args()

# Use the arguments in your code
dataset_name = args.dataset_name
test_mode = args.test_mode

print(f"Processing dataset: {dataset_name}", flush=True)
print(f"Test mode: {test_mode}", flush=True)


# In[15]:


dataset_manager = DatasetManager(dataset_name)


# In[16]:


df = dataset_manager.read_dataset()


# In[17]:


df.sort_values([dataset_manager.case_id_col, dataset_manager.timestamp_col], inplace=True)


# In[18]:


max_case_num = df[dataset_manager.case_id_col].nunique()


# In[19]:


filtered_df = get_first_n_cases(df, max_case_num, dataset_manager)


# In[20]:


for col in [dataset_manager.activity_col]:
        counts = filtered_df[col].value_counts()
        mask = filtered_df[col].isin(counts[counts >= 100].index)
        filtered_df.loc[~mask, col] = "other"


# In[21]:


use_encoding = False
use_bucketing = False
num_nearest_neighbors = 100
distance_metric = 'euclidean'

processor = processor_factory.get_processor(dataset_name, use_encoding, use_bucketing, num_nearest_neighbors, distance_metric)


# In[22]:


# determine min and max (truncated) prefix lengths
min_prefix_length = 1
if "traffic_fines" in dataset_name:
    max_prefix_length = 10
elif "bpic2017" in dataset_name:
    max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(filtered_df, 0.90))
else:
    max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(filtered_df, 0.90))


# In[23]:


start_test_prefix_generation = time.time()
print("Generating prefix data...", flush=True)
dt_prefixes = dataset_manager.generate_prefix_data(filtered_df, min_prefix_length, max_prefix_length)
test_prefix_generation_time = time.time() - start_test_prefix_generation


# In[24]:


encoder = EncoderFactory.get_encoder(method='agg', case_id_col=dataset_manager.case_id_col, static_cat_cols=dataset_manager.static_cat_cols, 
                                     static_num_cols=dataset_manager.static_num_cols, dynamic_cat_cols=dataset_manager.dynamic_cat_cols,
                                     dynamic_num_cols=dataset_manager.dynamic_num_cols, fillna=True, max_events=None, 
                                     activity_col=dataset_manager.activity_col, resource_col=None, 
                                     timestamp_col=dataset_manager.timestamp_col, scale_model=None)


# In[25]:


dt_transformed = encoder.transform(dt_prefixes)


# In[26]:


subset = dt_prefixes[[dataset_manager.case_id_col, dataset_manager.timestamp_col, dataset_manager.activity_col, dataset_manager.label_col, 'event_nr', 'case_length'] + dataset_manager.static_num_cols+dataset_manager.static_cat_cols]
subset = subset.groupby(dataset_manager.case_id_col).apply(return_last_row).reset_index(drop=True)


# In[27]:


# Create a new column 'finished' with values based on the condition
subset['finished'] = (subset['event_nr'] == subset['case_length']).astype(int)


# In[28]:


merged_df = pd.merge(subset, dt_transformed, on=[dataset_manager.case_id_col])


# In[29]:


# Step 1: Identify object columns
object_columns = merged_df.select_dtypes(include=['object']).columns

# Step 2: Check if object columns contain boolean values
for col in object_columns:
    if merged_df[col].isin(['True', 'False', 'TRUE', 'FALSE', 'true', 'false']).all():
        merged_df[col] = merged_df[col].str.lower().map({'true': True, 'false': False})

        # Step 3: Transform boolean object columns to boolean data type
        merged_df[col] = merged_df[col].astype('boolean')


# In[30]:


# Define the hyperparameter grid
start = time.time()
# Generate all combinations of hyperparameters
param_grid = {
    'num_nearest_neighbors': [50, 70, 100, 300, 500],
    'distance_metric': ['euclidean'],
    'encoding_method': ['catboost', 'target', 'woe', 'rank'],
    'model': ['Catboost'],
    'batch_size': [100]  # Added batch size as a hyperparameter
}
param_combinations = list(itertools.product(*param_grid.values()))
param_keys = list(param_grid.keys())

# Convert combinations to dictionaries
param_dicts = [dict(zip(param_keys, values)) for values in param_combinations]
end = time.time()
print(f"Parameter grid established in {end - start:.2f} seconds.", flush=True)

# Preprocess data once
start = time.time()
data = merged_df.sort_values([dataset_manager.case_id_col, dataset_manager.timestamp_col], ascending=True, kind='mergesort')
processor = processor_factory.get_processor(dataset_name, use_encoding=False, use_bucketing=False, 
            num_nearest_neighbors=num_nearest_neighbors, distance_metric=distance_metric)
historic, current = processor.split_data_strict(data, train_ratio=0.5)
historic.sort_values([dataset_manager.timestamp_col], ascending=True, kind='mergesort', inplace=True)
current.sort_values([dataset_manager.timestamp_col], ascending=True, kind='mergesort', inplace=True)
if test_mode:
    current = current.head(200)

features_used = historic.columns.difference(
    [dataset_manager.label_col, dataset_manager.timestamp_col, dataset_manager.case_id_col, 'event_nr', 'case_length', 'finished'], 
    sort=False
)
end = time.time()
print(f"Data preprocessing completed in {end - start:.2f} seconds.", flush=True)


# In[31]:


####### Baseline Version without JITL ########

results_baseline = pd.DataFrame()
method = 'Catboost'
print(f"Experimenting with baseline version", flush=True)

target = historic[dataset_manager.label_col].values
target_test = current[dataset_manager.label_col]

start_time = time.time()
if method == 'Catboost':
        model = CatBoostClassifier(iterations=100, loss_function='Logloss', eval_metric='AUC', verbose=0, cat_features=[dataset_manager.activity_col]+dataset_manager.static_cat_cols, task_type='CPU', random_seed=321)
        print('Now training')
        model.fit(historic[features_used], target, cat_features=[dataset_manager.activity_col]+dataset_manager.static_cat_cols)

if method == 'HMM':
    # Create an instance of the HMM model
    model = hmm.GaussianHMM(n_components=7)  # Specify the number of hidden states
    model.fit(historic[features_used])
if method == 'LogisticRegression':
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import make_pipeline 
    from sklearn.compose import make_column_selector, make_column_transformer

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include='object')),
        remainder='passthrough'
    )
    model = make_pipeline(
        preprocessor,
        LogisticRegression()
    )
    model.fit(X_train[features_used], y_train)

end_time = time.time()
training_time = (end_time - start_time) / 60
print("Training time: ", training_time, " minutes")
# trainingtimes.append(training_time)
batch_size = 100

print("Now predicting")
preds = model.predict(current[features_used])
probs = model.predict_proba(current[features_used])[:, 1]
y_true = current[dataset_manager.label_col].values
auc = roc_auc_score(y_true, probs)

results_baseline = current.copy()
results_baseline.loc[:, 'predicted_value'] = pd.Series(preds.flatten(), index=current.index)
results_baseline.loc[:, 'proba_of_regular'] = pd.Series(probs.flatten(), index=current.index)


# Calculate metrics
true_values = results_baseline[dataset_manager.label_col]
predicted_values = results_baseline['predicted_value']

accuracy = np.mean(true_values == predicted_values)

# Print metrics
print(f"Accuracy: {accuracy}", flush=True)
print(f"f1_score: {f1_score(true_values, predicted_values, average='weighted')}", flush=True)
print(f"Training Time: {training_time}", flush=True)
print(f"AUC: {auc}", flush=True)

output_dir = Path(f"results_HP_tuning/{dataset_name}")
output_dir.mkdir(parents=True, exist_ok=True)
results_baseline.to_csv(f'{output_dir}/baseline_{method}.csv', index=False)
print('***********************************', flush=True)

# Calculate metrics for baseline method
baseline_metrics = calculate_metrics(results_baseline, dataset_manager)


# In[32]:



# In[33]:


# Precompute and pass only picklable objects
start = time.time()
preprocessed_data = {
    'historic': historic,  # Convert DataFrame to dictionary
    'current': current,
    'features_used': list(features_used)  # Convert Index to list
}

dataset_manager_data = {
    'case_id_col': dataset_manager.case_id_col,
    'timestamp_col': dataset_manager.timestamp_col,
    'label_col': dataset_manager.label_col,
    'activity_col': dataset_manager.activity_col,
    'static_cat_cols': dataset_manager.static_cat_cols
}
end = time.time()
print("Preprocessed data ready for parallel processing in {:.2f} seconds.".format(end - start), flush=True)


# In[34]:


def process_combination(params, preprocessed_data, dataset_manager_data):
    start_time = time.time()
    results = []
    AUCs = []
    results_df = pd.DataFrame()
    results_dicts = []
    print(f"Testing combination: {params}", flush=True)

    # Extract preprocessed data
    historic = pd.DataFrame(preprocessed_data['historic'])
    current = pd.DataFrame(preprocessed_data['current'])
    features_used = preprocessed_data['features_used']

    # Configure preprocessor
    if params['encoding_method'] == 'quantile':
        preprocessor = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            (ce.quantile_encoder.SummaryEncoder(), make_column_selector(dtype_include=['object', 'category'])),
            remainder='drop'
        )
    elif params['encoding_method'] == 'onehot':
        preprocessor = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            (ce.OneHotEncoder(), make_column_selector(dtype_include=['object', 'category'])),
            remainder='drop'
        )
    elif params['encoding_method'] == 'catboost':
        preprocessor = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            (ce.CatBoostEncoder(), make_column_selector(dtype_include=['object', 'category'])),
            remainder='drop'
        )
    elif params['encoding_method'] == 'count':
        preprocessor = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            (ce.CountEncoder(normalize=True), make_column_selector(dtype_include=['object', 'category'])),
            remainder='drop'
        )
    elif params['encoding_method'] == 'target':
        preprocessor = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            (ce.TargetEncoder(), make_column_selector(dtype_include=['object', 'category'])),
            remainder='drop'
        )
    elif params['encoding_method'] == 'woe':
        preprocessor = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            (ce.WOEEncoder(), make_column_selector(dtype_include=['object', 'category'])),
            remainder='drop'
        )
    elif params['encoding_method'] == 'rank':
        preprocessor = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            (ce.RankHotEncoder(), make_column_selector(dtype_include=['object', 'category'])),
            remainder='drop'
        )

    # Transform data once
    historic_transformed = preprocessor.fit_transform(historic[features_used], historic[dataset_manager_data['label_col']])
    current_transformed = preprocessor.transform(current[features_used])

    # Train nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=params['num_nearest_neighbors'], metric=params['distance_metric'])
    nn_model.fit(historic_transformed)

    batch_size = params['batch_size']  # Define batch size for processing

    # Process batches
    for start in range(0, len(current), batch_size):  # Batch size = 100
        batch = current.iloc[start:start+batch_size]
        batch_transformed = preprocessor.transform(batch[features_used])
        distances, indices = nn_model.kneighbors(batch_transformed)
        nearest_neighbors = pd.concat([historic.iloc[indices[i]] for i in range(len(batch))])

        target = nearest_neighbors[dataset_manager_data['label_col']].values
        target_test = batch[dataset_manager_data['label_col']]

        X_train, X_cal, y_train, y_cal = train_test_split(nearest_neighbors, target, test_size=0.3, random_state=42)

        # Train model
        if params['model'] == 'Catboost':
            model = CatBoostClassifier(iterations=100, loss_function='Logloss', eval_metric='AUC', verbose=0, cat_features=[dataset_manager_data['activity_col']]+dataset_manager_data['static_cat_cols'], task_type='CPU', random_seed=42)
            model.fit(X_train[features_used], y_train, cat_features=[dataset_manager_data['activity_col']]+dataset_manager_data['static_cat_cols'])
        elif params['model'] == 'LogisticRegression':
            model = make_pipeline(preprocessor, LogisticRegression())
            model.fit(X_train[features_used], y_train)
        elif params['model'] == 'HMM':
            model = hmm.GaussianHMM(n_components=7)
            model.fit(nearest_neighbors[features_used])

        # Calibrate and predict
        # calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
        calibrated_model.fit(X_cal[features_used], y_cal)

        # Predict probabilities and labels
        probs = calibrated_model.predict_proba(batch[features_used])[:, 1]
        preds = calibrated_model.predict(batch[features_used])
        y_true = batch[dataset_manager_data['label_col']].values

        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, probs)
            AUCs.append(auc)
        else:
            auc = None

        # Check if array has null values
        has_null = pd.Series(preds.flatten(), index=batch.index).isna().any()

        if has_null:
            print("Preds has null values")

        batch.loc[:, 'predicted_value'] = pd.Series(preds.flatten(), index=batch.index)
        batch.loc[:, 'proba_of_regular'] = pd.Series(probs.flatten(), index=batch.index)
        is_null = batch['predicted_value'].isna().any()
        if is_null:
            print("Batch has null values")

        results.append(batch)

        # Add the current row with its prediction to the historic data
        finished_case_ids = batch[batch['finished'] == 1][dataset_manager_data['case_id_col']].unique()
        finished_cases = current[current[dataset_manager_data['case_id_col']].isin(finished_case_ids)]
        historic = pd.concat([historic, finished_cases], ignore_index=True)
        historic.sort_values([dataset_manager_data['case_id_col'], dataset_manager_data['timestamp_col']], ascending=True, kind='mergesort', inplace=True)
        historic_transformed = preprocessor.fit_transform(historic[features_used], historic[dataset_manager.label_col])
        nn_model.fit(historic_transformed) # Refit the model with the updated historic data

    results_df = pd.concat(results)

    # Calculate metrics
    true_values = results_df[dataset_manager.label_col]
    predicted_values = results_df['predicted_value']
    proba_values = results_df['proba_of_regular']

    accuracy = accuracy_score(true_values, predicted_values)
    f1 = f1_score(true_values, predicted_values, average='weighted')
    precision = precision_score(true_values, predicted_values, average='weighted', zero_division=0)
    recall = recall_score(true_values, predicted_values, average='weighted', zero_division=0)

    results_dicts.append({
        'params': params,
        'auc': sum(AUCs) / len(AUCs) if AUCs else None,
        'accuracy' : accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
    })
    end_time = time.time()
    print(f"Combination processed in {(end_time - start_time) / 60:.2f} minutes.", flush=True)

    return results_dicts, results_df


# Sequentially process each combination
all_results = []
for params in param_dicts:
    results_dict, results_df = process_combination(params, preprocessed_data, dataset_manager_data)
    all_results.extend(results_dict)  # Append results for this combination

    output_dir = Path(f"results_HP_tuning/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f'{output_dir}/{params}.csv', index=False)

# Flatten results and save
results_all_df = pd.DataFrame(all_results)
gridsearch_output_dir = Path(f"results_HP_tuning/{dataset_name}")
results_all_df.to_csv(f'{gridsearch_output_dir}/grid_search_results.csv', index=False)

# Print the best combination
best_result = results_all_df.loc[results_all_df['f1_score'].idxmax()]


# In[35]:


# print(f"Best combination: {best_result}")
print(best_result['params'], flush=True)


# In[36]:



# In[37]:


# Calculate metrics and save for best combo
best = pd.read_csv(f"{output_dir}/{best_result['params']}.csv")
proposed_metrics = calculate_metrics(best, dataset_manager)
save_results_to_csv(f'{dataset_name}', proposed_metrics, baseline_metrics)


# In[38]:


tmp = pd.read_csv(f'{output_dir}/baseline_{method}.csv')
tmp.sort_values([dataset_manager.timestamp_col], ascending=True, kind='mergesort', inplace=True)

tmp2 = pd.read_csv(f"{output_dir}/{best_result['params']}.csv")
tmp2.sort_values([dataset_manager.timestamp_col], ascending=True, kind='mergesort', inplace=True)

# Calculate moving average accuracy for tmp and tmp2
mae_df_tmp = calculate_moving_avg_acc_fast(tmp, dataset_manager.label_col, 'predicted_value', dataset_manager)
mae_df_tmp2 = calculate_moving_avg_acc_fast(tmp2, dataset_manager.label_col, 'predicted_value', dataset_manager)


# In[39]:


# Plot the moving average accuracy for both DataFrames
plt.figure(figsize=(10, 6))
plt.plot(mae_df_tmp['num_rows'], mae_df_tmp['moving_avg_mae'], label='baseline', color='blue')
plt.plot(mae_df_tmp2['num_rows'], mae_df_tmp2['moving_avg_mae'], label='JIT-Cat', color='red')
# plt.plot(mae_df_tmp3['num_rows'], mae_df_tmp3['moving_avg_mae'], label='euclidean', color='green')
# plt.plot(mae_df_tmp4['num_rows'], mae_df_tmp4['moving_avg_mae'], label='DTW', color='black')
plt.xlabel('Number of Observed Events over Time')
plt.ylabel('Moving Average Accuracy')
plt.title(f'{dataset_name} Moving Average Accuracy')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(f'results_HP_tuning/{dataset_name}/moving_avg_accuracy_score.png', dpi=600, bbox_inches='tight')
plt.savefig(f'results_HP_tuning/{dataset_name}/moving_avg_accuracy_score.pdf', dpi=600, bbox_inches='tight', format='pdf')  # Save as PDF

plt.close()


overall_end_time = time.time()
print(f"Overall execution time: {(overall_end_time - overall_start_time) / 60:.2f} minutes", flush=True)

# In[40]:


# Plot the moving average F1 score for both DataFrames
# plt.figure(figsize=(10, 6))
# plt.plot(mae_df_tmp['num_rows'], mae_df_tmp['moving_avg_f1'], label='baseline', color='blue')
# plt.plot(mae_df_tmp2['num_rows'], mae_df_tmp2['moving_avg_f1'], label='JIT-Cat', color='red')
# # plt.plot(mae_df_tmp3['num_rows'], mae_df_tmp3['moving_avg_mae'], label='euclidean', color='green')
# # plt.plot(mae_df_tmp4['num_rows'], mae_df_tmp4['moving_avg_mae'], label='DTW', color='black')
# plt.xlabel('Number of Observed Events over Time')
# plt.ylabel('Moving Average F1 Score')
# plt.title(f'{dataset_name} Moving Average F1 Score')
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig(f'results/{dataset_name}/moving_avg_f1_score.png', dpi=600, bbox_inches='tight')
# plt.savefig(f'results/{dataset_name}/moving_avg_F1_score.pdf', dpi=600, bbox_inches='tight', format="pdf")  # Save as PDF
# plt.close()


# In[ ]:





# In[ ]:




