import os

case_id_col = {}
timestamp_col = {}
work_day_col = {}
target_col = {}
amino_acid_cols = {}
cols_to_drop = {}
cols_to_drop_potentially = {}


activity_col = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = "labeled_logs_csv_processed"

### AMBR dataset from Sharepoint ###

dataset = "AMBR"

print(dataset)

case_id_col[dataset] = "UID"
timestamp_col[dataset] = "timestamp"
target_col[dataset] = "titre(g/l)"
work_day_col[dataset] = "working day"
amino_acid_cols[dataset] = ['aa_alanine(g/l)', 'aa_arginine(g/l)',
       'aa_asparticacid(g/l)', 'aa_asparagine(g/l)', 'aa_cysteine(g/l)',
       'aa_glutamicacid(g/l)', 'aa_glycine(g/l)', 'aa_histidine(g/l)',
       'aa_isoleucine(g/l)', 'aa_leucine(g/l)', 'aa_lysine(g/l)',
       'aa_methionine(g/l)', 'aa_phenylalanine(g/l)', 'aa_proline(g/l)',
       'aa_serine(g/l)', 'aa_threonine(g/l)', 'aa_tryptophan(g/l)',
       'aa_tyrosine(g/l)', 'aa_valine(g/l)', 'aa_cystine(g/l)',
       'aa_phosphotyrosine(g/l)', 'aa_pyruvicacid(g/l)']

cols_to_drop[dataset] = ['Batch ID', "level_0", "index", "total cell density(1e6 cells/ml)"]
cols_to_drop_potentially[dataset] = ['SEC High MW', 'SEC Intact', 'SEC Low MW', 'CHOP ng/mL', 'CHOP Total (ng)', 'CHO HCP (ng)/ total protein A (mg)',
       'cGE Caliper (NR) Result (% LMWS)', 'cGE Caliper (NR) Result (% Intact)', 'cGE Caliper (NR) Result (% HMWS)', 'HPLC - CEX Result  (% Acidic)',
       'HPLC - CEX Result (% Main)', 'HPLC - CEX Result (% Basic)', 'N-Glycan (%G0F)', 'N-Glycan (%G1F)', 'N-Glycan (%G2F)','N-Glycan (% Remaining)']

dynamic_cat_cols[dataset] = []
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['glutamate(g/l)', 'ammonia(g/l)','viable cell density(1e6 cells/ml)', 'lactate(g/l)', 'ph',
       'glucose(g/l)', 'glutamine(g/l)','viability(%)']
static_num_cols[dataset] = []

### 5L dataset from Databricks ###

dataset = "5L"

case_id_col[dataset] = "UID"
timestamp_col[dataset] = "Timestamp"
target_col[dataset] = "Titer (g/L)"
work_day_col[dataset] = "Work_Day_Index"
amino_acid_cols[dataset] = []

cols_to_drop[dataset] = []
cols_to_drop_potentially[dataset] = []

dynamic_cat_cols[dataset] = ['Cluster']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['Viability (%)', 'Viable Cell Density (1E06 cells/mL)', 'Glucose (g/L)', 'Lactate (g/L)',
                              'Temperature (ºC)', 'DO (%) [controlling probe]', 'Cumulative IVCD (1E06 cells*day/mL)',
                              'pH', 'Glutamine (g/L)', 'Glutamate (g/L)', 'Ammonia (g/L)', 'Sodium (g/L)', 'Potassium (g/L)',
                              'Calcium (g/L)']
static_num_cols[dataset] = []

### 5L dataset from Databricks ###

dataset = "CSL312"

print(dataset)

case_id_col[dataset] = "Identification No."
timestamp_col[dataset] = "Time (hrs)"
work_day_col[dataset] = "Working Day (d)"
target_col[dataset] = "Product Content pre clean up (g/L)"
amino_acid_cols[dataset] = []

cols_to_drop[dataset] = []
cols_to_drop_potentially[dataset] = []

dynamic_cat_cols[dataset] = ['Cluster']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['BR Temp (°C)', 'pH (Internal)', 'Viability (%)', 'VCD (10^6 cells/mL)', 
                'Glutamine (g/L)', 'Glutamate (g/L)', 'Glucose (g/L)','Lactate (g/L)', 'NH4+ (g/L)', 
                'Na+ (g/L)', 'K+ (g/L)', 'Ca2+ (g/L)']
static_num_cols[dataset] = []

dataset = "Astrazeneca"

print(dataset)

case_id_col[dataset] = "Culture ID"
timestamp_col[dataset] = "ECT"
work_day_col[dataset] = "Working_Day"
target_col[dataset] = "[mAb]"
amino_acid_cols[dataset] = []

cols_to_drop[dataset] = []
cols_to_drop_potentially[dataset] = []

dynamic_cat_cols[dataset] = ['Cluster']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['VCD', 'TCD', 'ACC', 'ACD', 'pH', '[Glutamine]', '[Glutamate]', 'EGN', 'ACV', '[Lactate] ', '[NH3]', 'Osmolality', '[Glucose]', 'CPDL', '[Na+]', '[K+]', '[HCO3-]', 'Temperature', 'pCO2', 'pO2', 'Culture Volume', 'Vol_original']
static_num_cols[dataset] = []


#### BPIC2011 settings ####
for formula in range(1,5):
    dataset = "bpic2011_f%s"%formula 
    print(dataset)
    
    filename[dataset] = os.path.join(logs_dir, "BPIC11_f%s.csv"%formula)
    
    case_id_col[dataset] = "Case ID"
    target_col[dataset] = "Activity code"
#     resource_col[dataset] = "Producer code"
    timestamp_col[dataset] = "time:timestamp"
#     label_col[dataset] = "label"
#     pos_label[dataset] = "deviant"
#     neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity code", "Producer code", "Section", "Specialism code.1", "group"]
    static_cat_cols[dataset] = ["Diagnosis", "Treatment code", "Diagnosis code", "Specialism code", "Diagnosis Treatment Combination ID"]
    dynamic_num_cols[dataset] = ["Number of executions", "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ["Age"]

 
