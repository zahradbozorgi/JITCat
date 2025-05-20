import os

case_id_col = {}
timestamp_col = {}
work_day_col = {}
target_col = {}
resource_col = {}
label_col = {}
pos_label = {}
neg_label = {}


activity_col = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = "labeled_logs_csv_processed"


#### BPIC2011 settings ####
for formula in range(1,5):
    dataset = "bpic2011_f%s"%formula 
    print(dataset)
    
    filename[dataset] = os.path.join(logs_dir, "BPIC11_f%s.csv"%formula)
    
    case_id_col[dataset] = "Case ID"
    target_col[dataset] = "Activity code"
    activity_col[dataset] = "Activity code"
    resource_col[dataset] = "Producer code"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity code", "Producer code", "Section", "Specialism code.1", "group"]
    static_cat_cols[dataset] = ["Diagnosis", "Treatment code", "Diagnosis code", "Specialism code", "Diagnosis Treatment Combination ID"]
    dynamic_num_cols[dataset] = ["Number of executions", "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ["Age"]

#### Sepsis Cases settings ####
datasets = ["sepsis_cases_%s" % i for i in range(1, 5)]

for dataset in datasets:
    
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % dataset)

    target_col[dataset] = "Target"
    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "org:group"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'org:group'] # i.e. event attributes
    static_cat_cols[dataset] = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                       'SIRSCritTemperature', 'SIRSCriteria2OrMore'] # i.e. case attributes that are known from the start
    dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['Age']


#### BPIC2012 settings ####
bpic2012_dict = {"bpic2012_cancelled": "bpic2012_O_CANCELLED-COMPLETE.csv",
                 "bpic2012_accepted": "bpic2012_O_ACCEPTED-COMPLETE.csv",
                 "bpic2012_declined": "bpic2012_O_DECLINED-COMPLETE.csv"
                }

for dataset, fname in bpic2012_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"
    target_col[dataset] = "label"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Resource"]
    static_cat_cols[dataset] = []
    dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['AMOUNT_REQ']

 
