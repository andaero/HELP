import pandas as pd
from sklearn.metrics import adjusted_rand_score
import glob


def get_group_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    accurate_events = 0  # Count of correctly parsed log lines
    accurate_templates = 0  # Count of correctly parsed templates

    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                accurate_templates += 1
                error = False

        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')

    GA = float(accurate_events) / series_groundtruth.size
    FGA = float(accurate_templates) / len(series_groundtruth.value_counts())
    # print(series_groundtruth.value_counts())
    print(series_groundtruth.size)
    return GA, FGA

def get_adjusted_rand_index(series_groundtruth, series_parsedlog):
    return adjusted_rand_score(series_groundtruth, series_parsedlog)


# get all files in the results directory as the predicted Path
def results_per_dataset(dataset):
    results = pd.DataFrame(columns=['Dataset', 'Predicted', 'NumGroups', 'GA', 'FGA', 'ARI'])
    for predictedPath in glob.glob(f'../Iudex/results_new/{dataset}/*.csv'):
        # Load and sort the CSV files by 'Content'
        groundTruthPath = f'../../full_dataset/{dataset}/{dataset}_full.log_structured.csv'
        ground_truth_df = pd.read_csv(groundTruthPath)
        if dataset == 'Thunderbird':
            ground_truth_df = ground_truth_df.head(200001)
        predicted_df = pd.read_csv(predictedPath)
        # Ensure the 'Content' columns match
        if not (ground_truth_df['Content'] == predicted_df['Content']).all():
            raise ValueError("The 'Content' columns in the two files do not match.")

        # store all the results in a dictionary to be exported to a CSV file
        dataset = groundTruthPath.split('/')[-2]
        predicted = predictedPath.split('/')[-1]
        numGroups = predicted_df['PatternID'].nunique()
        GA, FGA = get_group_accuracy(ground_truth_df['EventId'], predicted_df['PatternID'])
        ARI = get_adjusted_rand_index(ground_truth_df['EventId'], predicted_df['PatternID'])
        temp_df = pd.DataFrame({'Dataset': dataset, 'Predicted': predicted, 'NumGroups': numGroups, 'GA': GA, 'FGA': FGA, 'ARI': ARI}, index=[0])
        results = pd.concat([results, temp_df], ignore_index=True)
        print(results)
    results.to_csv(f'{dataset}_results.csv', index=False)
    print('exported to:', f'{dataset}_results.csv')

        # Calculate GA a

def get_specific_results(predictedPath, dataset, folder_name='wc', hyp_scan=False, batch=False):
    groundTruthPath = f'../../full_dataset/{dataset}/{dataset}_full.log_structured.csv'
    if hyp_scan:
        predictedPath = f'../Iudex/results_new_hyp_scan_{folder_name}/{dataset}/{predictedPath}.csv'
    elif batch:
        predictedPath = f'../Iudex/results_new_batch/{dataset}/{predictedPath}.csv'
    else:
        predictedPath = f'../Iudex/results_new/{dataset}/{predictedPath}'
    ground_truth_df = pd.read_csv(groundTruthPath)
    if dataset == 'Thunderbird':
        ground_truth_df = ground_truth_df.head(200001) # reality is 200001
    elif dataset == 'BGL':
        ground_truth_df = ground_truth_df.head(500001)
    elif len(ground_truth_df) > 400000:
        ground_truth_df = ground_truth_df.head(400001)
    # print number of categories in ground truth based on Event ID
    num_cat = ground_truth_df['EventId'].nunique()
    print(f'Num of ground truth categories: {num_cat}')
    predicted_df = pd.read_csv(predictedPath)
    # Ensure the 'Content' columns match
    if not (ground_truth_df['Content'] == predicted_df['Content']).all():
        raise ValueError("The 'Content' columns in the two files do not match.")

    # store all the results in a dictionary to be exported to a CSV file
    predicted = predictedPath.split('/')[-1]
    numGroups = predicted_df['PatternID'].nunique()
    GA, FGA = get_group_accuracy(ground_truth_df['EventId'], predicted_df['PatternID'])
    ARI = get_adjusted_rand_index(ground_truth_df['EventId'], predicted_df['PatternID'])
    temp_df = pd.DataFrame({'Dataset': dataset, 'Predicted': predicted, 'NumGroups': numGroups, 'GA': GA, 'FGA': FGA, 'ARI': ARI}, index=[0])
    print('Dataset:', dataset)
    print('Predicted:', predicted)
    print('NumGroups:', numGroups)
    print('GA:', GA)
    print('FGA:', FGA)
    print('ARI:', ARI)
    return temp_df

def merge_all_predictions(groundTruthPath, datapath):
    results = pd.read_csv(groundTruthPath, usecols=['Content', 'EventId'])
    print(results)
    for dataset in glob.glob(f'../Iudex/results_new/{datapath}/*.csv'):
        temp_df = pd.read_csv(dataset)
        predicted_vals = temp_df['PatternID'].values
        # merge the two dataframes and call the patternID column for the temp_df the name of the file
        results = pd.concat([results, pd.DataFrame({dataset.split('/')[-1]: predicted_vals})], axis=1)
        print(results)
    results.to_csv(f'{datapath}_all_results.csv', index=False)
    print('dataset:', datapath)
    print('exported to:', f'{datapath}_all_results.csv')

def find_diff_lines(predictedPath, dataset):
    groundTruthPath = f'../../full_dataset/{dataset}/{dataset}_full.log_structured.csv'
    predictedPath = f'../Iudex/results_new_hyp_scan_wc/{dataset}/{predictedPath}'
    ground_truth_df = pd.read_csv(groundTruthPath)
    if dataset == 'Thunderbird':
        ground_truth_df = ground_truth_df.head(200001)
    if dataset == 'BGL':
        ground_truth_df = ground_truth_df.head(500001)
    predicted_df = pd.read_csv(predictedPath)

    # Ensure the 'Content' columns match
    print(len(ground_truth_df['Content']))
    print(len(predicted_df['Content']))
    if not (ground_truth_df['Content'] == predicted_df['Content']).all():
        raise ValueError("The 'Content' columns in the two files do not match.")

    df = get_diff_df(ground_truth_df['EventId'], predicted_df['PatternID'], ground_truth_df['Content'] )
    df.to_csv(f'{dataset}_diff_v2.csv', index=False)
    print('exported to:', f'{dataset}_diff_v2.csv')

def get_diff_df(series_groundtruth, series_parsedlog, series_logs):
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    differences = []  # List of incorrectly parsed log lines

    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                error = False
        if error:
            for logId in logIds:
                differences.append({
                    'index': logId,
                    'Content': series_logs[logId],
                    'parsed_eventId': parsed_eventId,
                    'EventId': series_groundtruth[logId],
                })

    differences_df = pd.DataFrame(differences)
    return differences_df

# def find_same_eventID()

groundTruthPath= '../../full_dataset/Hadoop/Hadoop_full.log_structured.csv'
groundTruthPath2= '../../full_dataset/HealthApp/HealthApp_full.log_structured.csv'

# results_per_dataset('Mac')
# results_per_dataset('Linux')
# results_per_dataset('Hadoop')
# results_per_dataset('Thunderbird')

# get_specific_results(f'batch_upload_100_weighted_merge_Hadoop_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Hadoop')
# get_specific_results(f'weighted_merge_Hadoop_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Hadoop')
# get_specific_results(f'weighted_merge_Hadoop_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Hadoop')

# get_specific_results(f'batch_upload_1000_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'batch_upload_100_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'10k2_batch_upload_100_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'10k2_batch_upload_50_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'50k_batch_upload_10_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'50k2_batch_upload_10_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'50k_batch_upload_100_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'batch_upload_100_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'batch_upload_2_Hadoop_Linux_Mac_Thunderbird_out-750_bs_512_max_epochs_25_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'real_batch_upload_1000_Hadoop_Linux_Mac_Thunderbird_HealthApp_out-750_bs_1028_max_epochs_30_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'real_batch_upload_1_Hadoop_Linux_Mac_Thunderbird_HealthApp_out-750_bs_1028_max_epochs_30_dropout-0_lr-0.0001', 'HealthApp', batch=True)
# get_specific_results(f'Hadoop_Linux_Mac_Thunderbird_HealthApp_out-750_bs_1028_max_epochs_30_dropout-0_lr-0.0001', 'Apache', hyp_scan=True)
# get_specific_results(f'Hadoop_Linux_Mac_Thunderbird_HealthApp_out-750_bs_1028_max_epochs_30_dropout-0_lr-0.0001', 'BGL', hyp_scan=True)
# get_specific_results(f'Proxifier_out-750_bs_2048_max_epochs_50_dropout-0.2_lr-0.0005', 'Proxifier', hyp_scan=True)
# get_specific_results(f'matrix_3_weighted_merge_Linux_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Linux')
# get_specific_results(f'weighted_merge_Linux_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Linux')
# get_specific_results(f'matrix_HL_1_weighted_merge_Linux_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Linux')
# get_specific_results(f'matrix_HH_3_weighted_merge_Linux_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Linux')
# get_specific_results(f'10k2_batch_upload_100_Hadoop_HealthApp_Mac_Thunderbird_out-750_bs_1028_max_epochs_25_dropout-0_lr-0.0001', 'Linux', batch=True)
# get_specific_results(f'10k2_batch_upload_10_Hadoop_HealthApp_Mac_Thunderbird_out-750_bs_1028_max_epochs_25_dropout-0_lr-0.0001', 'Linux', batch=True)
# get_specific_results(f'10k2_batch_upload_50_Hadoop_HealthApp_Mac_Thunderbird_out-750_bs_1028_max_epochs_25_dropout-0_lr-0.0001', 'Linux', batch=True)
# get_specific_results(f'10k2_batch_upload_1000_Hadoop_HealthApp_Mac_Thunderbird_out-750_bs_1028_max_epochs_25_dropout-0_lr-0.0001', 'Linux', batch=True)

# get_specific_results(f'weighted_merge_Thunderbird_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Thunderbird')
# get_specific_results(f'matrix_3_weighted_merge_Thunderbird_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Thunderbird')

# get_specific_results(f'nn_3_weighted_merge_Mac_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Mac')
# get_specific_results(f'Hadoop_HealthApp_Linux_Thunderbird_out-250_bs_512_max_epochs_35_dropout-0_lr-0.0001', 'Mac', True)
# get_specific_results(f'nn_weighted_merge_Mac_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Mac')
# get_specific_results(f'weighted_merge_Mac_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Mac')
# get_specific_results(f'matrix_HL_2_weighted_merge_Mac_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Mac')
# get_specific_results(f'matrix_3_weighted_merge_Mac_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Mac')
# get_specific_results(f'weighted_merge_Mac_naive_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Mac')

# find_diff_lines('weighted_merge_Hadoop_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Hadoop')
# find_diff_lines('weighted_merge_HealthApp_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'HealthApp')
# find_diff_lines('weighted_merge_Linux_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Linux')
# find_diff_lines('weighted_merge_Mac_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Mac')
# find_diff_lines('weighted_merge_Thunderbird_wordcount2_center_rebalance_text-embedding-3-small_sim_0.9.csv', 'Thunderbird')
# find_diff_lines('Hadoop_Linux_Mac_Thunderbird_HealthApp_out-750_bs_1028_max_epochs_30_dropout-0_lr-0.0001.csv', 'Apache')
# merge_all_predictions(groundTruthPath2, 'HealthApp')

# list = ['BGL', 'Spark', 'Hadoop', 'Thunderbird', 'HealthApp', 'Linux', 'Mac']
# for dataset in list:
#     naive = f'{dataset}_naive_ORIGINAL_text-embedding-3-small_sim_0.9.csv'
#     wc = f'{dataset}_wordcount_ORIGINAL_text-embedding-3-small_sim_0.9.csv'
#     get_specific_results(naive, dataset)
