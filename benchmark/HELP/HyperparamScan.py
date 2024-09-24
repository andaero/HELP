from Customizing_embeddings_NN import train, train_prep, train_prep_v2
from RebalanceMergeNN import load_data
import sys
# sys.path.append("/Users/andyxu/LogTesting/")
from benchmark.evaluation.fga_accuracy import get_specific_results
from ParseMultiple import parse_templates_multiple
import pandas as pd
import os


# name = 'Hadoop_HealthApp_Mac_Thunderbird'
names = ['Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux', 'Mac',
        'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Zookeeper']

for name in names:
    # all but name
    input = [x for x in names if x != name]
    benchmark_name = name
    print(f'Running {benchmark_name}')
    folder_name = 'wc'
    size = 40 # 96
    batch_sizes = [2048]
    max_epochs = [50]
    learning_rates = [0.0005]
    dropouts = [0.2]
    output_lens = [750]

    models = ['text-embedding-3-small']
    sim_thresholds = [0.9]

    print('getting data')
    print('got data')
    results = pd.DataFrame(columns=['Dataset', 'Predicted', 'NumGroups', 'GA', 'FGA', 'ARI'])
    for bs in batch_sizes:
        for me in max_epochs:
            for lr in learning_rates:
                for d in dropouts:
                    for ol in output_lens:
                        print(f'Batch size: {bs}, Max epochs: {me}, Learning rate: {lr}, Dropout: {d}, Output length: {ol}')
                        # check if the training file exists
                        name = benchmark_name
                        hyp_str = f"{name}_out-{ol}_bs_{bs}_max_epochs_{me}_dropout-{d}_lr-{lr}"
                        if not os.path.exists(f"nn_models_hyp_scan_{folder_name}/{name}/{hyp_str}.pth"):
                            e1_train, e2_train, s_train, e1_test, e2_test, s_test = train_prep_v2(input, size)
                            print('Training model')
                            # concat the strings to get the model name by seperating with _
                            print(name)
                            nn_model_str = train(e1_train, e2_train, s_train, e1_test, e2_test, s_test, name,
                                                 hyp_str, ol, bs, me, lr, d, folder_name)
                        else:
                            nn_model_str = hyp_str
                        if not os.path.exists(f"results_new_hyp_scan_{folder_name}/{benchmark_name}/{nn_model_str}.csv"):
                            print('Loading data')
                            load_data([benchmark_name], models, nn_model_str, name, ol, sim_thresholds, folder_name, verbose=False)
                        temp_df = get_specific_results(nn_model_str, benchmark_name, folder_name, hyp_scan=True )
                        results = pd.concat([results, temp_df], ignore_index=True)
                        print('results')
                        print(results)

    # check if file exists, if it does append this to that csv
    if os.path.exists(f"Matrix_embeddings_{folder_name}/{benchmark_name}/hyperparam_scan.csv"):
        old_results = pd.read_csv(f"Matrix_embeddings_{folder_name}/{benchmark_name}/hyperparam_scan.csv")
        results = pd.concat([old_results, results], ignore_index=True)
    # export to csv
    results.to_csv(f"Matrix_embeddings_{folder_name}/{benchmark_name}/hyperparam_scan.csv", index=False)
    print('Exported results to csv')

    # run llm prompting
    models = ["claude-3.5-sonnet"]
    parse_templates_multiple(name, nn_model_str, models, max_logs=1, num_threads=10)