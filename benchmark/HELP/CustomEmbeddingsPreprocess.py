# imports
from typing import List, Tuple  # for type hints

import numpy as np  # for manipulating arrays
import pandas as pd  # for manipulating data in dataframes
import pickle  # for saving the embeddings cache
import random  # for generating run IDs
from sklearn.model_selection import train_test_split  # for splitting train & test data
import torch  # for matrix optimization
from collections import defaultdict
from Utils import cos_sim, get_openai_embeddings, get_openai_embeddings_v2
import os



def process_input_data(df: pd.DataFrame, num_pairs_to_embed: int) -> pd.DataFrame:
    # Initialize lists to hold the pairs
    similar_pairs = []
    dissimilar_pairs = []
    print(df)
    # Convert dataframe to a list of tuples for faster processing
    data = df[['Content', 'EventId']].values.tolist()
    print('converted to list')
    # Create buckets for each label, up to 1000 for each bucket
    label_buckets = defaultdict(list)
    for content, label in data:
        if len(label_buckets[label]) < 1000:
            label_buckets[label].append(content)
    print('created buckets')
    labels = list(label_buckets.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(labels)
    label_idx = 0

    while len(similar_pairs) < num_pairs_to_embed or len(dissimilar_pairs) < (num_pairs_to_embed * 2):
        # Select a random label for similar pairs
        if len(similar_pairs) < num_pairs_to_embed:
            label = labels[label_idx]
            label_idx = (label_idx + 1) % len(labels)  # Move to the next label
            if len(label_buckets[label]) > 1:
                text1, text2 = np.random.choice(label_buckets[label], 2, replace=False)
                similar_pairs.append((text1, text2))

        # Select two random labels for dissimilar pairs
        if len(dissimilar_pairs) < (num_pairs_to_embed * 2):
            label1, label2 = np.random.choice(labels, 2, replace=False)
            if label_buckets[label1] and label_buckets[label2]:
                text1 = np.random.choice(label_buckets[label1])
                text2 = np.random.choice(label_buckets[label2])
                dissimilar_pairs.append((text1, text2))

    # Create dataframes for similar and dissimilar pairs
    similar_df = pd.DataFrame(similar_pairs, columns=['text_1', 'text_2'])
    similar_df['label'] = 1

    dissimilar_df = pd.DataFrame(dissimilar_pairs, columns=['text_1', 'text_2'])
    dissimilar_df['label'] = 0

    # Combine similar and dissimilar pairs
    result_df = pd.concat([similar_df, dissimilar_df], ignore_index=True)

    return result_df


def preprocess_v3(datasets, name, num_pairs_to_embed):
    df = pd.read_csv(datasets[0])
    # process input data
    df = process_input_data(df, num_pairs_to_embed)
    for dataset_path in datasets[1:]:
        df2 = pd.read_csv(dataset_path)
        df2 = process_input_data(df2, num_pairs_to_embed)
        df = pd.concat([df, df2])

    # view data
    print('test')
    print(df)
    df_size = int(len(df) / 1000)
    print('df size: ', df_size)

    # split data into train and test sets
    test_fraction = 0.2  # 0.5 is fairly arbitrary
    random_seed = 70  # random seed is arbitrary, but is helpful in reproducibility
    train_df, test_df = train_test_split(
        df, test_size=test_fraction, stratify=df["label"], random_state=random_seed
    )
    train_df.loc[:, "dataset"] = "train"
    test_df.loc[:, "dataset"] = "test"

    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    df = pd.concat([train_df, test_df])

    embeddings_1 = get_openai_embeddings(df, 'text-embedding-3-small', 'text_1')
    embeddings_2 = get_openai_embeddings(df, 'text-embedding-3-small', 'text_2')
    print('exporting embeddings')
    # check if the directory exists
    if not os.path.exists(f'Matrix_embeddings_wc/{name}'):
        os.makedirs(f'Matrix_embeddings_wc/{name}')
    with open(f'Matrix_embeddings_wc/{name}/embeddings_1_{df_size}k_r5.pkl', 'wb') as f:
        pickle.dump(embeddings_1, f)
    with open(f'Matrix_embeddings_wc/{name}/embeddings_2_{df_size}k_r5.pkl', 'wb') as f:
        pickle.dump(embeddings_2, f)

    print('Applying cosine similarity to embeddings')
    # create column of cosine similarity between embeddings
    cos_sim_list = [cos_sim(e1, e2) for e1, e2 in zip(embeddings_1, embeddings_2)]
    df["cosine_similarity"] = cos_sim_list

    print('Exporting to csv')
    df.to_csv(f"Matrix_embeddings_wc/{name}/pairs_{df_size}k_r5.csv", index=False)
    print(df)

def preprocess_v4(datasets, num_pairs_to_embed):
    for dataset_path in datasets:
        # check if embeddings and pairs already exist
        name = dataset_path.split('/')[-1].split('_')[0]
        tot = num_pairs_to_embed * 3 / 1000
        if os.path.exists(f'Matrix_embeddings_wc/{name}/pairs_{int(tot)}k_r5.csv'):
            print(f'{name} embeddings and pairs already exist')
            continue

        df = pd.read_csv(dataset_path)
        df = process_input_data(df, num_pairs_to_embed)
        # each dataset path is of the form '../evaluation/BGL_diff_v2.csv', parse to just get out BGL
        # view data
        print(name)
        print(df)
        df_size = int(len(df) / 1000)
        print('df size: ', df_size)

        # split data into train and test sets
        test_fraction = 0.2  # 0.5 is fairly arbitrary
        random_seed = 70  # random seed is arbitrary, but is helpful in reproducibility
        train_df, test_df = train_test_split(
            df, test_size=test_fraction, stratify=df["label"], random_state=random_seed
        )
        train_df.loc[:, "dataset"] = "train"
        test_df.loc[:, "dataset"] = "test"

        train_df['dataset'] = 'train'
        test_df['dataset'] = 'test'
        df = pd.concat([train_df, test_df])

        embeddings_1 = get_openai_embeddings(df, 'text-embedding-3-small', 'text_1')
        embeddings_2 = get_openai_embeddings(df, 'text-embedding-3-small', 'text_2')
        print('exporting embeddings')
        # check if the directory exists
        if not os.path.exists(f'Matrix_embeddings_wc/{name}'):
            os.makedirs(f'Matrix_embeddings_wc/{name}')
        with open(f'Matrix_embeddings_wc/{name}/embeddings_1_{df_size}k_r5.pkl', 'wb') as f:
            pickle.dump(embeddings_1, f)
        with open(f'Matrix_embeddings_wc/{name}/embeddings_2_{df_size}k_r5.pkl', 'wb') as f:
            pickle.dump(embeddings_2, f)

        print('Applying cosine similarity to embeddings')
        # create column of cosine similarity between embeddings
        cos_sim_list = [cos_sim(e1, e2) for e1, e2 in zip(embeddings_1, embeddings_2)]
        df["cosine_similarity"] = cos_sim_list

        print('Exporting to csv')
        df.to_csv(f"Matrix_embeddings_wc/{name}/pairs_{df_size}k_r5.csv", index=False)
        print(df)


# input parameters
num_pairs_to_embed = 9000

apache = '../../full_dataset/Apache/Apache_full.log_structured.csv'
bgl = '../../full_dataset/BGL/BGL_full.log_structured.csv'
hadoop = '../../full_dataset/Hadoop/Hadoop_full.log_structured.csv'
hdfs = '../../full_dataset/HDFS/HDFS_full.log_structured.csv'
health_app = '../../full_dataset/HealthApp/HealthApp_full.log_structured.csv'
hpc = '../../full_dataset/HPC/HPC_full.log_structured.csv'
linux = '../../full_dataset/Linux/Linux_full.log_structured.csv'
mac = '../../full_dataset/Mac/Mac_full.log_structured.csv'
openssh = '../../full_dataset/OpenSSH/OpenSSH_full.log_structured.csv'
openstack = '../../full_dataset/OpenStack/OpenStack_full.log_structured.csv'
proxifier = '../../full_dataset/Proxifier/Proxifier_full.log_structured.csv'
spark = '../../full_dataset/Spark/Spark_full.log_structured.csv'
thunderbird = '../../full_dataset/Thunderbird/Thunderbird_full.log_structured.csv'
zookeeper = '../../full_dataset/Zookeeper/Zookeeper_full.log_structured.csv'


datasets = [apache, bgl, hadoop, hdfs, health_app, hpc, linux, mac,
            openssh, openstack, proxifier, spark, thunderbird, zookeeper]
preprocess_v4(datasets, num_pairs_to_embed)