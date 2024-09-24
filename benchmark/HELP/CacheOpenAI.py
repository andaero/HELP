from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import pickle
import numpy as np
import re
from Utils import log_preprocess
import argparse

# Load environment variables from the .env file
load_dotenv(dotenv_path=os.path.join( '../../.env'))
try:
    openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
except Exception as e:
    print("Failed to initialize OpenAI API")


def cache_openai_embeddings(dataset_names, model_names, batchSize=2000, folder_name='naive', naive=False):
    """Get openai embeddings and store them on disk to avoid repeated API calls."""
    for name in dataset_names:
        for model in model_names:
            input_dir = f'../../full_dataset/{name}/{name}_full.log_structured.csv'
            df = pd.read_csv(input_dir)
            # input df only contains Content column, treat as an array of strings
            log_inputs = df['Content'].values
            template_inputs = df['EventTemplate'].values
            batchGetVectors(log_inputs, model, name, batchSize, folder_name, naive)
            print('Stored embeddings for', name, 'and model', model)


def batchGetVectorsWithTemplate(raw_logs, templates, model_name, dataset_names, batchSize, folder_name):
    for batch_start in range(1, len(raw_logs), batchSize):
        batch_end = min(batch_start + batchSize, len(raw_logs))
        logs_batch = raw_logs[batch_start:batch_end]
        template_batch = templates[batch_start:batch_end]
        _, vectors = generateVectorsWithTemplate(logs_batch, template_batch, model_name)
        # check if the directory exists
        if not os.path.exists(f'openai_embeddings/{dataset_names}_{folder_name}_{model_name}'):
            os.makedirs(f'openai_embeddings/{dataset_names}_{folder_name}_{model_name}')
        # store all_vectors on disk
        with open(f'openai_embeddings/{dataset_names}_{folder_name}_{model_name}/{batch_start-1}.pkl', 'wb') as f:
            pickle.dump(vectors, f)
        print('Stored embeddings for', dataset_names, 'and model', model_name, 'batch', batch_start-1)
    return


def batchGetVectors(raw_logs, model_name, dataset_names, batchSize, folder_name, naive):
    for batch_start in range(1, len(raw_logs), batchSize):
        # check if file already exists
        if os.path.exists(f'openai_embeddings/{dataset_names}_{folder_name}_{model_name}/{batch_start-1}.pkl'):
            print('Embeddings already exist for', dataset_names, 'and model', model_name, 'batch', batch_start-1)
            continue
        batch_end = min(batch_start + batchSize, len(raw_logs))
        logs_batch = raw_logs[batch_start:batch_end]
        if naive:
            _, vectors = generateVectorsNaive(logs_batch, model_name)
        else:
            _, vectors = generateVectors_v2(logs_batch, model_name)
        # check if the directory exists
        if not os.path.exists(f'openai_embeddings/{dataset_names}_{folder_name}_{model_name}'):
            os.makedirs(f'openai_embeddings/{dataset_names}_{folder_name}_{model_name}')
        # store all_vectors on disk
        with open(f'openai_embeddings/{dataset_names}_{folder_name}_{model_name}/{batch_start-1}.pkl', 'wb') as f:
            pickle.dump(vectors, f)
        print('Stored embeddings for', dataset_names, 'and model', model_name, 'batch', batch_start-1)
    return

def generateVectors(raw_logs, model_name):
    embedded_texts = []
    for raw_log in raw_logs:
        # slice to only take the first 8000 characters per log
        if len(raw_log) > 8000:
            print('Log is too long, truncating to 8000 characters')
        log = raw_log[:8000]
        # find the number of words in the log
        num_words = len(raw_log[:8000].split())
        embedded_texts.append(str(num_words) + '\n' + log + '\n' + str(num_words))
    vectors = createEmbeddings(embedded_texts, model_name)
    return embedded_texts, vectors

def generateVectors_v2(raw_logs, model_name):
    # get the number of characters in each word and the number of words in each log
    embedded_texts = []
    num_words_tot = []
    countChanged = 0
    for raw_log in raw_logs:
        # slice to only take the first 8000 characters per log
        if len(raw_log) > 8000:
            print('Log is too long, truncating to 8000 characters')
        og_log = raw_log[:8000]
        log = log_preprocess(og_log)
        if log != og_log:
            countChanged += 1
        # find the number of words in the log
        num_words = len(log.split())
        num_words_tot.append(num_words)
        embedded_texts.append(str(num_words) + '\n' + log + '\n' + str(num_words))
    print('Changed logs per batch:', countChanged)
    vectors = createEmbeddings(embedded_texts, model_name)
    return embedded_texts, vectors

def generateVectorsNaive(raw_logs, model_name):
    embedded_texts = []
    for raw_log in raw_logs:
        # slice to only take the first 8000 characters per log
        log = raw_log[:8000]
        embedded_texts.append(log)
    vectors = createEmbeddings(embedded_texts, model_name)
    return embedded_texts, vectors

def generateVectorsWithSpace(raw_logs, model_name):
    embedded_texts = []
    for i, raw_log in enumerate(raw_logs):
        # slice to only take the first 8000 characters per log
        log = raw_log[:8000]
        # set emphasized_str to emphasize the structure of the log
        # first parse the log by the template, replacing the letters that match the template with the same letter in uppercase
        # emphasized_str = emphasize_and_tag_words(log, templates[i])
        # print(emphasized_str)
        # embedded_texts.append(emphasized_str)
        embedded_texts.append(' ' + log)
    vectors = createEmbeddings(embedded_texts, model_name)
    return embedded_texts, vectors


def generateVectorsWithTemplate(raw_logs, templates, model_name):
    embedded_texts = []
    for i, raw_log in enumerate(raw_logs):
        # slice to only take the first 8000 characters per log
        log = raw_log[:8000]
        # set emphasized_str to emphasize the structure of the log
        # first parse the log by the template, replacing the letters that match the template with the same letter in uppercase
        # emphasized_str = emphasize_and_tag_words(log, templates[i])
        # print(emphasized_str)
        # embedded_texts.append(emphasized_str)
        embedded_texts.append(templates[i] + '\n' + log + '\n' + templates[i])
    vectors = createEmbeddings(embedded_texts, model_name)
    return embedded_texts, vectors


def createEmbeddings(embedded_texts, model_name):
    res = openai.embeddings.create(model=model_name, input=embedded_texts)
    # get list of vectors from the response
    vectors = [d.embedding for d in res.data]
    return vectors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_wc', action='store_true', default=False, help='Enable or disable wc embeddings')
    args = parser.parse_args()
    names = ['Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux', 'Mac'
            'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Zookeeper']
    naive = args.no_wc
    if naive:
        print('running naive cache')
        cache_openai_embeddings(names, ['text-embedding-3-small'], folder_name='naive', naive=naive)
    cache_openai_embeddings(names, ['text-embedding-3-small'], folder_name='wordcount')