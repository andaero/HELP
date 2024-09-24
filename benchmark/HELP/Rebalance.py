import faiss
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from VectorSearch import simsearchPatterns
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Load environment variables from the .env file
load_dotenv(dotenv_path=os.path.join( '../../.env'))
try:
    openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
except Exception as e:
    print("Failed to initialize OpenAI API")

def load_data(dataset_names, model_names, simThresholds, folder_name='naive'):
    for name in dataset_names:
        for model in model_names:
            for simThreshold in simThresholds:
                print(f'Model: {model} simThreshold: {simThreshold}')
                input_dir = f'../../full_dataset/{name}/{name}_full.log_structured.csv'
                df = pd.read_csv(input_dir)
                # input df only contains Content column, treat as an array of strings
                log_inputs = df['Content'].values
                # read in only the Content and EventId columns
                ground_true_df = df[['Content', 'EventId']]
                patternStore = writeLogsToPatternsV5(log_inputs, model, folder_name, name, simThreshold)
                print(f'Number of unique patterns: {len(set(patternStore))}')
                # create a dataframe from the pattern store and corresponding log_inputs
                print(len(log_inputs))
                print(len(patternStore))
                pattern_df = pd.DataFrame({'Content': log_inputs, 'PatternID': patternStore})
                # check if the directory exists
                if not os.path.exists(f'results_new/{name}'):
                    os.makedirs(f'results_new/{name}')
                # export the dataframe to a csv file
                pattern_df.to_csv(f'results_new/{name}/{name}_{folder_name}_center_rebalance_{model}_sim_{simThreshold}.csv', index=False)
        print(pattern_df)

def generateVectors(raw_logs, model_name):
    embedded_texts = []
    for raw_log in raw_logs:
        # slice to only take the first 8000 characters per log
        log = raw_log[:8000]
        embedded_texts.append(log)
    vectors = createEmbeddings(embedded_texts, model_name)
    return embedded_texts, vectors

def createEmbeddings(embedded_texts, model_name):
    res = openai.embeddings.create(model=model_name, input=embedded_texts)
    # get list of vectors from the response
    vectors = [d.embedding for d in res.data]
    return vectors

def vectorStoreInitv2(logs, model_name, patternStore, patternStoreKey, pattern_id=0):
    embedded_texts, logVector = generateVectors([logs[0]], model_name)
    logVector = logVector[0]
    vectorStore = faiss.IndexFlatIP(len(logVector))
    patternStore.append(pattern_id)
    patternStoreKey.append(pattern_id)
    logVector = np.array([logVector], dtype=np.float32)
    vectorStore.add(logVector)
    return vectorStore, patternStore, patternStoreKey

def writeLogsToPatternsV4(logs, model_name, similarityThreshold, batchSize=2000):
    pattern_id = 0
    patternStore = []
    patternStoreKey = []
    # create vector store
    vectorStore, patternStore, patternStoreKey = vectorStoreInitv2(logs, model_name, patternStore, patternStoreKey, pattern_id)
    logs = logs[1:]
    i = 1
    for batch_start in range(0, len(logs), batchSize):
        batch_end = min(batch_start + batchSize, len(logs))
        logs_batch = logs[batch_start:batch_end]
        all_embedded_texts, logVectors = generateVectors(logs_batch, model_name)
        for logVector in logVectors:
            # search for similar patterns
            cosSim, index = simsearchPatterns(vectorStore, logVector)
            # add the log vector to the vector store
            logVector = np.array([logVector], dtype=np.float32)
            if cosSim < similarityThreshold:
                # create a new pattern
                pattern_id += 1
                patternStore.append(pattern_id)
                patternStoreKey.append(pattern_id)
                vectorStore.add(logVector)
            else:
                # find the pattern id that corresponds to the index
                patternIdFromStore = patternStoreKey[index]

                # incrementally update the centroid of the pattern

                n = patternStore.count(patternIdFromStore)
                reconstructed_vector = np.zeros(logVector.shape[1], dtype=np.float32)
                old_centroid = vectorStore.reconstruct(int(index), reconstructed_vector)
                new_centroid = old_centroid + (logVector - old_centroid) / (n + 1)
                new_centroid = new_centroid / np.linalg.norm(new_centroid)
                # remove the old centroid by index
                vectorStore.remove_ids(np.array([index], dtype=np.int64))
                # add the new centroid
                vectorStore.add(new_centroid)

                del patternStoreKey[index]
                patternStoreKey.append(patternIdFromStore)

                patternStore.append(patternIdFromStore)
            i += 1
        print("number of pattern ids: ", pattern_id)
        print("number of logs processed: ", i)
    return patternStore

def get_embeddings_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
def writeLogsToPatternsV5(logs, model_name, folder_name, dataset, similarityThreshold, batchSize=2000):
    pattern_id = 0
    patternStore = []
    patternStoreKey = []
    # create vector store
    vectorStore, patternStore, patternStoreKey = vectorStoreInitv2(logs, model_name, patternStore, patternStoreKey, pattern_id)
    logs = logs[1:]
    i = 1
    # allVectors = get_embeddings_from_file(f'openai_embeddings/{dataset}_naive_{model_name}.pkl')
    # print("got vectors ", len(allVectors))
    for batch_start in range(0, len(logs), batchSize):
        # batch_end = min(batch_start + batchSize, len(logs))
        # logVectors = allVectors[batch_start:batch_end]
        logVectors = get_embeddings_from_file(f'openai_embeddings/{dataset}_{folder_name}_{model_name}/{batch_start}.pkl')
        for logVector in logVectors:
            # search for similar patterns
            cosSim, index = simsearchPatterns(vectorStore, logVector)
            # add the log vector to the vector store
            logVector = np.array([logVector], dtype=np.float32)
            if cosSim < similarityThreshold:
                # create a new pattern
                pattern_id += 1
                patternStore.append(pattern_id)
                patternStoreKey.append(pattern_id)
                vectorStore.add(logVector)
            else:
                # find the pattern id that corresponds to the index
                patternIdFromStore = patternStoreKey[index]

                # incrementally update the centroid of the pattern

                n = patternStore.count(patternIdFromStore)
                reconstructed_vector = np.zeros(logVector.shape[1], dtype=np.float32)
                old_centroid = vectorStore.reconstruct(int(index), reconstructed_vector)
                new_centroid = old_centroid + (logVector - old_centroid) / (n + 1)
                new_centroid = new_centroid / np.linalg.norm(new_centroid)
                # remove the old centroid by index
                vectorStore.remove_ids(np.array([index], dtype=np.int64))
                # add the new centroid
                vectorStore.add(new_centroid)

                del patternStoreKey[index]
                patternStoreKey.append(patternIdFromStore)
                patternStore.append(patternIdFromStore)
            i += 1
        print("number of pattern ids: ", pattern_id)
        print("number of logs processed: ", i)
    return patternStore


raw_logs = ['org.apache.hadoop.hdfs.server.namenode.NameNode: STARTUP_MSG: ',
            'org.apache.hadoop.hdfs.server.namenode.FSNamesystem: Registered FSNamesystemStateMBean and NameNodeMXBean',
            'org.apache.hadoop.hdfs.server.namenode.FSEditLog: Starting log segment at 1',
            'org.apache.hadoop.hdfs.server.namenode.FSNamesystem: Starting services required for active state',
            'org.apache.hadoop.hdfs.server.namenode.FSNamesystem: Starting services required for standby state']

# patternStore = writeLogsToPatterns(raw_logs, 0.9)

# create_batch_file(raw_logs, "batch_requests.jsonl")
# load_data_batch(['Hadoop'], 0.9)

# vectors = generateVectors(raw_logs)

models = ['text-embedding-3-small']
simThresholds = [0.9]
load_data(['Hadoop'], models, simThresholds, folder_name='wc')
