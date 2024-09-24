import faiss
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from VectorSearch import simsearchPatterns
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import torch
from EmbeddingsModel import EmbeddingsModel
from CacheOpenAI import generateVectors, generateVectors_v2

# Load environment variables from the .env file
load_dotenv(dotenv_path=os.path.join( '../../.env'))
try:
    openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
except Exception as e:
    print("Failed to initialize OpenAI API")

def load_data(dataset_names, model_names, nn_model, nn_folder_name, output_dim, simThresholds, folder_name='naive', verbose=True):
    for name in dataset_names:
        for model in model_names:
            for simThreshold in simThresholds:
                if folder_name == 'wc':
                    folder_name = 'wordcount'
                print(f'Model: {model} simThreshold: {simThreshold}')
                input_dir = f'../../full_dataset/{name}/{name}_full.log_structured.csv'
                df = pd.read_csv(input_dir)
                # input df only contains Content column, treat as an array of strings
                log_inputs = df['Content'].values
                # read in only the Content and EventId columns
                patternStore = writeLogsToPatternsV5(log_inputs, model, nn_model, nn_folder_name, output_dim, folder_name, name, simThreshold, verbose)
                print(f'Number of unique patterns: {len(set(patternStore))}')
                # create a dataframe from the pattern store and corresponding log_inputs
                print(len(log_inputs))
                print(len(patternStore))
                pattern_df = pd.DataFrame({'Content': log_inputs, 'PatternID': patternStore})
                # check if the directory exists
                if not os.path.exists(f'results_new_hyp_scan_wc/{name}'):
                    os.makedirs(f'results_new_hyp_scan_wc/{name}')
                # export the dataframe to a csv file
                pattern_df.to_csv(f'results_new_hyp_scan_wc/{name}/{nn_model}.csv', index=False)
        print(pattern_df)

def createEmbeddings(embedded_texts, model_name):
    res = openai.embeddings.create(model=model_name, input=embedded_texts)
    # get list of vectors from the response
    vectors = [d.embedding for d in res.data]
    return vectors

def vectorStoreInitv2(logs, nn, model_name, patternStore, patternStoreKey, pattern_id=0):
    embedded_texts, logVector = generateVectors([logs[0]], model_name)
    logVector = logVector[0]
    logVector = np.array([logVector], dtype=np.float32)
    # multiply the log vector by the matrix
    print("logVector shape: ", logVector.shape)
    logTensor = torch.tensor(logVector)
    output = nn(logTensor)
    logVector = output.detach().numpy()
    print("logVector shape after nn: ", logVector.shape)
    vectorStore = faiss.IndexFlatIP(logVector.shape[1])
    patternStore.append(pattern_id)
    patternStoreKey.append(pattern_id)
    vectorStore.add(logVector)
    return vectorStore, patternStore, patternStoreKey

def get_embeddings_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_matrix(matrix_path):
    with open(matrix_path, 'rb') as f:
        return np.array(pickle.load(f))

def merge_centroids(vectorStore, simThreshold, patternStore, patternStoreKey):
    i = 0
    while i < vectorStore.ntotal:
        # get the first vector from the vector store
        reconstructed_vector = np.zeros(vectorStore.d, dtype=np.float32)
        v = vectorStore.reconstruct(i, reconstructed_vector)
        # search for vectors that are similar to the first vector in the vector store, excluding the first vector itself
        cosSim, index2 = simsearchPatterns(vectorStore, v, selfIndex=True)
        if cosSim >= simThreshold:
            print('merging centroids')
            # merge the vectors
            v2 = vectorStore.reconstruct(int(index2), reconstructed_vector)
            # avg of the two vectors (TODO: Weight merging based on # of vectors in the cluster)
            new_centroid = (v + v2) / 2
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            # remove the old centroids
            vectorStore.remove_ids(np.array([i, index2], dtype=np.int64))
            # add the new centroid
            new_centroid = np.expand_dims(new_centroid, axis=0)
            vectorStore.add(new_centroid)

            # get the pattern of the first vector
            patternId = patternStoreKey[i]
            # get the pattern of the second vector
            patternId2 = patternStoreKey[index2]
            # check if the two patterns are the same
            if patternId != patternId2:
                # merge the two patterns
                patternStore = [patternId if x == patternId2 else x for x in patternStore]
                patternStoreKey.remove(patternId)
                patternStoreKey.remove(patternId2)
                # add the new pattern id
                patternStoreKey.append(patternId)
            else:
                print('patterns are the same')
        else:
            i += 1
    return vectorStore, patternStore, patternStoreKey

def merge_centroids_weighted(vectorStore, simThreshold, patternStore, patternStoreKey):
    i = 0
    while i < vectorStore.ntotal:
        # get the first vector from the vector store
        reconstructed_vector = np.zeros(vectorStore.d, dtype=np.float32)
        v = vectorStore.reconstruct(i, reconstructed_vector)
        # search for vectors that are similar to the first vector in the vector store, excluding the first vector itself
        cosSim, index2 = simsearchPatterns(vectorStore, v, selfIndex=True)
        if cosSim >= simThreshold:
            print(f'merging centroids {i} {index2}')

            # get the pattern of the first vector
            patternId = patternStoreKey[i]
            # get the pattern of the second vector
            patternId2 = patternStoreKey[index2]
            # calculate the weight of the two vectors
            w1 = patternStore.count(patternId)
            w2 = patternStore.count(patternId2)
            # check if the two patterns are the same
            if patternId != patternId2:
                # merge the two patterns
                patternStore = [patternId if x == patternId2 else x for x in patternStore]
                # remove by value not index because the index will shift
                patternStoreKey.remove(patternId)
                patternStoreKey.remove(patternId2)
                # add the new pattern id
                patternStoreKey.append(patternId)
            else:
                print('patterns are the same')
                return vectorStore, patternStore, patternStoreKey

            # merge the vectors
            v2 = vectorStore.reconstruct(int(index2), reconstructed_vector)

            new_centroid = (v * w1 + v2 * w2) / (w1 + w2)
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            # remove the old centroids
            vectorStore.remove_ids(np.array([i, index2], dtype=np.int64))
            # add the new centroid
            new_centroid = np.expand_dims(new_centroid, axis=0)
            vectorStore.add(new_centroid)
        else:
            i += 1
    return vectorStore, patternStore, patternStoreKey

def writeLogsToPatternsV5(logs, model_name, nn_model, nn_folder_name, output_dim, folder_name, dataset, similarityThreshold, verbose, batchSize=2000):
    pattern_id = 0
    patternStore = []
    patternStoreKey = []
    # create emb model
    nn = EmbeddingsModel(1536, output_dim, 0)
    # name is for ex. Hadoop_HealthApp_Mac_Thunderbird
    nn.load_state_dict(torch.load(f'nn_models_hyp_scan_wc/{nn_folder_name}/{nn_model}.pth'))
    nn.eval()
    vectorStore, patternStore, patternStoreKey = vectorStoreInitv2(logs, nn, model_name, patternStore, patternStoreKey, pattern_id)
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
            logVector = np.array([logVector], dtype=np.float32)
            # multiply the log vector by the matrix
            logTensor = torch.tensor(logVector)
            output = nn(logTensor)
            logVector = output.detach().numpy()
            # reduce the dimensionality of the log vector
            logVector = np.squeeze(logVector)
            # normalize the log vector
            logVector = logVector / np.linalg.norm(logVector)
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
                vectorStore.add(new_centroid)

                del patternStoreKey[index]
                patternStoreKey.append(patternIdFromStore)
                patternStore.append(patternIdFromStore)
            i += 1
        vectorStore, patternStore, patternStoreKey = merge_centroids_weighted(vectorStore, similarityThreshold, patternStore, patternStoreKey)
        if verbose:
            print("number of vectors in the store: ", vectorStore.ntotal)
            print("number of logs processed: ", i)
    return patternStore

