from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pickle
import re

# Load environment variables from the .env file
load_dotenv(dotenv_path=os.path.join( '../../.env'))
try:
    openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
except Exception as e:
    print("Failed to initialize OpenAI API")


def get_openai_embeddings(df, model, col, batchSize=2000):
    # input df only contains Content column, treat as an array of strings
    log_inputs = df[col].values
    vectors = batchGetVectors(log_inputs, model, batchSize)
    return vectors


def batchGetVectors(raw_logs, model_name, batchSize):
    totalVectors = []
    for batch_start in range(0, len(raw_logs), batchSize):
        batch_end = min(batch_start + batchSize, len(raw_logs))
        logs_batch = raw_logs[batch_start:batch_end]
        vectors = generateVectors(logs_batch, model_name)
        print('Generated embeddings for batch', batch_start)
        totalVectors += vectors
    return totalVectors

def generateVectors(raw_logs, model_name):
    embedded_texts = []
    for raw_log in raw_logs:
        # slice to only take the first 8000 characters per log
        if len(raw_log) > 8000:
            print('Log is too long, truncating to 8000 characters')
        log = raw_log[:8000]
        # find the number of words in the log
        num_words = len(log.split())
        embedded_texts.append(str(num_words) + '\n' + log + '\n' + str(num_words))
    vectors = createEmbeddings(embedded_texts, model_name)
    return vectors

def createEmbeddings(embedded_texts, model_name):
    res = openai.embeddings.create(model=model_name, input=embedded_texts)
    # get list of vectors from the response
    vectors = [d.embedding for d in res.data]
    return vectors

def get_openai_embeddings_v2(df, model, col, batchSize=2000):
    # input df only contains Content column, treat as an array of strings
    log_inputs = df[col].values
    vectors = batchGetVectors_v2(log_inputs, model, batchSize)
    return vectors

def batchGetVectors_v2(raw_logs, model_name, batchSize):
    totalVectors = []
    for batch_start in range(0, len(raw_logs), batchSize):
        batch_end = min(batch_start + batchSize, len(raw_logs))
        logs_batch = raw_logs[batch_start:batch_end]
        vectors = generateVectors_v2(logs_batch, model_name)
        print('Generated embeddings for batch', batch_start)
        totalVectors += vectors
    return totalVectors

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
    return vectors

def log_preprocess(log):
    old_log = log
    # Patterns and their replacements
    patterns = {
        r'((?<=[^A-Za-z0-9])|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))((?=[^A-Za-z0-9])|$)': '$$ipv6_address$$',
        r'((?<=[^A-Za-z0-9])|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})((?=[^A-Za-z0-9])|$)': '$$ipv4_address$$',
        r'((?<=[^A-Za-z0-9])|^)([0-9a-f]{6,} ?){3,}((?=[^A-Za-z0-9])|$)': '$$mac_address$$',
        r'((?<=[^A-Za-z0-9])|^)([0-9A-F]{4} ?){4,}((?=[^A-Za-z0-9])|$)': '$$uuid$$',
        r'((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)': '$$hex_number$$',
        r'((?<=[^A-Za-z0-9])|^)([\-\+]?d+)((?=[^A-Za-z0-9])|$)': '$$number$$',
        r'(?<=executed cmd )(".+?")': '$$cmd$$',
        r'((?<=^)|(?<=[^A-Za-z0-9]))([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})(:[0-9]{1,5})((?=$)|(?=[^A-Za-z0-9]))': '$$hostname_with_port$$'
    }
    for pattern, replacement in patterns.items():
        log = re.sub(pattern, replacement, log)
    return log

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

