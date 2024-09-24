import faiss
import numpy as np
def simsearchPatterns(vectorStore, queryVector, selfIndex=False, debug=False):
    # convert query vector to numpy array
    queryVector = np.array([queryVector], dtype=np.float32)
    # normalize the query vector
    queryVector = queryVector / np.linalg.norm(queryVector)

    # Search for the most similar vector
    if selfIndex:
        distances, index = vectorStore.search(queryVector, 2)
        if debug:
            return distances[0], index[0]
        return distances[0][1], index[0][1]
    else:
        distances, index = vectorStore.search(queryVector, 1)
        return distances[0][0], index[0][0]

def batchSimsearchPatterns(vectorStore, queryVectors, selfIndex=False):
    # convert query vectors to numpy array
    queryVectors = np.array(queryVectors, dtype=np.float32)
    # normalize the query vectors
    queryVectors = queryVectors / np.linalg.norm(queryVectors, axis=1)[:, np.newaxis]

    # Search for the most similar vectors
    if selfIndex:
        distances, index = vectorStore.search(queryVectors, 2)
        return distances[:, 1], index[:, 1]
    else:
        distances, index = vectorStore.search(queryVectors, 1)
        return distances[:, 0], index[:, 0]

