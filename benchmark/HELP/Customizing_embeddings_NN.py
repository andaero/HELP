# imports
from typing import List, Tuple  # for type hints

import numpy as np  # for manipulating arrays
import pandas as pd  # for manipulating data in dataframes
import pickle  # for saving the embeddings cache
import plotly.express as px  # for plots
import torch  # for matrix optimization
import torch.nn.functional as F
from Utils import cos_sim, get_openai_embeddings
import os
from EmbeddingsModel import EmbeddingsModel

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

def train_prep(name, df_size):
    # name = 'Hadoop_HealthApp_Linux'
    # df_size = 72
    df = pd.read_csv(f"Matrix_embeddings_wc/{name}/pairs_{df_size}k_r5.csv")

    # import pickle files
    with open(f'Matrix_embeddings_wc/{name}/embeddings_1_{df_size}k_r5.pkl', 'rb') as f:
        embeddings_1 = pickle.load(f)
    with open(f'Matrix_embeddings_wc/{name}/embeddings_2_{df_size}k_r5.pkl', 'rb') as f:
        embeddings_2 = pickle.load(f)

    print('Embeddings loaded')

    for dataset in ["train", "test"]:
        data = df[df["dataset"] == dataset]
        a, se = accuracy_and_se(data["cosine_similarity"], data["label"])
        print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")

    # Extract and convert embeddings and labels from dataframe
    embeddings_1 = np.asarray(embeddings_1)
    print('embeddings_1 extracted', embeddings_1.shape)
    embeddings_2 = np.asarray(embeddings_2)
    print('embeddings_2 extracted', embeddings_2.shape)
    labels = df["label"].values.astype(float)
    dataset = df["dataset"].values
    train_mask = dataset == "train"
    test_mask = dataset == "test"

    e1_train, e2_train, s_train = embeddings_1[train_mask], embeddings_2[train_mask], labels[train_mask]
    e1_test, e2_test, s_test = embeddings_1[test_mask], embeddings_2[test_mask], labels[test_mask]
    print(e1_train.shape)
    print(s_train.shape)
    print(e1_test.shape)
    print('embeddings extracted')
    return
    # return e1_train, e2_train, s_train, e1_test, e2_test, s_test


def train_prep_v2(datasets, df_size):
    all_embeddings_1 = []
    all_embeddings_2 = []
    all_labels = []
    all_dataset = []
    for name in datasets:
        df = pd.read_csv(f"Matrix_embeddings_wc/{name}/pairs_{df_size}k_r5.csv")

        # import pickle files
        with open(f'Matrix_embeddings_wc/{name}/embeddings_1_{df_size}k_r5.pkl', 'rb') as f:
            embeddings_1 = pickle.load(f)
        with open(f'Matrix_embeddings_wc/{name}/embeddings_2_{df_size}k_r5.pkl', 'rb') as f:
            embeddings_2 = pickle.load(f)
        print(f'Embeddings loaded for {name}')

        all_embeddings_1.extend(embeddings_1)
        all_embeddings_2.extend(embeddings_2)
        all_labels.extend(df["label"].values.astype(float))
        all_dataset.extend(df["dataset"].values)

        for dataset in ["train", "test"]:
            data = df[df["dataset"] == dataset]
            a, se = accuracy_and_se(data["cosine_similarity"], data["label"])
            print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")
    # convert to np
    all_embeddings_1 = np.asarray(all_embeddings_1)
    all_embeddings_2 = np.asarray(all_embeddings_2)
    all_labels = np.asarray(all_labels)
    print(all_embeddings_1.shape)
    print(all_embeddings_2.shape)
    print(all_labels.shape)
    all_dataset = np.asarray(all_dataset)
    train_mask = all_dataset == "train"
    test_mask = all_dataset == "test"
    e1_train, e2_train, s_train = all_embeddings_1[train_mask], all_embeddings_2[train_mask], all_labels[train_mask]
    e1_test, e2_test, s_test = all_embeddings_1[test_mask], all_embeddings_2[test_mask], all_labels[test_mask]
    print(e1_train.shape)
    print(s_train.shape)
    print(e1_test.shape)
    return e1_train, e2_train, s_train, e1_test, e2_test, s_test

def train(e1_train, e2_train, s_train, e1_test, e2_test,
          s_test, name, hyp_str, output_len, batch_size, max_epochs, learning_rate, dropout, folder_name):

    model_str = optimize_nn(
        e1_train, e2_train, s_train, e1_test, e2_test, s_test, name, hyp_str,

        output_len,
        batch_size,
        max_epochs,
        learning_rate,
        dropout,
        folder_name,
        print_progress=True,
        save_results=True,
    )
    return model_str

# calculate accuracy (and its standard error) of predicting label=1 if similarity>x
def accuracy_and_se(cosine_similarity: float, labeled_similarity: int) -> Tuple[float]:
    accuracies = []
    # find optimal threshold
    threshold = 0.9
    total = 0
    correct = 0
    for cs, ls in zip(cosine_similarity, labeled_similarity):
        total += 1
        if cs >= threshold:
            prediction = 1
        else:
            prediction = 0
        if prediction == ls:
            correct += 1
    accuracy = correct / total
    accuracies.append(accuracy)
    a = max(accuracies)
    n = len(cosine_similarity)
    standard_error = (a * (1 - a) / n) ** 0.5  # standard error of binomial
    return a, standard_error


def accuracy_and_se_pt(cosine_similarity: torch.Tensor, labeled_similarity: torch.Tensor) -> Tuple[float, float]:
    # Move tensors to the MPS device
    cosine_similarity = cosine_similarity.to(mps_device)
    labeled_similarity = labeled_similarity.to(mps_device)

    # Calculate predictions based on the threshold
    threshold = 0.9
    predictions = (cosine_similarity >= threshold).float()

    # Calculate the number of correct predictions
    correct_predictions = (predictions == labeled_similarity.float()).sum().item()

    # Calculate accuracy
    total = cosine_similarity.size(0)
    accuracy = correct_predictions / total

    # Calculate standard error of binomial
    standard_error = (accuracy * (1 - accuracy) / total) ** 0.5

    return accuracy, standard_error

def eval_model(model, e1, e2):
    model.eval()
    with torch.no_grad():
        e1_tensor = torch.tensor(e1).float().to(mps_device)
        e2_tensor = torch.tensor(e2).float().to(mps_device)
        custom_emb_1 = model(e1_tensor)
        custom_emb_2 = model(e2_tensor)
        custom_emb_1 = F.normalize(custom_emb_1, p=2, dim=1)
        custom_emb_2 = F.normalize(custom_emb_2, p=2, dim=1)
        sim = F.cosine_similarity(custom_emb_1, custom_emb_2).cpu().numpy()
        return sim
def eval_model_pt(model, e1, e2):
    model.eval()
    with torch.no_grad():
        custom_emb_1 = model(e1)
        custom_emb_2 = model(e2)
        custom_emb_1 = F.normalize(custom_emb_1, p=2, dim=1)
        custom_emb_2 = F.normalize(custom_emb_2, p=2, dim=1)
        sim = F.cosine_similarity(custom_emb_1, custom_emb_2)
        return sim

def optimize_nn(
        e1_train: np.ndarray,
        e2_train: np.ndarray,
        s_train: np.ndarray,
        e1_test: np.ndarray,
        e2_test: np.ndarray,
        s_test: np.ndarray,
        name: str,
        hyp_str: str,
        output_len: int = 250,  # in my brief experimentation, bigger was better (2048 is length of babbage encoding)
        batch_size: int = 100,
        max_epochs: int = 100,
        learning_rate: float = 100.0,
        # seemed to work best when similar to batch size - feel free to try a range of values
        dropout_fraction: float = 0.0,
        # in my testing, dropout helped by a couple percentage points (definitely not necessary)
        folder_name: str = 'prep',
        print_progress: bool = True,
        save_results: bool = True,
) -> torch.tensor:
    """Return matrix optimized to minimize loss on training data."""
    # Create dataset and loader
    e1_train_t = torch.tensor(e1_train).float().to(mps_device)
    e2_train_t = torch.tensor(e2_train).float().to(mps_device)
    s_train_t = torch.tensor(s_train).float().to(mps_device)

    e1_test_t = torch.tensor(e1_test).float().to(mps_device)
    e2_test_t = torch.tensor(e2_test).float().to(mps_device)
    s_test_t = torch.tensor(s_test).float().to(mps_device)

    train_dataset = torch.utils.data.TensorDataset(
        e1_train_t,
        e2_train_t,
        s_train_t
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    print('dataset and loader created')
    input_dim = 1536
    # define model (similarity of projected embeddings)
    model = EmbeddingsModel(input_dim, output_len, dropout_fraction)
    model.to(mps_device)
    print('model created')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # define loss function to minimize
    def mse_loss(predictions, targets):
        difference = predictions - targets
        return torch.sum(difference * difference) / difference.numel()

    def bce_loss(predictions, targets):
        return torch.nn.functional.binary_cross_entropy(predictions, targets)


    for epoch in range(1, 1 + max_epochs):
        model.train()
        # iterate through training dataloader
        for a, b, actual_similarity in train_loader:
            # generate prediction
            a = a.to(mps_device)
            b = b.to(mps_device)
            actual_similarity = actual_similarity.to(mps_device)

            # zero the gradients
            optimizer.zero_grad()

            output1 = model(a)
            output2 = model(b)

            # normalize the embeddings
            output1 = F.normalize(output1, p=2, dim=1)
            output2 = F.normalize(output2, p=2, dim=1)

            # get sim
            predicted_similarity = F.cosine_similarity(output1, output2)

            # get loss and perform backpropagation
            loss = mse_loss(predicted_similarity, actual_similarity)
            loss.backward()
            optimizer.step()

        # compute custom embeddings and new cosine similarities

        sim_train = eval_model_pt(model, e1_train_t, e2_train_t)
        sim_test = eval_model_pt(model, e1_test_t, e2_test_t)
        # a_train, se_train = accuracy_and_se(sim_train, s_train)
        # a_test, se_test = accuracy_and_se(sim_test, s_test)
        a_train, se_train = accuracy_and_se_pt(sim_train, s_train_t)
        a_test, se_test = accuracy_and_se_pt(sim_test, s_test_t)
        if print_progress:
            print(f"Epoch {epoch}/{max_epochs}: train acuracy: {a_train:0.1%} ± {1.96 * se_test:0.1%}")
            print(f"Epoch {epoch}/{max_epochs}: test accuracy: {a_test:0.1%} ± {1.96 * se_test:0.1%}")

    # data = pd.DataFrame(
    #     {"epoch": epochs, "type": types, "loss": losses, "accuracy": accuracies}
    # )
    # data["run_id"] = run_id
    # data["modified_embedding_length"] = modified_embedding_length
    # data["batch_size"] = batch_size
    # data["max_epochs"] = max_epochs
    # data["learning_rate"] = learning_rate
    # data["dropout_fraction"] = dropout_fraction
    # data["matrix"] = matrices  # saving every single matrix can get big; feel free to delete/change
    if save_results is True:
        # check if the directory exists
        if not os.path.exists(f'nn_models_hyp_scan_{folder_name}/{name}'):
            os.makedirs(f'nn_models_hyp_scan_{folder_name}/{name}')
        model_str = hyp_str
        torch.save(model.state_dict(), f"nn_models_hyp_scan_{folder_name}/{name}/{model_str}.pth")
        print(f'{model_str} saved')
    return model_str

