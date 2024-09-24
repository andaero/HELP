# HELP: Hierarchical Embeddings-based Log Parsing

## Installation

### Conda
To create the conda environment 'HELP', run the following command:
```
conda env create -f environment.yml
```
Then activate the environment using:
```
conda activate HELP
```
### Setting up environment variables
Make a copy of the `.env.template` file and rename it to `.env`. Fill in the values for the environment variables in the `.env` file.


## Datasets
The datasets are publicly available via [Zenodo](https://zenodo.org/records/8275861). 
Please download the datasets and place them in the `full_dataset/` directory following the format of ```Apache/```.


## Benchmark Results
To get the performance of HELP on the benchmark datasets, run the following sets of commands:
``` 
# for precomputing the openai embeddings
python CacheOpenAI.py
# for creating cross validation training data
python CustomEmbeddingsPreprocess.py
# For training and testing the model
python HyperparamScan.py
```
To reproduce the UMap plots, run:
```
# for plotting nn plot
python UMapPlot.py --nn 
# for plotting non nn plot
python UMapPlot.py
```

To recreate the ablation results of the paper on HELP components:
HELP w/o LLM:
```
python RebalanceMergeNN.py
```
HELP w/o prev. & NN:
```
python Rebalance_Merge.py
```
HELP w/o prev. & WC:
```
python CacheOpenAI.py --no_wc
python Rebalance_Merge.py --no_wc
```
HELP w/o prev. & IRM:
```
python NaiveFixed.py
```