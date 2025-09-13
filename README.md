# 🧬 MUTscope

MUTscope is a flexible, modular pipeline for training and evaluating machine learning models for predicting the effects of single amino acid substitutions (currently for binary clssification problems). Designed to be  model-agnostic, it enables end-to-end training from raw sequences to interpretable outputs. This framework supports multiple classifier heads (MLP, CNN, etc.), and extensible interpretability modules including in silico deep mutational scanning (DMS) and 3D residue-level insights.

**AMRscope** is a trained instance of MUTscope using anti-microbial resistance data. Below are instructions for predicting using AMRscope or training your own model instance using MUTscope pipeline. This repository accompanies a manuscript [[preprint](https://www.biorxiv.org/content/10.1101/2025.09.12.672331v1)], full AMRscope data sources, OOD splits, baselines, and calibration analysis are detailed there. 

---

## 🚀 Features

* **Flexible Experiment Configuration**: Control every step via changes to single YAML config file
* **Classifier-Agnostic**: Easily plug in MLP, CNN, or classical models
* **Lightning Integration**: Built on PyTorch Lightning with support for callbacks, checkpointing, and logging
* **DeepSpeed Ready**: Efficient large-scale training on HPCs with optional DeepSpeed support
* **Embedding Cache**: Stores embeddings under the unique target id so they need only be generated once per mutation
* **Interpretability**: Dimensionality reduction, structural visualization, and DMS simulation

---
# Quickstart

## 🛠️ Installation

```bash
git clone https://github.com/drjjwood/MUTscope.git
cd MUTscope
```
Either
- Create your environment for use on **CPU**
```bash
conda env create -f environment-cpu.yml
conda activate MUTscope_cpu
```
or
- Create your environment for  use on **GPU**
```bash
conda env create -f environment-gpu.yml
conda activate MUTscope_gpu
```

Install optional extras if needed;
- to use weights and biases
```bash
pip install wandb
```
- to use DeepSpeed (GPU only):
```bash
pip install deepspeed
```

⚠️ If you hit a CUDA mismatch (HPCs)

Check the driver/runtime on your node:
```bash
nvidia-smi
```
If your site uses a different CUDA runtime (e.g., 12.1 not 12.4), install the matching meta-package inside the GPU env:
```bash
conda install -n MUTscope_gpu -c nvidia pytorch-cuda=12.1
```

Then reinstall the PyTorch trio (if needed):
```bash
conda install -n MUTscope_gpu -c pytorch pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1
```

---
##  📦 Usage

Below describes how to run 1)prediction and 2)training pipelines. If you are having issues, try running with the example files in the tests folder to make sure it runs with known data. 

### 1. 🔮 PREDICTION ONLY PIPELINE with AMRscope

- Step 1. Download the AMRscope model from OSF and place in AMRscope folder

- Step 2. Setup your data, this can either be for predictions of a) specific mutations or b) in silico DMS of a given proten as follows;

a) Setup csv file with mutations as described in 'Data format' section below (you can also look at tests/test_example.csv for a template)

b) Add fasta file in the following format
```bash
> organism_name.gene_name
IVLATD
```

- Step 3. Setup your config, you can use configs/config_predict.yaml as a template. All the fields are described in detail in the Configuration Guide below. However the most important thing is to set it up depending on your use case

a) For specific mutations fill in 'predict_file_name' with the name of your mutaiton file (note DO NOT indlude .csv), also set dms: False

b) For DMS fill in 'fasta_file_name' with the name of your fasta file (note DO NOT indlude .fasta), also set dms: True

- Step 4. Run predictions

Activate your conda environment an then run the predict pipeline, for example;

```bash
conda activate MUTscope
python main.py --mode predict --config_path configs/config_predict.yml
```

- Step 5. (Optional) Run DMS interpretation

If you are running full DMS then you can run the interpretation notebook in AMRscope/DMS_template.ipynb, following the instructions as described therein. 


### 2. 🚂 TRAIN YOUR OWN MODEL with MUTscope

Step 1. Setup your data .csv file with mutations as described in 'Data format' section below (you can also look at tests/test_example.csv for a template)

Step 2. Setup your config, you can use configs/config_train.yaml as a template. All the fields are described in detail in the Configuration Guide below. However, as a minimal start point you should add your train/val/test file names. 

Step 3. Run training

```bash
conda activate MUTscope
python main.py --mode train --config_path configs/config_train.yml
```
Step 4. (Optional) Use your model to run futher predictions with instructions in part 1, but make sure you **change the model path** to match your newly trained model.

---

## 🧪 Data Format

Each input CSV should contain (at minimum):

* `target_id`: Unique ID - **MUST BE IN FORMAT organism.gene.mutation (e.g.Mycobacterium_tuberculosis.embB.F330V)**
* `wt`: wt amino acid (e.g. F)
* `mutation`: mutant amino acid (e.g. V)
* `position`: position of mutation (e.g. 330)
* `wt_seq`: Wild-type amino acid sequence
* `mt_seq`: Mutant sequence (single substitution)
* `label`: 0/1 binary classification label (not required for prediciton only) 

---

## 📂 Project Structure Overview

```
MUTscope/
├── AMRscope/                          # Optional: pretrained model + AMR-specific configs/notebooks
├── configs/                           # YAML config files for training/inference/sweeps
├── datasets/                          # Training/validation/test/prediction datasets
├── tests/                             
│
├── base_embeddings.py                # General embedding loader logic
├── esm_embeddings.py                 # ESM-specific embedding generator
├── base_model.py                     # PyTorch Lightning model wrapper
├── classifiers.py                    # Modular classifier heads (MLP, CNN, etc.)
├── data_io.py                        # File reading/writing and I/O utilities
├── dataloaders.py                    # Dataset class and DataLoader construction
├── cached_dataloaders.py             # Generates dataloaders with caching
├── embedding_cache.py                # Caching helpers
│
├── base_evaluation.py                # Shared evaluation logic
├── evaluation.py                     # Evaluation for deep learning models
├── evaluation_classical.py          # Evaluation for classical models (e.g., sklearn)
│
├── training.py                       # Deep learning training loop
├── training_classical.py            # Classical model training loop
│
├── dms_predictor.py                  # In silico DMS generation pipeline
├── base_interpretation.py           # General interpretation logic
├── plots_interpretation.py          # Plotting functions for interpretability
├── dimensionality_interpretation.py # Dimensionality reduction (e.g. PCA/t-SNE)
├── run_interpretation.py            # Master script for generating interpretation outputs
│
├── main.py                           # Unified entry point (train/predict)
└── README.md                         # Project documentation

```

---

## 📈 Evaluation & Logging

Metrics computed:

* Accuracy, F1, Precision, Recall, MCC
* Confusion matrices (saved as PNG)
* Custom WandB plots (ROC, PR curve)

Numerical metrics are logged and also saved as CSV

---

## 🧠 Optional Interpretation Tools

* **Dimensionality Reduction**: t-SNE / PCA on latent space
* **DMS Simulation**: Predict effects of all possible substitutions at each position in a specified gene
* **3D Structure Mapping**: Map predictions onto protein structures (via Py3Dmol in notebooks)

---

## ⚙️ Configuration Guide

All pipeline behavior is defined via a YAML configuration file. Below are the key sections and options available.

---

### 🧪 `settings`

| Key                      | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `seed`                   | Random seed for reproducibility                          |
| `track_wb`               | Whether to track metrics/logs in Weights & Biases        |
| `wandb_key`              | API key for WandB login (required if `track_wb` is True) |
| `project_name`           | WandB project name (for grouping runs)                   |
| `experiment_name`        | Experiment folder name (used for saving outputs/logs)    |
| `run_sweep`              | If True, uses wandb sweep settings                       |
| `embedding_type`         | Choose between `esm2`, `esm1b`                           |
| `randomised_control_run` | If True, randomizes labels (for null/control runs)       |

---

### 📊 `dataset`

| Key                                                  | Description                                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------------- |
| `disable_cache`                                      | If False, loads precomputed embeddings from disk instead of regenerating   |
| `input_path`                                         | Path to dataset CSV files or saved embeddings                              |
| `train_file_name`, `val_file_name`, `test_file_name` | Filenames (without `.csv`) for each dataset split                          |
| `predict_file_name`                                  | Used only in inference/prediction mode                                     |
| `save_path`                                          | Output directory for saving logs, models, predictions, etc.                |
| `loader_batch_size`                                  | Batch size for PyTorch DataLoaders                                         |
| `num_workers`                                        | Number of subprocesses for DataLoader (depends on CPU cores)               |

---

### 📉 `pca` *(optional)*

| Key             | Description                                            |
| --------------- | -------------------------------------------------------|
| `dim_reduction` | Whether to apply PCA to embeddings **BEFORE training** |
| `n_components`  | Number of PCA components if enabled                    |

---

### 🧠 `model`

| Key                          | Description                                                         |
| ---------------------------- | ------------------------------------------------------------------- |
| `classifier_head`            | Type of classifier: `MLP1lyr`, `MLP2lyr`, `CNN`, `GBC`, `LR`, `SVC` |
| `save_path`                  | Directory to save trained models                                    |
| `embedding_size`             | *(Inference only)* Dimensionality of input embeddings               |
| `threshold`                  | Classification threshold for binary prediction (e.g., 0.3)          |

---

### ⚡ `trainer` *(Training only)*

| Key                    | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `max_epochs`           | Max number of training epochs                      |
| `log_every_n_steps`    | Logging frequency in steps                         |
| `fast_dev_run`         | If True, does a single run for debugging           |
| `profiler`             | Enable profiling; e.g., `'simple'`, `'advanced'`   |
| `accelerator`          | Hardware accelerator (`'gpu'`, `'cpu'`)            |
| `strategy`             | Distributed strategy; e.g., `'deepspeed'`, `'ddp'` |
| `devices`              | Number of GPUs per node                            |
| `num_nodes`            | Number of nodes across which to run                |
| `num_sanity_val_steps` | Number of sanity validation steps before training  |
| `precision`            | Floating-point precision (e.g., `'16-mixed'`)      |

---

### 🔮 `predictions`

| Key               | Description                                                                           |
| ----------------- | --------------------------------------------------------------------------------------|
| `interpretation`  | If True, runs interpretation pipeline after predictions                               |
| `dms`             | *(Inference only)* If True, performs in silico deep mutational scanning               |
| `fasta_file_name` | *(Inference only)* Used with `dms=True` — name of reference sequence for substitution |

---

### 🔁 `sweep` *(if `run_sweep=True`)*

| Key          | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| `method`     | Sweep search strategy: `random`, `grid`, etc.                    |
| `metric`     | Dict specifying optimization target (`val_loss`, `val_f1`, etc.) |
| `run_cap`    | Max number of sweep runs                                         |
| `parameters` | Dictionary of hyperparameter ranges for the sweep                |

**Example sweep parameters:**

* `num_hidden1`, `num_hidden2`: Hidden layer sizes for MLP
* `lr`: Learning rate
* `dropout`: Dropout rate
* `weight_decay`: L2 regularization strength
* `optimizer`: e.g., `"adam"`, `"adamw"`
* `scheduler`: e.g., `"step"`, `"cosine_annealing"`
* `lr_decay`: Decay factor for learning rate scheduler
