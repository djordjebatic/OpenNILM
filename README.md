# OpenNILM: A Deep Learning Framework for Energy Disaggregation

OpenNILM is a research-focused framework for Non-Intrusive Load Monitoring (NILM), also known as energy disaggregation. It leverages deep learning models to disaggregate a household's total electricity consumption (the aggregate signal) into the power consumption of individual appliances.

This framework is built with flexibility and reproducibility in mind, utilizing Hydra for configuration management and supporting various datasets and deep learning architectures.

---
## ✨ Features

* **Multiple Datasets**: Pre-processing scripts for popular NILM datasets like **REFIT** and **PLEGMA**.
* **Diverse Models**: Implementations of state-of-the-art deep learning models for NILM, including **CNN**, **GRU**, and **TCN**.
* **Flexible Configuration**: Easily configure datasets, appliances, models, and training parameters using Hydra.
* **Reproducible Experiments**: Manages experiment outputs, including model checkpoints, logs, and performance metrics.
* **Built with TensorFlow**: Leverages the TensorFlow 2.x ecosystem for building and training neural networks.

---
## 📂 Directory Structure

The project is organized to separate configuration, data processing, and source code, promoting clarity and maintainability.

```
OpenNILM/
├── cfg/                    # Hydra configuration files
│   ├── appliance/          # Appliance-specific configs (power thresholds, etc.)
│   ├── dataset/            # Dataset-specific configs (paths, splits, stats)
│   ├── model/              # Model-specific configs (hyperparameters, architecture)
│   ├── training/           # Training loop configurations (optimizer, epochs)
│   └── config.yaml         # Main configuration file that ties everything together
│
├── data/                   # Raw and processed datasets
│   ├── dataset_management/ # Python scripts for parsing raw datasets
│   └── data.py             # Main script to run the data processing pipeline
│
├── src/                    # Main source code
│   ├── data_loader/        # TensorFlow data loader
│   ├── models/             # Model definitions (CNN, GRU, TCN)
│   ├── nilm_metric.py      # NILM-specific evaluation metrics
│   ├── trainer.py          # Training logic
│   ├── tester.py           # Evaluation logic
│   └── utils.py            # Utility functions
│
├── main.py                 # Main script to run training and evaluation
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```

---
## 🚀 Getting Started

Follow these steps to set up the environment and run your first experiment.

### 1. Installation

It's recommended to use `conda` to manage the environment, especially for handling CUDA dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/djordjebatic-strath/OpenNILM
    cd OpenNILM
    ```

2.  **Create and activate the conda environment:**
    ```bash
    # Create a new conda environment named 'open-nilm'
    conda create --name open-nilm python=3.10

    # Activate the environment
    conda activate open-nilm
    ```

3.  **Install CUDA and cuDNN:**
    This project is tested with CUDA 11.2 and cuDNN 8.1.0. Install them via conda's `conda-forge` channel.
    ```bash
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    ```

4.  **Set up the library path:**
    This step ensures that TensorFlow can find the CUDA libraries.
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
    ```

5.  **Install Python dependencies:**
    Install all required Python packages using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

6.  **Install the source package:**
    Install the source package using `pip`.
    ```bash
    pip install -e .
    ```

7.  **Verify the installation:**
    Check if TensorFlow can detect your GPU.
    ```bash
    python -c "import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))"
    ```
    You should see your GPU listed in the output.

    **Note:** In case of protobuf error, downgrade to protobuf==3.20.0
 
    ```bash
    pip install protobuf==3.20.0
    ```

### 2. Data Preparation

Before training, you need to download the raw datasets and place them in the `data/` directory. Then, run the parsing scripts to convert them into a processed format.

* **Download Datasets**:
    * **REFIT**: Download from [the official source](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned) and place **the content** of `CLEAN_REFIT_081116` into `data/CLEAN_REFIT_081116/Data`. 
    * **PLEGMA**: Download dataset (cleaned) from the [official source](https://pureportal.strath.ac.uk/en/datasets/plegma-dataset) and place `PlegmaDataset` into `data/`.

* **Run Parsing Scripts**:
    Use `data/data.py` to process the raw data. You can specify which datasets and appliances to process.

    * **To process REFIT data for the washing machine and dishwasher:**
        ```bash
        python data/data.py --multirun dataset=refit appliance=washing_machine,dishwasher
        ```

    * **To process PLEGMA data for the boiler and washing machine:**
        ```bash
        python data/data.py --multirun dataset=plegma appliance=boiler,washing_machine
        ```

    Processed files will be saved in `data/processed/<dataset_name>/<appliance_name>/`.

* **Understanding the Output**
    
    The processing script performs several key actions:
    * Loads the raw data for the specified appliance and corresponding houses.
    * Cleans the data by handling missing values and removing problematic readings.
    * Resamples the data to a consistent time frequency.
    * Calculates normalization statistics (mean and standard deviation) from the training set only. **To adjust to your own data, set the mean and std in cfg.dataset document.**
    * Normalizes the aggregate power signal (mains) and scales the appliance power signal.
    * Saves the final data into three separate files: `training_.csv`, `validation_.csv`, and `test_.csv`.

The processed files will be saved in the `data/processed/` directory, organized by dataset and appliance name. For example, processing the REFIT washing machine will create files in: `OpenNILM/data/processed/refit/washing_machine/*`.

### 3. Training and Evaluation

Once the data is processed, you can train a model using `main.py`. Hydra allows you to easily switch between models, datasets, and appliances from the command line. For more information on Hydra, visit [the official docs](https://hydra.cc/docs/intro/).

1.  **Run Training**:
    The following command trains a Temporal Convolutional Network (TCN) on the processed REFIT dataset for the washing machine.

    ```bash
    python main.py dataset=refit appliance=washingmachine model=tcn
    ```

    * To train a GRU model instead, simply change `model=tcn` to `model=gru`.
    * To use a different dataset or appliance, change the `dataset` and `appliance` arguments.

2.  **View Results**:
    Hydra automatically creates a new output directory for each run (e.g., `output/YYYY-MM-DD/HH-MM-SS/`, or `multirun/YYYY-MM-DD/HH-MM-SS/run_id`). Inside this directory, you will find:
    * `checkpoint/model.h5`: The best trained model weights.
    * `metrics/`: A CSV file with the final performance metrics (MAE, F1-Score, etc.).
    * `tensorboard/`: Logs for visualizing training progress in TensorBoard.
    * `.hydra/`: A copy of the configuration used for that run.

---
## ⚙️ How It Works: Configuration with Hydra

This project uses **Hydra** to manage all configurations. Instead of hardcoding parameters (like learning rates or file paths) in the Python code, we define them in simple `.yaml` files located in the `cfg/` directory.

### What is Hydra?

Hydra is a framework that simplifies setting up and running complex applications. Its key benefits are:
* **Composition**: It lets you build a final configuration from smaller, reusable files. For example, a main config file can pull in separate files for the model, dataset, and training parameters.
* **Command-Line Overrides**: You can easily change any setting from the command line without editing the configuration files. This is extremely useful for running experiments (e.g., `python main.py model=gru training.epochs=50`).
* **Automatic Working Directory**: It automatically creates a unique output directory for each run, which is great for organizing experiments and preventing results from being overwritten.
* **Dynamic Instantiation**: It can create Python objects (like your models) directly from the configuration files, which keeps your code clean and flexible.

### Example: Dynamic Model Instantiation

A key feature used in this codebase is **instantiation**. Let's see how the line `model = instantiate(cfg.model.init)` in `main.py` works.

**1. The Model Configuration (`cfg/model/tcn.yaml`)**

This file is a "recipe" for creating a `TCN_NILM` object.

```yaml
name: tcn
batch_size: 50

init:
  # This is the most important key! It points to the Python class.
  _target_: src.models.tcn.TCN_NILM

  # These are the arguments for the class's __init__ method.
  input_window_length: 600
  depth: 9
  nb_filters: [512, 256, 256, 128, 128, 256, 256, 256, 512]
  res_l2: 0
  stacks: 1
  dropout: 0.2
```

* **_target_**: This special key tells Hydra the full import path to the Python class we want to create.

* **Other Keys**: All other keys inside the init block (input_window_length, depth, etc.) are the keyword arguments that will be passed to that class's __init__ method.

**2. The Instantiation Call in Python**

When you run `python main.py model=tcn`, Hydra loads the configuration from `cfg/model/tcn.yaml` into the `cfg.model` object.

The line `model = instantiate(cfg.model.init)` then performs the following steps:

1. It looks at the `cfg.model.init` section of the configuration.

2. It reads the `_target_` path (`src.models.tcn.TCN_NILM`) and imports that class.

3. It calls the class's constructor, passing all the other keys from the `init` block as arguments.

In short, `instantiate(cfg.model.init)` is the dynamic equivalent of writing this in plain Python:

```python
# 1. Manually import the class
from src.models.tcn import TCN_NILM

# 2. Manually create the object with hardcoded parameters
model = TCN_NILM(
    input_window_length=600,
    depth=9,
    nb_filters=[512, 256, 256, 128, 128, 256, 256, 256, 512],
    res_l2=0,
    stacks=1,
    dropout=0.2
)
```

By using Hydra's instantiate function, we avoid hardcoding model parameters in our script. This allows us to easily switch between different models (`cnn`, `gru`, `tcn`) and their unique configurations just by changing a single command-line argument.

# Guide: Changing the Data Sampling Rate

Follow these three steps to adjust the sampling rate for a specific dataset.

### Step 1: Locate and Edit the Configuration File

Navigate to the `cfg/dataset/` directory and open the configuration file for the dataset you want to modify.

For example, to change the rate for the **REFIT** dataset, you would edit `cfg/dataset/refit.yaml`.

### Step 2: Add or Modify the `sampling` Parameter

Inside the YAML file, add or modify the `sampling` key. The value for this key must be a string that corresponds to a pandas resampling frequency.

* `'10s'` for 10 seconds
* `'30s'` for 30 seconds
* `'1min'` for 1 minute

**Example:** To change the sampling rate for the REFIT dataset from the default to **10 seconds**, modify `cfg/dataset/refit.yaml` as follows:

```yaml
name: refit
location: 'CLEAN_REFIT_081116'
sampling: '10s'  # <-- Add or modify this line to set the new rate

split:
  washing_machine:
    train: [2, 5, 7, 9, 15, 16, 17]
    val: [18]
    test: [8]
  # ... rest of the file
```

### Step 3: Reprocess the Data
After saving the configuration changes, you must re-run the data processing script. This is a critical step, as it generates new processed files based on your new sampling rate.

Run the following command from the project's root directory, specifying the dataset and appliance you modified:

Bash

```bash
python data/data.py dataset=refit appliance=washing_machine
```
Your data in `data/processed/refit/washing_machine/` will now be updated with the new 10-second sampling rate.

## How It Works in the Code ⚙️
The data parsers, such as `data/dataset_management/refit/refit_parser.py`, are designed to read this configuration value directly. The core logic is found in the `_load_house_data` method within the parser.

The script fetches the `sampling` value from the Hydra configuration object. If the value is not set in the YAML file, it uses a default (e.g., `'8s'`).

```Python
# The 'sampling' value is fetched from the Hydra config object
# It defaults to '8s' if the key is not found in the YAML file
sampling_rate = self.cfg.dataset.get('sampling', '8s')

# This value is then passed directly to the pandas resample function
df = df.resample(sampling_rate).mean().fillna(method='ffill', limit=30)
```

This implementation provides a flexible and easy way to control data granularity for your experiments directly through configuration files.

## Inference Model Files 🧠

You can access trained model files in `models` folder. To load them, first instantiate the appropriate model using the `model = instantiate(cfg.model.init)`. Make sure that the `cfg.model` matches the .h5 model type (e.g. `CNN_AC.h5` model and `cfg.model: cnn`). Next, change the model_path to the desired model in `models` folder. You might want to set the `model.built = True` before calling `model.load_weights(model_path)` in case of `HDF5` errors.
