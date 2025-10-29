# GBM 3D-Segmentation and Reconstruction

## Introduction

This project is a work in progress.

We aim to leverage deep learning and high-resolution microscopy to invent a novel technique for nano-scale 3D segmentation and reconstruction of Glomerular Basement Membrane (GBM), a ribbon-like extracellular matrix that lies between the endothelium and the podocyte foot processes.

<br/>

<p align="center">
  <img src="res/prediction.jpg" alt="prediction" width="80%" style="border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);">
</p>

<br/>

Additionally, we focus on developing GPU-based algorithms to extract useful morphometric features from the 3D reconstruction to acquire a better understanding of GBM's role as a filtration barrier and its alteration in pathological scenarios.

<br/>

<p align="center">
  <img src="res/gbm_render.jpg" alt="GBM Render" width="80%" style="border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);">
</p>

<br/>

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd gbm-seg
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Blender:**
    For 3D visualization, you need to have Blender installed and available in your system's PATH.

## Hardware Requirements

The project was developed and tested on servers equipped with:

-   **GPUs:** 4x NVIDIA A100 or 4x NVIDIA V100 GPUs
-   **CPU:** 64-core AMD EPYC processors
-   **RAM:** 500+ GB

## Usage

This application features an experiment management system accessible via the `gbm.py` command-line interface. Users can create, train, and manage experiments, perform inference, and visualize results. The workflow is designed to be flexible, supporting both local execution and SLURM-based job submission.

### Configuration

The application uses a YAML configuration file located at `./configs/template.yaml`. This file contains various settings such as the root directory for experiments, model architecture, training parameters, and inference settings. Before running any experiments, you should review and customize this file to match your environment and requirements.

### Local Execution (Without SLURM)

The `gbm.py` script provides a command-line interface for managing experiments directly on your local machine or a non-SLURM environment.

#### Global Options

- `-d`, `--debug`: Enable debugging mode

#### List Experiments

List created experiments or snapshots of a specific experiment.

```bash
python gbm.py list [-r] [-s SNAPSHOTS]
```

Options:
- `-r`, `--root`: Specify the root directory of experiments
- `-s SNAPSHOTS`, `--snapshots SNAPSHOTS`: List the snapshots of a specific experiment

#### Create a New Experiment

Create a new experiment with the given name.

```bash
python gbm.py create <name> [-bs BATCH_SIZE]
```

Options:
- `-bs BATCH_SIZE`, `--batch-size BATCH_SIZE`: Set the batch size for training (default: 8)

#### Delete an Experiment

Delete the selected experiment.

```bash
python gbm.py delete <name>
```

#### Train an Experiment

Start or continue training for the specified experiment.

```bash
python gbm.py train <name>
```

#### Run Inference

Create an inference session for the specified experiment.

```bash
python gbm.py infer <name> -s SNAPSHOT [-bs BATCH_SIZE] [-sd SAMPLE_DIMENSION] [-st STRIDE] [-sf SCALE_FACTOR]
```

Options:
- `-s SNAPSHOT`, `--snapshot SNAPSHOT`: Select the snapshot for inference (required)
- `-bs BATCH_SIZE`, `--batch-size BATCH_SIZE`: Set the batch size for inference (default: 8)
- `-sd SAMPLE_DIMENSION`, `--sample-dimension SAMPLE_DIMENSION`: Set sample dimension for inference (default: '12, 256, 256')
- `-st STRIDE`, `--stride STRIDE`: Set the stride for inference (default: '1, 64, 64')
- `-sf SCALE_FACTOR`, `--scale-factor SCALE_FACTOR`: Set the scale for interpolation (default: 1)

#### Post-processing

Perform post-processing to remove noise and artifacts from inference results.

```bash
python gbm.py psp <name> -it INFERENCE_TAG -mc MAX_CONCURRENT
```

Options:
- `-it INFERENCE_TAG`, `--inference-tag INFERENCE_TAG`: Tag of the inference session to process.
- `-mc MAX_CONCURRENT`, `--max-concurrent MAX_CONCURRENT`: Number of concurrent processes for post-processing.

#### Morphometric Analysis

Perform morphometric analysis on a processed sample.

```bash
python gbm.py morph <name> -it INFERENCE_TAG -sn SAMPLE_NAME
```

Options:
- `-it INFERENCE_TAG`, `--inference-tag INFERENCE_TAG`: Tag of the inference session.
- `-sn SAMPLE_NAME`, `--sample-name SAMPLE_NAME`: Name of the sample to analyze.

#### Prepare Blender Visualizations

Prepare data for Blender visualizations.

```bash
python gbm.py blender <name> -it INFERENCE_TAG -sn SAMPLE_NAME
```

Options:
- `-it INFERENCE_TAG`, `--inference-tag INFERENCE_TAG`: Tag of the inference session.
- `-sn SAMPLE_NAME`, `--sample-name SAMPLE_NAME`: Name of the sample for visualization.

#### Render Blender Visualizations

Render Blender visualizations.

```bash
python gbm.py render <name> -it INFERENCE_TAG
```

Options:
- `-it INFERENCE_TAG`, `--inference-tag INFERENCE_TAG`: Tag of the inference session.

#### Export Results

Export inference and analysis results.

```bash
python gbm.py export <name> -it INFERENCE_TAG
```

Options:
- `-it INFERENCE_TAG`, `--inference-tag INFERENCE_TAG`: Tag of the inference session.

#### Generate Statistics

Generate statistics from the analysis results.

```bash
python gbm.py stats <name> -it INFERENCE_TAG
```

Options:
- `-it INFERENCE_TAG`, `--inference-tag INFERENCE_TAG`: Tag of the inference session.

#### Examples for Local Execution

1.  **Create a new experiment:**
    ```bash
    python gbm.py create my_experiment --batch-size 16
    ```

2.  **List all experiments:**
    ```bash
    python gbm.py list
    ```

3.  **Train an experiment:**
    ```bash
    python gbm.py train my_experiment
    ```

4.  **Run inference:**
    ```bash
    python gbm.py infer my_experiment --snapshot best_model --batch-size 4 --sample-dimension "24, 512, 512" --stride "2, 128, 128" --scale-factor 2
    ```

### SLURM Execution

For environments utilizing the SLURM workload manager, the project provides scripts to submit jobs for various tasks, especially for the full inference pipeline. Individual SLURM job scripts are located in the `sbatch/` directory.

#### Inference Pipeline using `gbm_inference.sh`

The `gbm_inference.sh` script provides a convenient way to run a complete inference pipeline, which includes:
1.  Inference
2.  Post-processing (noise removal)
3.  Morphometric analysis
4.  Blender visualization

This script will submit a series of dependent SLURM jobs to perform the entire inference workflow.

**Usage:**
```bash
./gbm_inference.sh --name=<project_name> --snapshot=<snapshot_file> --batch-size=<batch_size> --sample-dimension=<dims> --stride=<stride> --scale-factor=<factor>
```

**Example:**
```bash
./gbm_inference.sh --name=my_experiment --snapshot=002-2920.pt --batch-size=4 --sample-dimension='12, 256, 256' --stride='1, 64, 64' --scale-factor=6
```

#### Individual SLURM Job Submission

You can also submit individual jobs using the scripts in the `sbatch/` directory. For example, to submit an inference job:

```bash
sbatch ./sbatch/infer.sbatch <name> <snapshot> <batch_size> "<sample_dimension>" "<stride>" <scale_factor>
```
Refer to the specific `.sbatch` files for their required arguments and usage.


## Project Structure
```
├───.gitignore
├───.pylintrc
├───blender.py
├───gbm_inference.sh
├───gbm.py
├───infer.py
├───README.md
├───setup.cfg
├───train.py
├───__pycache__/
├───configs/
│   └───template.yaml
├───res/
│   ├───blender_template.blend
│   ├───gbm_render.jpg
│   └───prediction.jpg
├───sbatch/
│   ├───blender.sbatch
│   ├───export.sbatch
│   ├───infer.sbatch
│   ├───morph.sbatch
│   ├───psp.sbatch
│   └───render.sbatch
└───src/
    ├───data/
    ├───infer/
    ├───models/
    ├───scripts/
    ├───train/
    └───utils/
```

- **`gbm.py`**: Main CLI entry point.
- **`train.py`, `infer.py`**: Core training and inference scripts.
- **`configs/template.yaml`**: Main configuration file.
- **`src/`**: Source code, including data loaders, models, and utilities.
- **`sbatch/`**: SLURM scripts for job submission on a cluster.
- **`res/`**: Resources like Blender templates and images.

## Debugging

Add the `--debug` flag to any command to enable debugging mode.