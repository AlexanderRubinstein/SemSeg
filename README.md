# keras-semantic-segmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction
This repo contains code for semantic segmentation using convolutional neural networks built on top of the [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) libraries.
The major part of code was taken from https://github.com/azavea/raster-vision and https://github.com/mrubash1/keras-semantic-segmentation/tree/develop/src.

The following datasets and model architectures are implemented.

### Datasets
* [ISPRS Potsdam 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html) ✈️
* [ISPRS Vaihingen 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) ✈️

### Model architectures
* [FCN](https://arxiv.org/abs/1411.4038) (Fully Convolutional Networks) using [ResNets](https://arxiv.org/abs/1512.03385)
* [U-Net](https://arxiv.org/abs/1505.04597)
* [Fully Convolutional DenseNets](https://arxiv.org/abs/1611.09326) (aka the 100 Layer Tiramisu)

## Usage

### Scripts

| Name     | Description                              |
| -------- | ---------------------------------------- |
| `cipublish`  | Publish docker image to ECR |
| `clean`  | Remove build outputs inside virtual machine |
| `infra`  | Execute Terraform subcommands            |
| `test`   | Run unit tests and lint on source code |
| `run` | Run container locally or remotely |
| `setup`  | Bring up the virtual machine and install dependent software on it |
| `update` | Install dependent software inside virtual machine |
| `upload_code` | Upload code to EC2 instance for rapid debugging |

## Running locally on CPUs

### Data directory

All data including datasets and results are stored in `/opt/data`. The datasets are stored in `/opt/data/datasets` and results are stored in `/opt/data/results`.

### Preparing datasets

Before running any experiments locally, the data needs to be prepared so that Keras can consume it. For the
[ISPRS 2D Semantic Labeling Potsdam dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html), you can download the data after filling out the [request form](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html).
After following the link to the Potsdam dataset, download
`1_DSM_normalisation.zip`, `4_Ortho_RGBIR.zip`, `5_Labels_for_participants.zip`, and `5_Labels_for_participants_no_Boundary.zip`. Then unzip the files into
`/opt/data/datasets/potsdam`, resulting in `/opt/data/datasets/potsdam/1_DSM_normalisation/`, etc.

Then run `python -m semseg.data.factory --preprocess`. This will generate `/opt/data/datasets/processed_potsdam`. As a test, you may want to run `python -m semseg.data.factory --plot` which will generate PDF files that visualize samples produced by the data generator in  `/opt/data/results/gen_samples/`.
 To make the processed data available for use on EC2, upload a zip file of `/opt/data/datasets/processed_potsdam` named `processed_potsdam.zip` to the `otid-data` bucket.

### Running experiments

An experiment consists of training a model on a dataset using a set of hyperparameters. Each experiment is defined using an options `json` file.
An example can be found in [src/experiments/quick_test.json](src/experiments/quick_test.json), and this
can be used as a quick integration test.
In order to run an experiment, you must also provide a list of tasks to perform. These tasks
include `setup_run`, `train_model`, `plot_curves`, `validation_eval`, `test_eval`. More details about these can be found in [src/semseg/run.py](src/semseg/run.py).

Here are some examples of how to use the `run` command.
```shell
# Run all tasks by default
python -m semseg.run experiments/quick_test.json
# Only run the plot_curves tasks which requires that setup_run and train_model were previously run
python -m semseg.run experiments/quick_test.json plot_curves
```
This will generate a directory structure in `/opt/data/results/<run_name>/` which contains the options file, the learned model, and various metrics and visualization files.