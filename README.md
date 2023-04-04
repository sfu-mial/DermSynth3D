# DermSynth3D
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sfu-mial/DermSynth3D/tree/main.svg?style=svg&circle-token=176de57353747d240e619bdf9aacf9f716e7d04f)](https://dl.circleci.com/status-badge/redirect/gh/sfu-mial/DermSynth3D/tree/main) 
![GPLv3](https://img.shields.io/static/v1.svg?label=📃%20License&message=GPL%20v3.0&color=green)
[![arXiv](https://img.shields.io/static/v1.svg?label=📄%20arXiv&message=N/A&color=red)](#) 
[![DOI](https://img.shields.io/static/v1.svg?label=📄%20DOI&message=N/A&color=orange)](#) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Copyright](https://img.shields.io/static/v1.svg?label=DermSynth3D%20©️%20&message=%202023&labelColor=green&color=blue) 

This is the official code repository following our work [DermSynth3D](#link-to-arxiv).

## TL;DR

A data generation pipeline for creating photorealistic _in-the-wild_  synthetic dermatalogical data with rich annotations such as semantic segmentation masks, depth maps, and bounding boxes for various skin analysis tasks.

![main pipeline](assets/pipeline.png) 

## Motivation

Existing datasets for dermatological image analysis have significant limitations, including a small number of image samples, limited disease conditions, insufficient annotations, and non-standardized image acquisitions.
To address this problem, we propose **DermSynth3D**, or generating synthetic 2D skin image datasets using 3D human body meshes blended with skin disorders from clinical images.
Our approach uses a differentiable renderer to blend the skin lesions within the texture image of the 3D human body and generates 2D views along with corresponding annotations, including semantic segmentation masks for skin conditions, healthy skin, non-skin regions, and anatomical regions. 
Moreover, we include a diverse range of skin tones and background scenes, that enables us to generate semantically rich and meaningful labels for 2D _in-the-wild_ clinical images that can be used for a variety of dermatological tasks, as opposed to just one.

<!-- We present a novel framework called **DermSynth3D** that blends skin disease patterns onto 3D textured meshes of human subjects using a differentiable renderer and generates 2D images from various camera viewpoints under chosen lighting conditions in diverse background scenes. -->

<!-- The 2D dermatological images generated using DermSynth3D are:
- meaningful _i.e._ anatomically relevant
- mimic _in-the-wild_ acquistions
- photo-realistic
- densely annotated with
  - Semantic segmentation labels of healthy skin, skin-condition, and human anatomy
  - Bounding boxes around skin-condition
  - Depth maps
  - 3D scene parameters, such as camera position and lightning conditions -->

## Repository layout

```bash
DermSynth3D_private/
┣ assets/                      # assets for the README
┣ configs/                     # YAML config files to run the pipeline
┣ logs/                        # experiment logs are saved here (auto created)
┣ out/                         # the checkpoints are saved here (auto created)
┣ data/                        # directory to store the data
┃  ┣ ...                       # detailed instructions in the dataset.md
┣ dermsynth3d/                 # 
┃  ┣ datasets/                 # class definitions for the datasets
┃  ┣ deepblend/                # code for deep blending
┃  ┣ losses/                   # loss functions 
┃  ┣ models/                   # model definitions
┃  ┣ tools/                    # wrappers for synthetic data generation
┃  ┗ utils/                    # helper functions
┣ notebooks/                   # demo notebooks for the pipeline
┣ scripts/                     # scripts for traning and evaluation
┗ skin3d/                      # external module
```

## Table of contents
- [Installation](#installation)
  - [using conda](#using-conda)
  - [using Docker](#using-docker) **Recommended**
- [Datasets](#datasets)
- [Usage](#usage)
  - [Generating Synthetic Dataset](#generating-synthetic-dataset)
  - [Preparing Dataset for Experiments](#preparing-dataset-for-experiments)
- [Cite](#cite)
- [Demo Notebooks for Dermatology Tasks](#demo-notebooks-for-dermatology-tasks)
  - [Lesion Segmentation](#lesion-segmentation)
  - [Multi-Task Prediction](#multi-task-prediction)
  - [Lesion Detection](#lesion-detection)


<a name="installation"></a>

### Installation

<a name="conda"></a>

#### using conda

```bash
git clone --recurse-submodules https://github.com/sfu-mial/DermSynth3D.git 
cd DermSynth3D
conda create --name dermsynth3d -f dermsynth3d.yml
conda activate dermsynth3d
```

<a name="docker"></a>

#### using Docker

```bash
# Build the container in the root dir
docker build -t dermsynth3d --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f Dockerfile .
# Run the container in interactive mode for using DermSynth3D
# See 3. Usage
docker run --gpus all -it --rm -v /path/to/downloaded/data:/data dermsynth3d
```
For provide the pre-built docker image, which can be be used as well:
```bash
# pull the docker image
docker pull sinashish/dermsynth3d:latest
# Run the container in interactive GPU mode for generating data and training models
# mount the data directory to the container
docker run --gpus all -it --rm -v /path/to/downloaded/data:/data dermsynth3d
```

#### <span style="color: red">NOTE</span>

The code has been tested on Ubuntu 20.04 with CUDA 11.1, python 3.8, pytorch 1.10.0, and pytorch3d 0.7.2, and we don't know if it will work on CPU.

If you face any issues installing pytorch3d, please refer to their [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) or this issue [link](https://github.com/facebookresearch/pytorch3d/issues/1076).


<a name="data"></a>

## Datasets

Follow the instructions in [dataset.md](./dataset.md) to download the datasets for generating the synthetic data and training models for various tasks.

<a name="usage"></a>

## Usage

<a name='gen'></a>

### Generating Synthetic Dataset
![annotated data](./assets/AnnotationOverview.png)

Before running any code, make sure that you have downloaded the data necessary for blending as mentioned in [dataset.md](./dataset.md) and folder structure is as described above.
If the folder structure is different, then please update the paths accordingly in `configs/blend.yaml`.

Now, to generate the synthetic data with the default parameters, simply run the following command to generate 2000 views for a specified mesh:

```bash
python -u scripts/gen_data.py
```

To change any blending or synthesis parameters only, run using:
```bash
python -u scripts/gen_data.py --lr <learning rate> \ 
            -m <mesh_name> \
            -s <path to save the views> \
            -ps <skin threshold> \
            -i <blending iterations> \
            -v <number of views> \
            -n <number of lesions per mesh> 
```
Feel free to play around with `random` parameter in `configs/blend.yaml` to control lighting, material and view points.

#### Post-Process Renderings with Unity
![synthetic data](./assets/fig_1-min.png)
Follow the detailed instructions outlined [here](./unity.md) to create photorealistic renderings using Unity.

<a name='prep'></a>

### Preparing Dataset for Experiments

After creating the syntheic dataset in the previous step, it is now the time to test the utility of the dataset on some real world tasks.

Before, you start with any experiments, ideally you would want to organize the generated data into `train/val/test` sets. 
We provide a script to do the same:
```bash
python scripts/prep_data.py
```

You can look at `scripts/prep_data.py` for more details.

<a name="cite"></a>

## Cite
If you find this work useful or use any part of the code in this repo, please cite our paper:
```bibtext
@unpublished{kawahara2023ds3d,
  title={DermSynth3D: Synthesis of in-the-wild annotated dermatology images},
  author={Kawahara, Jeremy\textsuperscript{*}\textsuperscript{1} and Sinha, Ashish\textsuperscript{*}\textsuperscript{1} and Pakzad, Arezou\textsuperscript{1} and Abhishek, Kumar and Ruthven, Matthieu and Baratab, Enjie and Kacem, Anis and Aouada, Djamila and Hamarneh, Ghassan},
  year={2023},
  note={Preprint. Currently under review.},
}
```

<a name="repro"></a>

## Demo Notebooks for Dermatology Tasks

![Qualitative Results](./assets/results.png)

<a name='seg'></a>
<a name='train'></a>

### Lesion Segmentation
**Note**: Update the paths to relevant datasets in `configs/train_mix.yaml`.

To train a lesion segmentation model with default parameters, on a combination of Synthetic and Real Data, simply run:

```bash
python -u scripts/train_mix_seg.py
```

Play around with the following parameters for a combinatorial mix of datasets.
```yaml
    real_ratio: 0.5                 # fraction of real images to be used from real dataset
    real_batch_ratio: 0.5           # fraction of real samples in each batch
    pretrain: True                  # use pretrained DeepLabV3 weights
    mode: 1.0                       # Fraction of the number of synthetic images to be used for training
```

You can also look at [this notebook](./notebooks/train_segmentation.ipynb) for a quick overview for training lesion segmention model.

For inference of pre-trained models/checkpoints, look at [this notebook](./notebooks/inference_segmentation.ipynb).

<a name='multi'></a>

### Multi-Task Prediction

We also train a multi-task model for predicting lesion, anatomy and depth, and evaluate it on multiple datasets.

For a quick overview of multi-task prediction task, checkout [this notebook](./notebooks/inference_multitask.ipynb).

For performing inference on your trained models for this task. First update the paths in `configs/multitask.yaml`. Then run:
```bash
python -u scripts/infer_multi_task.py
```

<a name='det'></a>

### Lesion Detection

For a quick overview for training lesion detection models, please have a look at [this notebook](./notebooks/train_detection.ipynb).

For doing a quick inference using the pre-trained detection models/ checkpoints, have a look at [this notebook](./notebooks/inference_detection.ipynb).

## Acknowledgements

We are thankful to the authors of [Skin3D](https://github.com/jeremykawahara/skin3d) for making their code and data public for the task of lesion detection on 3DBodyTex.v1 dataset.
