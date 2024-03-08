# Multimodal Self-Supervised Learning and Fine-Tuning Repository

This repository is designed for running multimodal self-supervised learning (SSL) pretraining, text-only regression fine-tuning, and various prediction and analysis scripts related to the model's performance and outputs. Below are the instructions to effectively utilize this repository.

## Prerequisites

Before running any scripts, ensure you have all the necessary dependencies installed. This may involve setting up a virtual environment and installing the packages listed in a `requirements.txt` file, if provided.

## Data

The dataset required for training and prediction includes equiformer embeddings and text strings from catberta. This data can be accessed through the following Google Drive link: [dummy google drive link](#).

Please download and place the data in the appropriate directory as per the configurations defined in the YAML files.

## Multimodal SSL Pretraining

To run the multimodal SSL pretraining, execute the following command:

```bash
python clip_run.py
```

Adjustments to the data path, training configurations, and other settings can be made in the `clip_train.yml` file located in the root directory.

Additionally, settings specific to the SSL multimodal approach are defined in `model/clip.yml`.

## Text-Only Regression Fine-Tuning

For text-only regression fine-tuning, the following command should be used:

```bash
python regress_run.py
```

Configuration for this process is managed through the `regress_train.yml` file, where you can define settings specific to the fine-tuning task.

## Making Predictions with Text-Only Data

To make predictions using text-only data, utilize the `regress_predict.py` script as follows:

```bash
python regress_predict.py --data_path <PATH_TO_DATA> --checkpoint_path <PATH_TO_CHECKPOINT> --save_path <PATH_TO_SAVE_PREDICTIONS>
```



## Obtaining Section-wise Attention

To extract section-wise attention from the model, use the `get_section_attention.py` script:

```bash
python get_section_attention.py --data_path <PATH_TO_DATA> --checkpoint_path <PATH_TO_CHECKPOINT> --save_path <PATH_TO_SAVE_OUTPUT>
```

## Extracting Text Encoder Embeddings for t-SNE Plot

To obtain text encoder embeddings suitable for visualization with t-SNE plots, execute:

```bash
python get_text_embedding.py --data_path <PATH_TO_DATA> --checkpoint_path <PATH_TO_CHECKPOINT> --save_path <PATH_TO_SAVE_EMBEDDINGS>

