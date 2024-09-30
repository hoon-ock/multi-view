# Multimodal Graph and Language Learning for Adsorption Configuration in Catalysis
[arXiv](https://arxiv.org/abs/2401.07408)

**Important Note:** This repository is currently in the process of being cleaned up for better organization and usability. The cleanup will be completed before the publication of the associated paper. We appreciate your patience and understanding.


![TOC_resized](https://github.com/hoon-ock/multi-view/assets/93333323/0ad53e44-18df-43a0-a413-1bc5438777e6)

This repository is designed for running multimodal self-supervised learning (SSL) pretraining, text-only regression fine-tuning, and various prediction and analysis scripts related to the model's performance and outputs. Below are the instructions to effectively utilize this repository.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or above (Recommended: Python 3.8)
- pip (Python Package Installer)

This project requires the following packages:

- `torch==1.11.0`
- `transformers==4.38.2`

## Data & Checkpoint

The dataset required for training and prediction includes equiformer embeddings and text strings from catberta. This data can be accessed through the following link: [Data](https://cmu.box.com/s/6d2zbi00yoizyg60ppztdgqiaes1msqw).

Please download and place the data in the appropriate directory and update the data paths in the YAML files.

The example checkpoints can be found in the following link: [Checkpoint](https://cmu.box.com/s/2i4kyyfrlrtilbm8n39xtd8piramnbz5).

### Original Structure Data

The original structure data of adsorbate-catalyst systems can be found at the Open Catalyst Project's dataset documentation: [https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md).


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

## Making Predictions with Text-Only Data

To make predictions using text-only data, utilize the `regress_predict.py` script as follows:

```bash
python regress_predict.py --data_path <PATH_TO_DATA> --pt_ckpt_dir_path <PATH_TO_CHECKPOINT> --save_path <PATH_TO_SAVE_PREDICTIONS>
```


## Obtaining Section-wise Attention

To extract section-wise attention from the model, use the `get_section_attention.py` script:

```bash
python get_section_attention.py --data_path <PATH_TO_DATA> --pt_ckpt_dir_path <PATH_TO_CHECKPOINT> --save_path <PATH_TO_SAVE_OUTPUT>
```

## Extracting Text Encoder Embeddings for t-SNE Plot

To obtain text encoder embeddings suitable for visualization with t-SNE plots, execute:

```bash
python get_text_embedding.py --data_path <PATH_TO_DATA> --pt_ckpt_dir_path <PATH_TO_CHECKPOINT> --save_path <PATH_TO_SAVE_EMBEDDINGS>
```

## Inquiries

For any questions or further information, please reach out to [jock@andrew.cmu.edu](mailto:jock@andrew.cmu.edu).

## Citation

If you use this work in your research, please cite two papers as follows:

```bibtex
@misc{ock2024multimodal,
      title={Multimodal Language and Graph Learning of Adsorption Configuration in Catalysis}, 
      author={Janghoon Ock and Rishikesh Magar and Akshay Antony and Amir Barati Farimani},
      year={2024},
      eprint={2401.07408},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
      }
```

```bibtex
@article{ock2023catberta,
         author = {Ock, Janghoon and Guntuboina, Chakradhar and Barati Farimani, Amir},
         title = {Catalyst Energy Prediction with CatBERTa: Unveiling Feature Exploration Strategies through Large Language Models},
         journal = {ACS Catalysis},
         volume = {13},
         number = {24},
         pages = {16032-16044},
         year = {2023},
         doi = {10.1021/acscatal.3c04956},
         URL = {https://doi.org/10.1021/acscatal.3c04956},
         eprint = {https://doi.org/10.1021/acscatal.3c04956}
         }
```
