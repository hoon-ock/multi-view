[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13922448.svg)](https://doi.org/10.5281/zenodo.13922448)

# Multimodal Graph and Language Learning for Adsorption Configuration in Catalysis
[arXiv Preprint](https://arxiv.org/abs/2401.07408)


---


![TOC_resized](https://github.com/hoon-ock/multi-view/assets/93333323/0ad53e44-18df-43a0-a413-1bc5438777e6)

This repository provides the tools for multimodal self-supervised learning (SSL) pretraining, text-only regression fine-tuning, as well as prediction and analysis scripts related to model performance and outputs.

Below are the instructions to effectively use this repository.

---

## 1. Prerequisites

**Note:** 🚧🚧 This section is currently under construction and will be updated soon 🚧🚧.

Before you begin, ensure you have met the following requirements:

- Python 3.6 or above (Recommended: Python 3.8)
- pip (Python Package Installer)

This project requires the following packages:

- `torch==2.5.1`
- `transformers==4.47.1`
- `tokenizers==0.21.0`
- `pandas==2.2.3`
- `pydantic==2.10.3`
- `tqdm`, `wandb`

---

## 2. Data & Checkpoint

### 2-1. Preprocessing

For detailed preprocessing steps, please refer to [`data/README.md`](data/README.md).

### 2-2. Data Files & Checkpoint
The dataset required for training and prediction includes equiformer embeddings and text strings from catberta. This data and checkpoints can be accessed through the following link: [Data](https://doi.org/10.6084/m9.figshare.27208356.v2).

Please download and place the data in the appropriate directory and update the data/checkpoints paths in the YAML files.

---

## 3. Training & Prediction

### 3-1. Graph-assisted Pre-trainin (Multi-modal SSL Pre-training)

To run the graph-assisted pre-training, execute the following command:

```bash
python clip_run.py
```

Adjustments to the data path, training configurations, and other settings can be made in the `clip_train.yml` file located in the root directory.

Additionally, settings specific to the SSL multimodal approach are defined in `model/clip.yml`.

### 3-2. Text-Only Fine-Tuning

For text-only regression fine-tuning, the following command should be used:

```bash
python regress_run.py
```

Specific settings should be defined in `regress_train.yml`.

## 3-3. Text-Only Prediction

To make predictions using text-only data, utilize the `regress_predict.py` script as follows:

```bash
python regress_predict.py --data_path <PATH_TO_DATA> \
                          --pt_ckpt_dir_path <PATH_TO_CHECKPOINT> \
                          --save_path <PATH_TO_SAVE_PREDICTIONS>
```

---

## 4. Analysis 

### 4-1. Test Prediction Comparison with Valid DFT Energies

In the paper, test predictions are made on ML-relaxed structures. To assess their accuracy, the predicted values are compared with the valid DFT energies of the ML-relaxed systems.

You can generate the comparison using the following command:

```bash
python analysis/parity_plot.py --pred_path <PATH_TO_PRED_RESULTS> \
                               --save_dir <SAVE_DIRECTORY> \
                               --mapping_file_path <PATH_TO_MAPPING_FILE> \
                               --dft_target_path <PATH_TO_DFT_ENERGIES> \
                               --model <MODEL_TYPE; gnoc or scn or escn>
```

- **OC20-Dense metadata file**: `oc20dense_mapping.pkl` [link](https://fair-chem.github.io/core/datasets/oc20dense.html)
- **OC20-Dense OCP Challenge DFT energiees**: `ml_relaxed_dft_targets.pkl` [link](https://opencatalystproject.org/challenge.html)

### 4-2. Section-wise Attention

To extract section-wise attention from the model, use the `get_section_attention.py` script:

```bash
python analysis/get_section_attention.py --data_path <PATH_TO_DATA> \
                                         --pt_ckpt_dir_path <PATH_TO_CHECKPOINT> \ 
                                         --save_path <PATH_TO_SAVE_OUTPUT>
```

### 4-3. Extracting Text Encoder Embeddings for t-SNE Plot

To obtain text encoder embeddings suitable for visualization with t-SNE plots, execute:

```bash
python analysis/get_text_embedding.py --data_path <PATH_TO_DATA> \
                                      --pt_ckpt_dir_path <PATH_TO_CHECKPOINT> \
                                      --save_path <PATH_TO_SAVE_EMBEDDINGS>
```

---

## 5. Generation & Prediction with Fine-tuned CrystaLLM

For detailed preprocessing steps, please refer to [`generation/README.md`](generation/README.md).


---

## Inquiries

For any questions or further information, please reach out to [jock@andrew.cmu.edu](mailto:jock@andrew.cmu.edu) or [srivathb@andrew.cmu.edu](mailto:srivathb@andrew.cmu.edu).

---

## Citation

If you use this work in your research, please cite two papers as follows:

```bibtex
@misc{ock2024multimodal,
      title={Multimodal Language and Graph Learning of Adsorption Configuration in Catalysis}, 
      author={Janghoon Ock and Srivathsan Badrinarayanan and Rishikesh Magar and Akshay Antony and Amir Barati Farimani},
      year={2024},
      eprint={2401.07408},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2401.07408}, 
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
