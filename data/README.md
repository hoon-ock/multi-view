# OC20 and OC20-Dense Data Processing

This repository contains scripts and guidelines for handling the OC20 and OC20-Dense datasets, including steps for extracting relaxed positions, generating string representations, and integrating string files.

## Original Datasets

The original datasets for OC20 and OC20-Dense can be accessed from the following links:
- [OC20 Dataset](https://fair-chem.github.io/core/datasets/oc20.html)
- [Original OC20-Dense Dataset](https://fair-chem.github.io/core/datasets/oc20dense.html)
- [Challenge OC20-Dense Dataset](https://opencatalystproject.org/challenge.html)

In the paper, OC20-Dense datasets for OCP 2023 Challenge were used.
- **OC20-Dense Training set**: lmdb(`oc20dense_is2re_train_v2.tar.gz`), traj (`oc20dense_s2ef_train.tar.gz`)
- **OC20-Dense Validation set**: lmdb(`oc20dense_ood_val.tar.gz`), traj(`oc20dense_val_trajectories.tar.gz`)
- **ML-relaxed Test set**: gemnet-oc (`gemnet_oc_2M_oc20dense_val_ood.tar.gz`), scn (`scn_2M_oc20dense_val_ood.tar.gz`), escn (`escn_2M_oc20dense_val_ood.tar.gz`)


### Note:
- **OC20 IS2RE LMDB files**: Contain relaxed positions.
- **OC20-Dense IS2RE LMDB files**: Do not contain relaxed positions. To extract relaxed positions for the OC20-Dense dataset, an additional step is required.

---

## LMDB Creation with Relaxed Frames (OC20-Dense)

To create an LMDB file with the relaxed positions for the OC20-Dense dataset:

```bash
python relaxed_frame_lmdb.py --mapping_file_path <PATH_TO_MAPPING_FILE> \
                             --traj_path <DIRECTORY_OF_TRAJ_FILES> \
                             --save_path <PATH_TO_SAVE_FILE; lmdb extension> \
                             --refE_path <PATH_TO_REF_E_FILE> \
                             --tag_path <PATH_TO_TAG_FILE>
```

### Required Files:

- **Mapping file**: `oc20dense_mapping.pkl`
- **Reference Energy file**: `oc20dense_ref_energies.pkl`
- **Tag file**: `oc20dense_tags.pkl`

## String Generation

After preparing the LMDB files, strings can be generated for further processing.

### For OC20 Dataset:

```bash
python string_generator_oc20.py --lmdb_path <PATH_TO_LMDB> \
                                --save_path <PATH_TO_SAVE_FILE; pkl extension> \
                                --metadata_path <PATH_TO_METADATA> \
                                --task <is2re or rs2re or s2ef>
```

### For OC20-Dense Dataset:
```bash
python string_generator_oc20dense.py --lmdb_path <PATH_TO_LMDB> \
                                     --save_path <PATH_TO_SAVE_FILE; pkl extension> \
                                     --metadata_path <PATH_TO_METADATA> \
                                     --split <train or val or eval or test>
```

### Required Files:

- **OC20 metadata file**: `oc20_data_mapping.pkl`
- **OC20-Dense metadata file**: `oc20dense_mapping.pkl`

## String File Integration

Once string files are generated, they can be integrated using the following command:

```bash
python string_integrator.py --string_dir <DIRECTORY_WHERE_STRINGS_SAVED> \
                            --save_path <PATH_TO_SAVE_FILE; pkl extension>
```