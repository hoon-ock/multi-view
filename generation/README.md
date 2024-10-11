# Generating CIFs Using the CrystaLLM Framework

To generate CIF files using the fine-tuned CrystaLLM framework, follow these steps:

## 1. **Setup**

### 1-1. **Clone the Forked Repository**  
   Begin by cloning the [forked/modified CrystaLLM repository](https://github.com/hoon-ock/CrystaLLM/tree/adslab).

   This repository is based on commit `666c7f0` of the [original CrystaLLM repository](https://github.com/lantunes/CrystaLLM/tree/main).

   Key modifications from the original repo include:
   - Added `replace_data_formula_with_catberta_string` in `crystallm/_utils.py`.
   - Added `process_adslab.py` and `process_dense_adslab.py` to the `bin` directory.
   - Introduced the `<\s>` token in `crystallm/_tokenizer.py`.

### 1-2. **Prerequisites**
   Follow the setup instructions provided in the forked repository to get the CrystaLLM framework up and running. After setup, download the pre-trained model checkpoints from the [original repository](https://github.com/lantunes/CrystaLLM) and place them in the appropriate directory (`<PATH_TO_CHECKPOINT>`).

   You can download our training data and checkpoints from this link: [link](https://doi.org/10.6084/m9.figshare.27208356.v1). Find the `gen_data` and `gen_checkpoint` directory.

---

## 2. **Data Generation & Pre-processing** 

### 2-1. **CIF Generation**
   Run `get_cifs_from_lmdb.py` to convert LMDB file into CIFs.
   
   ```bash
   python get_cifs_from_lmdb.py --lmdb_path <PATH_TO_LMDB> \
                                --save_dir_path <DIR_TO_SAVE_CIFs> \
                                --num_workers 4
   ```

   Then, zip them to create `pkl.gz` format.

### 2-2. **Custom Preprocessing**
   Instead of using the original CrystaLLM format, we adapted a CatBERTa-compatible input string as the initial prompt for the CIFs.

   To preprocess, skip the `bin/preprocess.py` script mentioned in the original CrystaLLM instructions. Instead, run either `bin/preprocess_adslab.py` for the OC20 dataset or `bin/preprocess_dense_adslab.py` for the OC20Dense dataset. The primary purpose of this preprocessing step is to convert the initial prompt in the CIFs to a CatBERTa-styled string (e.g., `data_NO</s>Y8Pb24 (2 1 1)`).

   ```bash
   python bin/preprocess_adslab.py <PATH_TO_CIF_PKL_GZ> \
                                   --out <OUTPUT_PKL_GZ_FILE> \
                                   --decimal-places 3 \
                                   --workers 4 \
                                   --meta-path <PATH_TO_OC20_METADATA>
   ```

   ```bash
   python bin/preprocess_dense_adslab.py <PATH_TO_CIF_PKL_GZ> \
                                   --out <OUTPUT_PKL_GZ_FILE> \
                                   --decimal-places 3 \
                                   --workers 4 \
                                   --meta-path <PATH_TO_OC20DENSE_METADATA>
   ```

   Refer to [`data/README.md`](../data/README.md) for more details.
   - **OC20 metadata file**: `oc20_data_mapping.pkl`
   - **OC20-Dense metadata file**: `oc20dense_mapping.pkl`

### 2-3. **Initial Input Prompt Set Generation**
   To generate the input prompts for adsorbate-catalyst pairs (e.g., `data_NO</s>Y8Pb24 (2 1 1)`), run `input_prompts.py` on the preprocessed `pkl.gz` file from the previous step. This will produce the `input_prompts.tar.gz` file containing the formatted input prompts.

   A number filter is applied to exclude CIFs that don't meet the tokenizer length requirements, extracting only CIFs that satisfy the following criteria:
   - Number of adsorbate elements ≤ 3
   - Number of adsorbate atoms ≤ 5
   - Number of catalyst elements ≤ 3
   - Number of catalyst atoms ≤ 72

   Use the following command to apply the number filter:
   
   ```bash
   python input_prompts.py --data_path <PATH_TO_DATA>
   ```

---

## 3. **Fine-tuning**
   Follow the rest steps instructed in the original CrystaLLM repository.
   

---

## 3. **Generation & Analysis**
### 3-1. **Generate CIF Files**  
   Using the inference steps in the original repository, generate CIF files by running the following command. In this study, we generate 3 CIF files per inference prompt:

   ```bash
   python bin/generate_cifs.py --model <PATH_TO_CHECKPOINT> \
                               --prompts <PATH_TO_PROMPTS.TAR.GZ> \
                               --out <OUTPUT_PATH_TAR.GZ> \
                               --device cuda \
                               --num-gens 3
   ```

   Note. the fine-tuned checkpoint should be loaded

### 3-2. **Filter CIFs Based on Composition**  
   To ensure a minimum level of validity in the generated CIFs, the following criteria are applied:
   - The adsorbate must exactly match the given input prompt.
   - The catalyst (bulk or surface) is allowed up to 12 error atoms.

   Valid CIFs will be saved in the `valid_cifs` directory.

   Use the following command to apply the composition filter:

   ```bash
   python composition_filter.py --data_path <PATH_TO_DATA> --num_gens 3
   ```

### 3-3. **Extract Energies for Evaluation**  
   Run `extract_energies.py` using the file that contains label energies and the valid CIFs generated from the previous step. This step is essential for extracting energies for the Prediction Inclusion Ratio analysis. Ensure that the `data_path` contains the label DFT energies corresponding to the input prompts of the valid CIFs.

   This process will generate the following outputs:  
   - `DFT_energies.csv`: Label DFT energies and configuration strings for each adsorbate-catalyst pair.
   - `adsorbate_catalyst_GT.pkl`: A collection of unique adsorbate-catalyst pairs.
   - `adsorbate_catalyst_config_GT.pkl`: All configurations for each adsorbate-catalyst pair.
   - `CIFS_for_conversion`: CIFs prepared for CIF2String conversion.
   - `DFT_energies_reordered.csv`: The same content as `DFT_energies.csv`, but reordered for indexing.

### 3-4. **Convert CIFs to Strings**  
   Run `CIF2string.py` to convert the CIFs stored in the `CIFS_for_conversion` directory, generated in the previous step, into strings for inference.

   ```bash
   python CIF2string.py --cif_dir <PATH_TO_CIFs> --output_file_name <FILE_NAME>
   ```
   This will generate the CatBERTa input strings in two formats: CSV and PKL.

### 3-5. **Run CatBERTa Predictions & Evaluate Predictions**  
   To obtain CatBERTa predictions using LLM-derived configuration strings, run predictions on the output pickle file generated from the previous step. These predictions correspond to the red "x" marks in Figure 5 of the paper.

   To generate CatBERTa predictions based solely on adsorbate-catalyst pair information (without the LLM-derived configuration), run predictions on the `adsorbate_catalyst_GT.pkl` from step 6. These results correspond to the black "x" marks in Figure 5.

   To obtain CatBERTa predictions using ground truth configuration strings, make predictions with the `adsorbate_catalyst_config_GT.pkl` from step 6. You can evaluate the intrinsic uncertainty of the CatBERTa predictions by calculating the standard deviation for each adsorbate-catalyst pair. These results are represented by the green lines in Figure 5.

   The entire evaluation process can be executed by running `evaluation.py`:

   ```bash
   python evaluation.py --pred1 <PATH_TO_STRINGS_WO_CONFIGS> \
                        --pred2 <PATH_TO_STRINGS_W_LLM_DERIVED_CONFIGS> \
                        --pred3 <PATH_TO_STRINGS_W_GT_CONFIGS>
   ```
   - `--pred1`: Path to predictions without configuration strings.
   - `--pred2`: Path to predictions with LLM-derived configuration strings.
   - `--pred3`: Path to predictions with ground truth configuration strings.  

---

With these steps, you will successfully generate and evaluate CIF files using the CrystaLLM framework.

# Inquiries

For any questions or further information, please reach out to [srivathb@andrew.cmu.edu](mailto:srivathb@andrew.cmu.edu) or [jock@andrew.cmu.edu](mailto:jock@andrew.cmu.edu).
