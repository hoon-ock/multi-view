## Generating CIF Files Using the CrystaLLM Framework

To generate CIF files using the fine-tuned CrystaLLM framework, follow these steps:

## 1. **Clone the Original Repository**  
   First, clone the original CrystaLLM repository:  
   [CrystaLLM](https://github.com/lantunes/CrystaLLM)

## 2. **Set Up the Framework**  
   Follow the setup instructions in the original repository to get the CrystaLLM framework working. Once set up, download the fine-tuned model's checkpoints from [here](https://github.com/lantunes/CrystaLLM) **(make sure to also host the checkpoint on HuggingFace!)** and save it to an appropriate directory `<PATH_TO_CHECKPOINT>`.

## 3. **Initial Input Prompt Set Generation**
   Run `input_prompts.py` to generate `input_prompts.tar.gz`, which contains input prompts for adsorbate-catalyst pairs (e.g., `data_NO</s>Y8Pb24 (2 1 1)`).

   A number filter is applied to exclude CIFs that don't meet the tokenizer length requirements, extracting only CIFs that satisfy the following criteria:
   - Number of adsorbate elements ≤ 3
   - Number of adsorbate atoms ≤ 5
   - Number of catalyst elements ≤ 3
   - Number of catalyst atoms ≤ 72

   Use the following command to apply the number filter:
   
   ```bash
   python input_prompts.py --data_path <PATH_TO_DATA>
   ```
   

## 4. **Generate CIF Files**  
   Using the inference steps in the original repository, generate CIF files by running the following command. In this study, we generate 3 CIF files per inference prompt:

   ```bash
   python bin/generate_cifs.py --model <PATH_TO_CHECKPOINT> \
                               --prompts <PATH_TO_PROMPTS.TAR.GZ> \
                               --out <OUTPUT_PATH_TAR.GZ> \
                               --device cuda \
                               --num-gens 3
   ```

## 5. **Filter CIFs Based on Composition**  
   To ensure a minimum level of validity in the generated CIFs, the following criteria are applied:
   - The adsorbate must exactly match the given input prompt.
   - The catalyst (bulk or surface) is allowed up to 12 error atoms.

   Valid CIFs will be saved in the `valid_cifs` directory.

   Use the following command to apply the composition filter:

   ```bash
   python composition_filter.py --data_path <PATH_TO_DATA> --num_gens 3
   ```

## 6. **Extract Energies for Evaluation**  
   Run `extract_energies_new.py` using the following inputs:  
   - `oc20dense_train_rev.pkl`
   - `valid_cifs` directory

   This will generate the following outputs:  
   - `DFT_energies_new_CC.csv`  
   - `adsorbate_catalyst_GT_new_CC.pkl`  
   - `adsorbate_catalyst_config_GT_new_CC.pkl`  
   - `CIFS_for_conversion`  
   - `DFT_energies_reordered.csv`

## 7. **Convert CIFs to Strings**  
   Run `CIF2string.py` on the `CIFS_for_conversion` directory to get inference strings. This will generate:  
   - `LLM_strings.csv`  
   - `LLM_strings.pkl`

## 8. **Run CatBERTa Predictions**  
   - Run CatBERTa on `LLM_strings.pkl` to obtain `preds_LLM_strings.pkl` (red).
   - Then, run CatBERTa on both `adsorbate_catalyst_GT_new_CC.pkl` and `adsorbate_catalyst_config_GT_new_CC.pkl` to get:  
     - `preds_adsorbate_catalyst_GT_new_CC.pkl` (black)  
     - `preds_adsorbate_catalyst_config_GT_new_CC.pkl` (green line)

## 9. **Evaluate Predictions**  
    Finally, run `prediction_evaluation.py` on the three prediction files (`preds_LLM_strings.pkl`, `preds_adsorbate_catalyst_GT_new_CC.pkl`, `preds_adsorbate_catalyst_config_GT_new_CC.pkl`) along with `DFT_energies_reordered.csv` to evaluate the predictions.

---

With these steps, you will successfully generate and evaluate CIF files using the CrystaLLM framework.