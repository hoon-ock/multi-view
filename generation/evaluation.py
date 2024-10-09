import pandas as pd
import ast
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import statistics

# Function to read the CSV file and parse the energies
def read_csv_file(file_path):
    df = pd.read_csv(file_path)

    # Print a few entries to check the format
    print(df['Energies'].head())

    # Try to parse the 'Energies' column
    def parse_energies(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If parsing fails, return the value as is
            return value

    df['Energies'] = df['Energies'].apply(parse_energies)

    # Check if we have lists and flatten them if needed
    if isinstance(df['Energies'].iloc[0], list):
        df['Energies'] = df['Energies'].apply(lambda x: [float(i) for i in x])

    return df

# Function to calculate MAE with respect to the minimum of energy ground truths
def calculate_mae_min(energy_values, predicted_value):
    min_val = min(energy_values)
    mae = abs(predicted_value - min_val)
    return mae

# Function to calculate MAE with respect to the average of energy ground truths
def calculate_mae_average(energy_values, predicted_value):
    avg_val = sum(energy_values) / len(energy_values)
    mae = abs(predicted_value - avg_val)
    return mae

def plot_energies_with_threshold(df, preds1, preds2, preds3):
    fig, ax = plt.subplots(figsize=(10, 6))

    count = 0

    unique_systems = df['Adsorbate-Catalyst'].unique()
    counter=0


    accuracy1 = 0
    accuracy_with_threshold1 = 0

    accuracy2 = 0
    accuracy_with_threshold2 = 0

    overall_min_ads_cat = 0
    overall_min_ads_cat_string = 0

    overall_avg_ads_cat = 0
    overall_avg_ads_cat_string = 0
    
    for i, system in enumerate(unique_systems):
        system_data = df[df['Adsorbate-Catalyst'] == system]
        energy_values = system_data['Energies'].tolist()
        
        if isinstance(energy_values[0], list):
            energy_values = [val for sublist in energy_values for val in sublist]

        predicted_value1 = preds1[i]
        predicted_value2 = preds2[i]

        min_ads_cat = calculate_mae_min(energy_values, predicted_value1)
        overall_min_ads_cat+=min_ads_cat
        min_ads_cat_string = calculate_mae_min(energy_values, predicted_value2)
        overall_min_ads_cat_string+=min_ads_cat_string

        avg_ads_cat = calculate_mae_average(energy_values, predicted_value1)
        overall_avg_ads_cat+=avg_ads_cat
        avg_ads_cat_string = calculate_mae_average(energy_values, predicted_value2)
        overall_avg_ads_cat_string+=avg_ads_cat_string

        ax.scatter([i] * len(energy_values), energy_values, color='blue', s=5, label='DFT energy values' if i == 0 else "")

        preds3sub = []
        for j in range(counter, counter+len(energy_values)):
            preds3sub.append(preds3[j])

        counter +=len(energy_values)

        if isinstance(preds3sub[0], list):
            preds3sub = [val for sublist in preds3sub for val in sublist]

        # threshold = statistics.stdev(preds3sub)
        threshold = np.std(preds3sub) if len(preds3sub) > 1 else 0

        min_val = min(energy_values)
        max_val = max(energy_values)

        ax.hlines(min_val - threshold, i - 0.1, i + 0.1, color='green', label='CatBERTa error threshold' if i == 0 else "")
        ax.hlines(max_val + threshold, i - 0.1, i + 0.1, color='green')

        if i < len(preds1):
            ax.scatter([i], [preds1[i]], color='black', marker="x", s=20, label='CatBERTa prediction (w/o configuration)' if i == 0 else "")

        if i < len(preds2):
            ax.scatter([i], [preds2[i]], color='red', marker="x", s=20, label='CatBERTa prediction (w/ LLM-derived configuration)' if i == 0 else "")

        if min_val <= preds1[i] <= max_val:
            accuracy1 += 1
            accuracy_with_threshold1 += 1
        elif min_val - threshold <= preds1[i] <= max_val + threshold:
            accuracy_with_threshold1 += 1

        if min_val <= preds2[i] <= max_val:
            accuracy2 += 1
            accuracy_with_threshold2 += 1
        elif min_val - threshold <= preds2[i] <= max_val + threshold:
            accuracy_with_threshold2 += 1

        count += 1

    i+=1

    print("Accuracy without configurations:", accuracy1/i*100)
    print("Accuracy with threshold without configurations:", accuracy_with_threshold1/i*100)

    print("Accuracy with configurations:", accuracy2/i*100)
    print("Accuracy with threshold with configurations:", accuracy_with_threshold2/i*100)


    overall_min_ads_cat/=i
    overall_min_ads_cat_string/=i

    overall_avg_ads_cat/=i
    overall_avg_ads_cat_string/=i

    print("MAE wrt min energy value without configurations:", overall_min_ads_cat)
    print("MAE wrt min energy value with configurations:", overall_min_ads_cat_string)

    print("MAE wrt avg energy value without configurations:", overall_avg_ads_cat)
    print("MAE wrt avg energy value with configurations:", overall_avg_ads_cat_string)

    ax.set_xticks(range(len(unique_systems)))
    ax.set_xticklabels(unique_systems, rotation=45, ha='right')
    ax.set_ylabel('Energy [eV]', fontsize=15)

    ax.legend()

    plt.tight_layout()
    plt.savefig('Energy_plot.jpg', dpi=960)
    plt.show()

    print("1 = without configuration, 2 = with")



# Main script
def evaluate(preds1_path, preds2_path, preds3_path):
    file_path = 'DFT_energies_reordered.csv'  # Update with your file path
    # preds1_path = 'preds_adsorbate_catalyst_GT.pkl'  # Update with your file path
    # preds2_path = 'preds_LLM_strings.pkl'   # Update with your file path
    # preds3_path = 'preds_adsorbate_catalyst_config_GT.pkl'

    # file_path = 'DFT_energies.csv'  # Update with your file path
    # preds1_path = 'preds_adsorbate_catalyst_GT.pkl'  # Update with your file path
    # preds2_path = 'preds_LLM_strings.pkl'   # Update with your file path
    # preds3_path = 'preds_adsorbate_catalyst_config_GT.pkl'

    with open(preds1_path, 'rb') as f:
        preds1_data = pkl.load(f)

    with open(preds2_path, 'rb') as f:
        preds2_data = pkl.load(f)
    
    with open(preds3_path, 'rb') as f:
        preds3_data = pkl.load(f)


    df = read_csv_file(file_path)
    plot_energies_with_threshold(df, preds1_data, preds2_data, preds3_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred1", type=str, required=True, help="Path to the inferences on adsorbate-catalyst pair string without configuration part.")
    parser.add_argument("--pred2", type=str, required=True, help="Path to the inferences on strings with LLM-derived configurations.")
    parser.add_argument("--pred3", type=str, required=True, help="Path to the inference on strings with ground truth configuration parts.")
    args = parser.parse_args()
    evaluate(args.pred1, args.pred2, args.pred3)


