import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["gnoc", "escn", "scn"], default="gnoc")
parser.add_argument("--enhancement", type=str, choices=["vanilla", "ca", "oc20-ssl", "combined-ssl"], default="oc20-ssl")
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--perplexity", type=int, default=30)   
args = parser.parse_args()
model = args.model
enhancement = args.enhancement
checkpoint = args.checkpoint
perplexity = args.perplexity


def determine_category(id_value):
    if str(id_value).startswith('random'):
        return 'oc20-train'
    try:
        int(id_value)
        return 'dense-train'
    except ValueError:
        return None  

# ========================= PATHS ====================================
# label 
test_label_path = f'/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/oc20dense_eval_{model}_relaxed.pkl'
train_label_path = '/home/jovyan/shared-scratch/jhoon/ocp2023/clip_data/oc20_oc20dense_train_relaxed.pkl'

# embedding
path = "/home/jovyan/shared-scratch/jhoon/ocp2023/results/encoder_embedding/"
test_emb_path = path+f"emb-ssl-oc20-eqv2_1204_2048-eval-{model}-strc.pkl"
train_emb_mapping = {
                    "vanilla": path+"emb-rebase3-vanilla-catberta_1223_1537-train-combined-strc.pkl",
                    "ca": path+"emb-rebase3-da-catberta_1224_1518-train-combined-strc.pkl",
                    "oc20-ssl": path+"emb-ssl-oc20-eqv2_1204_2048-train-combined-strc.pkl",
                    "combined-ssl": path+"emb-ssl-combined-eqv2_1207_1531-train-combined-strc.pkl",
                     }
train_emb_path = train_emb_mapping[enhancement]

# ========================= BUILD DATAFRAME ===============================
dict_train_emb = pd.read_pickle(train_emb_path)
dict_test_emb = pd.read_pickle(test_emb_path)

df_train_emb = pd.DataFrame(list(dict_train_emb.items()), columns=['id', 'emb'])
df_test_emb = pd.DataFrame(list(dict_test_emb.items()), columns=['id', 'emb'])

df_train_label = pd.read_pickle(train_label_path)
df_test_label = pd.read_pickle(test_label_path)

df_train_label = df_train_label.drop(columns=['text', 'chg_emb', 'gnoc_emb', 'eq_emb'])
df_test_label = df_test_label.drop(columns=['text', 'chg_emb'])

# combine df_data and df_emb along with 'id' column
df_train = pd.merge(df_train_label, df_train_emb, on='id') 
df_test = pd.merge(df_test_label, df_test_emb, on='id')

# set category column
df_train['category'] = df_train['id'].apply(determine_category)
df_test['category'] = 'test'

# sample subset from df_train['category']=='oc20-train'
df_train_oc20 = df_train[df_train['category']=='oc20-train'].sample(n=60000, random_state=1)
df_train_dense = df_train[df_train['category']=='dense-train']

df_train = pd.concat([df_train_oc20, df_train_dense], ignore_index=True)

# concatenate df_train and df_test
df = pd.concat([df_train, df_test])

# category counts
category_counts = df['category'].value_counts()
print(category_counts)

# ========================= tSNE ====================================
# Extract embeddings and targets
embeddings = np.array(df['emb'].tolist())
targets = df['category'].values

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Create a new DataFrame for t-SNE results
tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne1', 'tsne2'])
tsne_df['category'] = targets

# Plot the t-SNE results
fig, ax = plt.subplots(figsize=(10, 8))

# Define categories and their corresponding colors from the 'tab10' colormap
categories = ['oc20-train', 'dense-train', 'test']
legend_mapping = {'oc20-train':'Train (OC20)', 'dense-train':'Train (OC20-Dense)', 'test':'Test'}
#colors = plt.cm.plasma(np.linspace(0, 0.9, len(categories)))

custom_colors = {
    'oc20-train': ('grey', 0.2),  # Grey color with alpha=0.2 for 'oc20-train'
    'dense-train': ('brown', 0.7), # Brown color for 'dense-train'
    'test': ('blue', 0.7)       # Orange color for 'test'
}

# custom_colors = {
#     'oc20-train': ('grey', 0.2),  # Grey color with alpha=0.2 for 'oc20-train'
#     'dense-train': ('tan', 0.4), # Brown color for 'dense-train'
#     'test': ('lightcoral', 0.4)       # Orange color for 'test'
# }

# Scatter plot for each category
# for category, color in zip(categories, colors):
for category in categories:
    subset = tsne_df[tsne_df['category'] == category]
    legend_text = legend_mapping.get(category, category)
    # Get custom color and alpha for each category
    color, alpha = custom_colors.get(category, (None, 0.7))

    ax.scatter(subset['tsne1'], subset['tsne2'], c=[color], label=legend_text, s=50, alpha=alpha)


# Adding the legend
ax.legend(fontsize=16, loc='upper right')

# Remove box frame, ticks, and labels
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('')

plt.show()
fig.savefig(f"dataset/train-{enhancement}-test-{model}-p{perplexity}.png", dpi=300, bbox_inches='tight')