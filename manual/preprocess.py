"""

"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Constants ──────────────────────────────────────────────
RAW_PATH = 'data/raw/'
CLEANED_PATH = 'data/cleaned/'
TEST_SIZE = 0.2
RANDOM_SEED = 42

def prepare_data(raw_path, cleaned_path):
    """
    
    """

    # load raw data
    print("Loading raw data...")
    exp_raw = pd.read_csv(raw_path + 'MicroarrayExpression.csv', index_col=0, header=None)
    probes = pd.read_csv(raw_path + 'Probes.csv', index_col='probe_id')
    annotations = pd.read_csv(raw_path + 'SampleAnnot.csv', index_col='structure_id')

    # recode chromosome labels
    print("Cleaning data...")
    group_map = {
        '19': 1, '17': 1, '16': 1, '22': 1,
        '11': 2, '20': 2,
        '1': 3, '2': 3, '3': 3,
        '6': 4, '7': 4, '8': 4, '9': 4, '12': 4, '15': 4,
        '4': 5, '5': 5, '10': 5, '13': 5, '14': 5, '18': 5, '21': 5, 'X': 5, 'Y': 5
    }
    probes['group'] = probes['chromosome'].astype(str).map(group_map)
    probes['group'] = probes['group'].astype('Int64')

    # annotate raw data
    exp = exp_raw.copy()
    exp.columns = annotations.index
    exp_full = probes.join(exp, how='left').drop(columns=['probe_name', 'gene_id', 'gene_symbol', 'gene_name', 'entrez_id'])

    # filter out missing labels
    exp_na = exp_full[exp_full['group'].isna()]
    exp_use = exp_full[~exp_full['group'].isna()]

    # split into training and testing sets
    exp_train, exp_test = train_test_split(exp_use, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=exp_use['group'])

    # save data
    print("Saving cleaned data...")
    exp_full.to_pickle(cleaned_path + 'exp_full.pkl')
    exp_na.to_pickle(cleaned_path + 'exp_na.pkl')
    exp_train.to_pickle(cleaned_path + 'exp_train.pkl')
    exp_test.to_pickle(cleaned_path + 'exp_test.pkl')

    print("Data loaded and cleaned")

if __name__ == "__main__":
    prepare_data(RAW_PATH, CLEANED_PATH)
