import pandas as pd

def load_labels(csv_path):
    df = pd.read_csv(csv_path, sep=';')  # IMPORTANT: you used semicolons
    return df
