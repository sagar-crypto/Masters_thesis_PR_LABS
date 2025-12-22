import numpy as np
import pandas as pd
from pathlib import Path

def load_replica(replica_path):
    replica_path = Path(replica_path)

    if replica_path.suffix == ".parquet":
        df = pd.read_parquet(replica_path)
        return df

    elif replica_path.suffix == ".npy":
        data = np.load(replica_path, allow_pickle=True)
        returnArr = {}
        for i, val in enumerate(data[0]):
            returnArr.update({ i : val})
        return data, returnArr

    else:
        raise ValueError(f"Unsupported file format: {replica_path}")
