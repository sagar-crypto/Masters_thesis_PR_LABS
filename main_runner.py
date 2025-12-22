from preprocessing.load_labels import load_labels
from preprocessing.load_replicas import load_replica
from config import CACHE_FILE_PATH, MAIN_DATA_PATH


def main():
    labels = load_labels(f'{MAIN_DATA_PATH}/labels.csv')
    print(labels.head())

    replica, val_arr = load_replica(f'{CACHE_FILE_PATH}/replica_27.npy')
    print(val_arr)


if __name__ == "__main__":
    main()



