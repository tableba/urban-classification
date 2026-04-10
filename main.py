import numpy as np
from src.models.kcluster import run_kmeans
from src.dataset.read_data import ReadTifs
from src.dataset.preprocessing import clean_s2_data


def main():
    reader = ReadTifs()

    for key, s2_data, dw_data in reader.loop_through_files():
        s2_data = clean_s2_data(s2_data)

        try :
            run_kmeans(key, s2_data)

        except:
            print(f"Failed at tile_{key}.")
        

if __name__ == "__main__":
    main()
