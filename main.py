import numpy as np
from src.models.kcluster import run
from dataset.read_data import ReadTifs
from dataset.preprocessing import clean_data, filter_clouds


def main():
    reader = ReadTifs()

    for name, t, data in reader.loop_through_files():
        if t == "SAT_2016" or t == "SAT_2023":
            data = clean_data(data)
            filter_clouds(f"./data/{name}")

            # run(name, data)

if __name__ == "__main__":
    main()
