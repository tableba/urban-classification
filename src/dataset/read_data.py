import os
import re
import numpy as np
import rasterio


class ReadTifs:
    def __init__(self,
                 s2_dir="./data/s2",
                 dw_dir="./data/dw"):
        self.s2_dir = s2_dir
        self.dw_dir = dw_dir

    def get_tif_files(self, folder):
        for root, _, files in os.walk(folder):
            for f in files:
                yield os.path.join(root, f)

    def _read_tif(self, path, window=None):
        with rasterio.open(path) as src:
            return src.read(window=window)

    def loop_through_s2(self):
        for path in self.get_tif_files(self.s2_dir):
            name = os.path.basename(path)
            data = self._read_tif(path)
            yield name, "S2", data

    def loop_through_dw(self):
        for path in self.get_tif_files(self.dw_dir):
            name = os.path.basename(path)
            data = self._read_tif(path)
            yield name, "DW", data

    def loop_through_files(self):
        """
        loops through Sentinel-2 and Dynamic World tiles with common tile number
        """
        s2_files = {
            os.path.splitext(f)[0].split("_")[1]: os.path.join(self.s2_dir, f)
            for f in os.listdir(self.s2_dir)
            if f.lower().endswith(".tif")
        }

        dw_files = {
            os.path.splitext(f)[0].split("_")[1]: os.path.join(self.dw_dir, f)
            for f in os.listdir(self.dw_dir)
            if f.lower().endswith(".tif")
        }

        # Find common tiles
        common_tiles = sorted(set(s2_files.keys()) & set(dw_files.keys()))

        for key in common_tiles:
            s2_path = s2_files[key]
            dw_path = dw_files[key]

            s2_data = self._read_tif(s2_path)
            dw_data = self._read_tif(dw_path)

            yield key, s2_data, dw_data

# testing
if __name__ == "__main__":
    reader = ReadTifs()
    for key, s2, dw in reader.loop_through_files():
        print("key: ", key)
        print("s2 shape:", s2.shape)        # (bands, height, width)
        print("s2: ", s2)
        print("s2 dtype:", s2.dtype)
        print("s2 min/max:", np.nanmin(s2), np.nanmax(s2))
        print("dw shape:", dw.shape)
        print("dw:", dw)
        print("---")

    # reader.loop_through_dw()


