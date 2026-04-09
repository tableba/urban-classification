import os
import re
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
                if f.lower().endswith(".tif"):
                    yield os.path.join(root, f)

    def _read_preview(self, path, window=None):
        with rasterio.open(path) as src:
            return src.read(window=window)

    def loop_through_s2(self):
        for path in self.get_tif_files(self.s2_dir):
            name = os.path.basename(path)
            data = self._read_preview(path)
            yield name, "S2", data

    def loop_through_dw(self):
        for path in self.get_tif_files(self.dw_dir):
            name = os.path.basename(path)
            data = self._read_preview(path)
            yield name, "DW", data
