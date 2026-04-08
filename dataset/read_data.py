import os
import re
import rasterio

class ReadTifs:
    def __init__(self,
                 images_dir = "./data",
                 rules = [
                    (re.compile(r"NEW[_]?BUILDINGS", re.I), "BUILD_NEW"),
                    (re.compile(r"2016.*BUILDINGS", re.I), "BUILD_2016"),
                    (re.compile(r"2023.*BUILDINGS", re.I), "BUILD_2023"),
                    (re.compile(r"S2.*2016", re.I), "SAT_2016"),
                    (re.compile(r"S2.*2023", re.I), "SAT_2023"),
                ]
                 ): 
        self.images_dir = images_dir
        self.rules = rules

    def get_tif_files(self, folder):
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".tif"):
                    yield os.path.join(root, f)

    def get_type(self, filename):
        for pattern, t in self.rules:
            if pattern.search(filename):
                return t
        raise ValueError(f"Unknown file: {filename}")

    def _read_preview(self, path, window=None):
        with rasterio.open(path) as src:
            return src.read(window=window)

    # loops through files with variables t (type), name, data
    def loop_through_files(self):
        for path in self.get_tif_files(self.images_dir):
            name = os.path.basename(path)
            try:
                t = self.get_type(name)
                data = self._read_preview(path)
                yield name, t, data
            except ValueError as e:
                print(e)
