import numpy as np
import matplotlib.pyplot as plt
from dataset.read_data import ReadTifs

# Dynamic World colors for classes 0-8
DW_PALETTE = [
    (65, 155, 223),    # water
    (57, 125, 73),     # trees
    (136, 176, 83),    # grass
    (122, 135, 198),   # flooded_vegetation
    (228, 150, 53),    # crops
    (223, 195, 90),    # shrub_and_scrub
    (196, 40, 27),     # built
    (165, 155, 143),   # bare
    (179, 159, 225)    # snow_and_ice
]
# Normalize to 0-1
DW_PALETTE = np.array(DW_PALETTE) / 255.0

def show_dw_label(data, title):
    data = np.nan_to_num(data, nan=-1)

    data = np.squeeze(data, axis=0).astype(int)  # remove leading band dimension

    rgb = np.zeros(data.shape + (3,), dtype=np.float32)  # shape: (H,W,3)
    for i, color in enumerate(DW_PALETTE):
        mask = data == i
        rgb[mask] = color

    plt.imshow(rgb)

    plt.title(f"{title} ({t})")
    plt.axis("off")
    plt.show()


reader = ReadTifs()

for name, t, data in reader.loop_through_dw():
    show_dw_label(data, name)
