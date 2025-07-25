import warpCorrection
import tkinter as tk
from tkinter import filedialog, simpledialog
import os

# This version retrieves bands 1-4

root = tk.Tk()
root.withdraw()  # Hide the root window
OneKMPath = filedialog.askopenfilename(defaultextension='.hdf', title='Select 1km spectral data file')
LatLongPath = filedialog.askopenfilename(defaultextension='.hdf', title='Select Lat/Long data file')

output_folder = filedialog.askdirectory(title="Please specify a folder for the output file")
output_tif = simpledialog.askstring(title="Output File", prompt="Enter Desired Output Filename")
full_path = os.path.join(output_folder, output_tif)
# want bands 1-4. These are used in NASA's algorithm

print("Creating .tif conversions for 1km bands")
# convert 250m resolution bands
for i in range(1, 3):
    newPath = full_path + "_band_" + str(i) + "_1KM.tif"
    print(f"Creating tif for band {i} at {newPath}")
    warpCorrection.create_vrt_and_warp(OneKMPath, LatLongPath, band_index=i, output_tif=newPath, swath_type_indicator=3)
    print(f"Band {i} conversion complete")

for i in range(1, 3):
    newPath = f"{full_path}_band_{str(i + 2)}_1KM.tif"
    print(f"Creating tif for band {i + 2} at {newPath}")
    warpCorrection.create_vrt_and_warp(OneKMPath, LatLongPath, band_index=(i + 2), output_tif=newPath, swath_type_indicator=2)
    print(f"Band {i + 2} conversion complete")
