import rasterio
from rasterio.transform import rowcol
import os
import csv
import threading

def strip_tif_files(tif_files: list, row: int, col: int, values: list, active_index: int):
    # make list of touples [(tif_name, active_index)]
    tif_index_list = []
    for filename in tif_files: 
        tif_index_list.append((filename, active_index))
        active_index += 1
    threads = []
    # make one thread per element
    for pair in tif_index_list:
        # strip_tif_file(pair[0], row, col, values, pair[1])
        t = threading.Thread(target=strip_tif_file, args=(pair[0], row, col, values, pair[1]))
        t.start()
        threads.append(t)
    for t in threads:
         t.join()

def strip_tif_file(tif_name: str, row: int, col: int, values: list, active_index: int):
    tif_path = os.path.join(tif_folder, tif_name)
    with rasterio.open(tif_path) as src:
        try:
            value = src.read(1)[row, col]
            print(f"Value of {row}, {col} read as {value}")
        except:
            print(f"Pixel {row}, {col} is out of bounds. Skipping.")
            value = 65533 # this value will be imputed out to the average.
            # If an entire feature (band) is thrown out by the imputer, using this value prevents an exception from occuring
        values[active_index] = value

tif_folder = "workingTifs"
output_csv = "isWater_test_FL.csv"

results = []
coordinates = []

# read in desired coordinates from file
# assuming this contains 4 corners of a box, with sides parallel to lat/long
with open ('band_box.csv') as lat_long_file:
    reader = csv.reader(lat_long_file)
    coordinates = list(reader)
    coordinates = coordinates[1:] # skip header

# stop if band box csv does not have correct fields
if len(coordinates) < 5:
    raise ValueError("Missing fields in band box file")

# convert coordinates to floats
coordinates = [[float(val) for val in row] for row in coordinates]

# name box coordinates for easier tracking
lat_top_left = coordinates[0]
lon_top_left = coordinates[1]
lat_bottom_right = coordinates[2]
lon_bottom_right = coordinates[3]

# create list of tif files
tif_files = sorted([f for f in os.listdir(tif_folder) if f.endswith('.tif')])
first_file = os.path.join(tif_folder, tif_files[0])

# record pixel size from tif file
with rasterio.open(first_file) as src:
    transform = src.transform
    pixel_width = transform.a
    pixel_height = -transform.e
    # pixel_height is negative normally so this flips the sign
    pixel_dimensions = [pixel_width, pixel_height]

with open(output_csv, 'a', newline='') as csvfile:
            # write pixel size to file
            writer = csv.writer(csvfile)
            writer.writerow(pixel_dimensions)
            print(f"Pixel size recorded as: {pixel_width} x {pixel_height}")

print(f"getting indices from {first_file}")

# initialize banding box limits to set scope outside
top_side_index, left_side_index, bottom_side_index, right_side_index = 0, 0, 0, 0

# determine indices for left, right, top, and bottom limits
with rasterio.open(first_file) as src:
    row_top, col_left = rowcol(src.transform, lon_top_left, lat_top_left)
    row_bottom, col_right = rowcol(src.transform, lon_bottom_right, lat_bottom_right)

# convert indices from lists of length 1 to integers
row_top = row_top[0]
col_left = col_left[0]
row_bottom = row_bottom[0]
col_right = col_right[0]

print(f"Top: {row_top}, left: {col_left}, bottom: {row_bottom}, right: {col_right}")

for row in range(row_top, row_bottom + 1):
    for col in range(col_left, col_right + 1):
        values = values = [0] * 7

        strip_tif_files(tif_files, row, col, values, 0)

        # print to csv output file
        values[4] = int(coordinates[4][0]) # this requires coordinates[4] to be either 0 or 1 to indicate if
        # the band box contains targeted data or not.

        # get longitude and latitude from pixel
        # can comment this out if undesired later
        lon, lat = src.transform * (col, row)
        values[5] = lat
        values[6] = lon

        with open(output_csv, 'a', newline='') as csvfile:
            # write results to file
            writer = csv.writer(csvfile)
            # writer.writerow(values)
            writer.writerow(values)
            print(f"Writing values for {row} {col}")

print(f"Saved values to {output_csv}")