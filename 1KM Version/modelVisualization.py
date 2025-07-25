import rasterio
import rasterio.transform
from rasterio.windows import Window
import numpy as np
import os
import csv

class AlgalModelVisualizer:
    # makes tif files mapping model results to lat/long

    def make_prediction_tif(self, source_tif_path, pred_coordinates, predictions, probabilities):
        # take transform, pixel size, and crs
        with rasterio.open(source_tif_path) as src:
            transform = src.transform
            pixel_width = transform.a
            pixel_height = -transform.e
            orig_crs = src.crs # CRS stands for coordinate reference system

        # determine bounding box of predictions
        lats = []
        lons = []
        for coordinate_set in pred_coordinates:
            lats.append(coordinate_set[0])
            lons.append(coordinate_set[1])

        print(f"\nlats size: {len(lats)}")
        print(f"\nlons size: {len(lons)}")

        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)

        print(f"min_lat: {min_lat}, max_lat: {max_lat}")
        print(f"min_lon: {min_lon}, max_lon: {max_lon}")

        # Below works but sometimes causes off by 1 errors
        row_top, col_left = src.index(min_lon, max_lat)
        row_bot, col_right = src.index(max_lon, min_lat)

        # determine output tif dimensions
        height = max(row_top, row_bot) - min(row_top, row_bot) + 1
        width = max(col_right, col_left) - min(col_right, col_left) + 1

        print(f"row_top: {row_top}, row_bot: {row_bot}")
        print(f"col_right: {col_right}, col_left: {col_left}")

        print(f"\n\n\nHeight found to be: {height}, Width found to be {width}\n\n\n")

        # find origin point of new tif
        new_transform = rasterio.transform.from_origin(
            (transform * (col_left, row_top))[0], # x (longitude)
            (transform * (col_left, row_top))[1], # y (latitude)
            pixel_width,
            pixel_height
        )

        # organize predictions, probs into height x width NumPy array, height = # lats, width = # longs
        prediction_array = np.ones((height, width), dtype=np.int32)
        probs_array = np.zeros((height, width), dtype=np.float32)

        # load predictions into predictions array
        # this depends on the array being the correct size
        predictions_index = 0
        for i in range(height):
            for j in range(width):
                prediction_array[i][j] = predictions[predictions_index]
                predictions_index += 1

        predictions_index = 0
        for i in range(height):
            for j in range(width):
                probs_array[i][j] = probabilities[predictions_index][1]
                predictions_index += 1  

        # write predictions tif
        with rasterio.open(
            "predictions.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=orig_crs,
            transform=new_transform
        ) as dst:
            dst.write(prediction_array, 1)
            # prediction_array is a height x width NumPy array of prediction values
            # must be in correct pixel orientation

        with rasterio.open(
            "probabilities.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=orig_crs,
            transform=new_transform
        ) as dst:
            dst.write(probs_array, 1)
            # prediction_array is a height x width NumPy array of prediction values
            # must be in correct pixel orientation
        
        output_path = os.path.join(os.getcwd(), "predictions.tif")
        print("Saving prediction to:", output_path)

    def make_prediction_tif_from_bandbox(self, source_tif_path, source_bandbox_path, predictions, probabilities):
        # read in coordinates for band box
        coordinates = []
        with open(source_bandbox_path) as csvfile:
            reader = csv.reader(csvfile)
            coordinates = list(reader)
            coordinates = coordinates[1:] # remove header

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
        
        # take transform, pixel size, and crs from original tif
        with rasterio.open(source_tif_path) as src:
            transform = src.transform
            pixel_width = transform.a
            pixel_height = -transform.e
            orig_crs = src.crs # CRS stands for coordinate reference system

            # determine bounding box of prediction
            # Below works but sometimes causes off by 1 errors
            row_top, col_left = rasterio.transform.rowcol(src.transform, lon_top_left, lat_top_left)
            row_bot, col_right = rasterio.transform.rowcol(src.transform, lon_bottom_right, lat_bottom_right)
        
        # convert indices from lists of length 1 to integers
        row_top = row_top[0]
        col_left = col_left[0]
        row_bot = row_bot[0]
        col_right = col_right[0]

        # determine output tif dimensions
        height = max(row_top, row_bot) - min(row_top, row_bot) + 1
        width = max(col_right, col_left) - min(col_right, col_left) + 1

        print(f"row_top: {row_top}, row_bot: {row_bot}")
        print(f"col_right: {col_right}, col_left: {col_left}")

        print(f"\n\n\nHeight found to be: {height}, Width found to be {width}\n\n\n")

        # find origin point of new tif
        new_transform = rasterio.transform.from_origin(
            (transform * (col_left, row_top))[0], # x (longitude)
            (transform * (col_left, row_top))[1], # y (latitude)
            pixel_width,
            pixel_height
        )

        # organize predictions, probs into height x width NumPy array, height = # lats, width = # longs
        prediction_array = np.ones((height, width), dtype=np.int32)
        probs_array = np.zeros((height, width), dtype=np.float32)

        # load predictions into predictions array
        # this depends on the array being the correct size
        predictions_index = 0
        for i in range(height):
            for j in range(width):
                prediction_array[i][j] = predictions[predictions_index]
                predictions_index += 1

        predictions_index = 0
        for i in range(height):
            for j in range(width):
                probs_array[i][j] = probabilities[predictions_index][1]
                predictions_index += 1  

        # write predictions tif
        with rasterio.open(
            "predictions.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=orig_crs,
            transform=new_transform
        ) as dst:
            dst.write(prediction_array, 1)
            # prediction_array is a height x width NumPy array of prediction values
            # must be in correct pixel orientation
        
        output_path = os.path.join(os.getcwd(), "predictions.tif")
        print("Saving prediction to:", output_path)

        # write probabilities tif
        with rasterio.open(
            "probabilities.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=orig_crs,
            transform=new_transform
        ) as dst:
            dst.write(probs_array, 1)
            # prediction_array is a height x width NumPy array of prediction values
            # must be in correct pixel orientation
        
        output_path = os.path.join(os.getcwd(), "probabilities.tif")
        print("Saving probabilities to:", output_path)

