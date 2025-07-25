from DataPoint import DataPoint4Point
import numpy as np
import rasterio
from affine import Affine

class HalfKMVis:

    def write_tif(self, dpoints: np.ndarray[DataPoint4Point], output_path: str, projection: str, transform: Affine):
        """Writes bands 1-4 to a tif at the path indicated as param output_path"""
        # gather data from dpoints
            # unpack from object
        rows, cols = dpoints.shape

        band1 = np.zeros((rows, cols), dtype=np.int16)
        band2 = np.zeros((rows, cols), dtype=np.int16)
        band3 = np.zeros((rows, cols), dtype=np.int16)
        band4 = np.zeros((rows, cols), dtype=np.int16)

        for i in range(cols):
            for j in range(rows):
                band1[i, j] = (dpoints[i, j].band1)
                band2[i, j] = (dpoints[i, j].band2)
                band3[i, j] = (dpoints[i, j].band3)
                band4[i, j] = (dpoints[i, j].band4)

        # write to tif
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=4,
            dtype=np.int16,
            crs=projection,
            transform=transform
        ) as dst:
            dst.write(band1, 1)
            dst.write(band2, 2)
            dst.write(band3, 3)
            dst.write(band4, 4)

    def write_probs_tif(self, dpoints: np.ndarray[DataPoint4Point], probabilities: np.ndarray[float], coords:list,  output_path: str, projection: str, transform: Affine):
        """THIS METHOD DOES NOT WORK
        Writes prediction probabilities to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform"""

        # find how many rows, cols off of hdf transform start
        rows, cols = self.get_col_row(dpoints, coords[0][1], coords[0][0])
        # probabilities[0][2] - longitude, probabilities[0][1] - latitude

        new_origin_x = transform[0] + cols * transform[1]
        # top left x + cols * px width
        new_origin_y = transform[3] + rows * transform[5]
        # top left y + rows * px height

        # write new transform for overlay
        new_transform = (new_origin_x, transform[1], 0, new_origin_y, 0, transform[5])

        # make 2D array of only probabilities
        bounds = self.reform_box(coords[0][0], coords[1][0], coords[0][1], coords[1][1], dpoints)
        x_size = bounds[1] - bounds[0] + 1
        y_size = bounds[3] - bounds[2] + 1

        probs_2D = np.zeros(shape=(x_size, y_size), dtype=float)
        prob_index = 0

        print(f"x_size: {x_size}, y_size: {y_size}")
        print(f"Array total size: {x_size * y_size}")

        for i in range(x_size):
            for j in range(y_size):
                probs_2D[i][j] = probabilities[prob_index][1] # This takes the probability of class 1
                prob_index += 1
                # print(f"Wrote probability number {prob_index + 1} to array")


        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=y_size,
            width=x_size,
            count=1,
            dtype="float32",
            crs=projection,
            transform=new_transform,
        ) as dst:
            dst.write(probs_2D)

    def write_predictions_tif_whole(self, predictions: np.ndarray[int], output_path: str, projection: str, transform: Affine):
        """Writes binary predictions to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform"""
        rows, cols = predictions.shape

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype="int32",
            crs=projection,
            transform=transform,
        ) as dst:
            # add a fake dimension so that rastorio isn't confused.
            dst.write(predictions[np.newaxis, :, :])

    def write_probs_tif_whole(self, probs: np.ndarray[float], output_path: str, projection: str, transform: Affine):
        """Writes binary predictions to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform"""
        rows, cols = probs.shape

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs=projection,
            transform=transform,
        ) as dst:
            dst.write(probs[np.newaxis, :, :])

    def get_col_row(self, dpoints: np.ndarray[DataPoint4Point], lon: float, lat: float):
        min_dist = float('inf')
        best_col, best_row = -1, 1

        for col in range(len(dpoints)):
            for row in range(len(dpoints[0])):
                dp = dpoints[col][row]
                dist = abs(float(dp.lat) - lat) + abs(float(dp.long) - lon)

                if dist < min_dist:
                    min_dist = dist
                    best_col, best_row = col, row
        
        if best_col == -1 or best_row == -1:
            raise ValueError("No matching point found for given lat/lon.")
        
        return best_col, best_row
                
    def reform_box(self, lat_high: float, lat_low: float, long_high: float, long_low: float, dpoints: np.ndarray[DataPoint4Point]):
        """[start_col, end_col, start_row, end_row]"""
        col1, row1 = self.get_col_row(dpoints, long_high, lat_high)
        col2, row2 = self.get_col_row(dpoints, long_low, lat_low)

        start_col = min(col1, col2)
        end_col = max(col1, col2)
        start_row = min(row1, row2)
        end_row = max(row1, row2)

        return [start_col, end_col, start_row, end_row]