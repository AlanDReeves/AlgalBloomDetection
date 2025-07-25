from osgeo import gdal, osr
import numpy as np
import csv
from DataPoint import DataPoint4Point, DataPoint5Point
from affine import Affine
from HalfKMAlgaeModel import FourParamForest
from joblib import load

class HalfKMStripper:
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        self.bands: list[np.ndarray] = None
        self.lons: np.ndarray = None
        self.lats: np.ndarray = None
        self.isWater: np.ndarray = None
        self.dPoints: np.ndarray[DataPoint4Point] = None
        self.dPointsFive: np.ndarray[DataPoint5Point] = None

        gdal.UseExceptions()

        self.strip_from_hdf()
        self.find_lat_long()
        self.make_DPoints()

    def strip_from_hdf(self):
        # bands: list[np.ndarray] = []
        bands = np.zeros(shape=(4, 2400, 2400), dtype=int)
        for i in range(1, 5):
            # check if each set can be opened
            add_set:gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b0{i}_1')
            if add_set is None:
                # if not opened, raise exception
                raise RuntimeError(f"Band {i} Failed to Open")
            else:
                # append as ndArray
                # bands.append(add_set.ReadAsArray())
                bands[i - 1] = add_set.ReadAsArray()
        
        self.bands = bands
        return bands
    
    def get_transform(self):
        dSet:gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b01_1')
        transform = dSet.GetGeoTransform()
        transform = Affine.from_gdal(*transform)
        return transform

    def get_projection(self):
        dSet:gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b01_1')
        return dSet.GetProjection()

    
    def find_lat_long(self):
        # open an arbitrary set to gather transform and projection data
        temp_set: gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b01_1')

        geo_trans = temp_set.GetGeoTransform() # (origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height)
        proj = temp_set.GetProjection() # gets a string that describes the spatial reference system

        # make a spatial reference object which matches the dataset loaded (sinusoidal)
        src = osr.SpatialReference()
        src.ImportFromWkt(proj)

        # make a spatial reference object with WGS84 reference system
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)

        # make an object which converts sinusoidal reference to WGS84
        transform = osr.CoordinateTransformation(src, dst)

        temp_band_ref = self.bands[0]

        self.lons = np.zeros(temp_band_ref.shape)
        self.lats = np.zeros(temp_band_ref.shape)

        def calc_transform(self, col: int, row: int, geo_trans, transform):
                x_proj = geo_trans[0] + float(col) * geo_trans[1] + float(row) * geo_trans[2]
                # origin_x + (pixel_number * pixel_width) + (line_number * rotation_x)
                y_proj = float(geo_trans[3] + float(col) * geo_trans[4] + float(row) * geo_trans[5])
                # origin_y + (pixel_number * rotation_y) + (line_number * pixel_height)

                # calculate transform to WGS84 and fill lons, lats
                lat, lon, _ = transform.TransformPoint(x_proj, y_proj)
                self.lons[col, row] = lon
                self.lats[col, row] = lat


        for col in range(len(temp_band_ref)):
            for row in range(len(temp_band_ref[0])):
                calc_transform(self, col, row, geo_trans, transform)

        return True
    
    def make_DPoints(self):
        dpoints: np.ndarray[DataPoint4Point] = np.empty((len(self.lats), len(self.lats[0])), dtype=object)
        for col in range(len(self.lats)):
            for row in range(len(self.lats[0])):
                # create a datapoint object with the bands, lat, and long for a specific pixel.
                dpoints[col, row] = DataPoint4Point(
                    self.bands[0][col][row], self.bands[1][col][row], self.bands[2][col][row], self.bands[3][col][row], False,
                    self.lats[col, row], self.lons[col, row]
                )
        self.dPoints = dpoints
        return True

    def strip_by_box(self, lat_high: float, lat_low: float, long_high: float, long_low: float, isTarget: bool):
        result_list = []
        for col in range(len(self.dPoints)):
            # create new col to add to list
            new_col = []
            for row in range(len(self.dPoints[0])):
                dp: DataPoint4Point = self.dPoints[col, row]
                lat_good = False
                long_good = False
                # check if location within lat/long box
                # if inside limits, add band info, lat/long, isTarget to array
                if lat_low <= float(dp.lat) <= lat_high:
                    lat_good = True
                if long_low <= float(dp.long) <= long_high:
                    long_good = True

                if lat_good and long_good:
                    dp.isTarget = isTarget
                    new_col.append(dp)
            if len(new_col) > 0:
                result_list.append(new_col)
        return result_list

    def write_to_csv(self, output_filename: str, box_data: list[list]):
        """box_data: must be a 2D array of datapoint objects"""
        with open(output_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for col in box_data:
                for row in col:
                    file_row = [
                        int(row.band1), 
                        int(row.band2), 
                        int(row.band3), 
                        int(row.band4), 
                        int(row.isTarget), 
                        float(row.lat), 
                        float(row.long)
                        ]
                    writer.writerow(file_row)
        return True
    
    def write_5param_to_csv(self, output_filename: str, box_data: list[list]):
        with open(output_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for col in box_data:
                for row in col:
                    file_row = [
                        int(row.band1), 
                        int(row.band2), 
                        int(row.band3), 
                        int(row.band4), 
                        int(row.isTarget), 
                        float(row.isWater), 
                        float(row.lat), 
                        float(row.long)
                        ]
                    writer.writerow(file_row)
        return True  

    def gen_5th_param(self, model_path: str):
        # load in 4 param model
        four_model: FourParamForest = load(model_path)

        # use existing predict_for_whole_source method to get results
        results = four_model.predict_for_whole_source(self.bands)
        probs = results[1]

        # reserve space for 5 point dPoints
        self.dPointsFive = np.empty(shape=(2400,2400), dtype=DataPoint5Point)

        for i in range(2400):
            for j in range(2400):
                self.dPointsFive[i][j] = DataPoint5Point(self.dPoints[i][j], probs[i][j])

        return self.dPointsFive
    
    def gen_5th_param_csv(self, csv_path):
        pass
