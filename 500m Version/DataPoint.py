import numpy as np

class DataPoint4Point:
    def __init__(self, band1: np.int16, band2: np.int16, band3: np.int16, band4: np.int16, isTarget: bool, lat: float, long: float):
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3
        self.band4 = band4
        self.isTarget = isTarget
        self.lat = lat
        self.long = long

    def __repr__(self):
        return (f"{self.band1}, {self.band2}, {self.band3}, {self.band4}, {self.isTarget}, {self.lat}, {self.long}")

class DataPoint5Point:
    def __init__(self, band1: np.int16, band2: np.int16, band3: np.int16, band4: np.int16, isTarget: bool, lat: float, long: float, isWater: float = None):
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3
        self.band4 = band4
        self.isTarget = isTarget
        self.lat = lat
        self.long = long
        self.isWater = isWater

    def __init__(self, fourPoint: DataPoint4Point, isWater: float = None):
        self.band1 = fourPoint.band1
        self.band2 = fourPoint.band2
        self.band3 = fourPoint.band3
        self.band4 = fourPoint.band4
        self.isTarget = fourPoint.isTarget
        self.lat = fourPoint.lat
        self.long = fourPoint.long
        self.isWater = isWater

    def __repr__(self):
        return (f"{self.band1}, {self.band2}, {self.band3}, {self.band4}, {self.isTarget}, {self.lat}, {self.long}, {self.isWater}")