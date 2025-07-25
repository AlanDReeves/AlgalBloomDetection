import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from DataPoint import DataPoint5Point

class FourParamForest:
    def __init__(self):
        self.model: RandomForestClassifier = None
        self.global_Imputer: SimpleImputer = None
        self.scaler: StandardScaler = None
        self.spectra_train, self.spectra_test, self.results_train, self.results_test = (None, None, None, None)

    def calc_NDWI(self, pixel):
        epsilon = 1e-6 # used to avoid divide by zero errors
        num = pixel[1] - pixel[3]
        den = pixel[1] + pixel[3] + epsilon
        ans = num / float(den)
        # (B2 - B4) / (B2 + B4), B2 = green, B4 = NIR
        return ans

    def calc_NDVI(self, pixel):
        epsilon = 1e-6
        num = pixel[3] - pixel[0]
        den = pixel[3] + pixel[0] + epsilon
        ans = num / float(den)
        # (B4 - B1) / (B4 + B1), B1 = red
        return ans

    def train_model(self, training_data_path: str):
        # read in data
        with open(training_data_path, mode='r') as file:
            csv_data = csv.reader(file)
            # convert to list
            csv_data = list(csv_data)
            # remove header if there is one
            if not any(char.isdigit() for char in csv_data[0]):
                csv_data = csv_data[1:]

            # separate spectral data and results
            spectra: np.array = np.zeros((len(csv_data), 4), dtype=int)
            results: np.array = np.zeros(len(csv_data), dtype=int)

            # fill spectra and results
            for i in range(len(csv_data)):
                for j in range(4):
                    spectra[i][j] = int(csv_data[i][j])
                results[i] = int(csv_data[i][4])

            # replace bad data with np.nan. 
            # see MOD09 user guide for details
            spectra = np.where(spectra < -100, np.nan, spectra)

            # impute both targets and non-targets together
            # imputer will not be usable for predictions otherwise
            self.global_Imputer = SimpleImputer(strategy='mean')
            spectra_imputed = self.global_Imputer.fit_transform(spectra)

            # normalize data
            self.scaler = StandardScaler()
            spectra_imputed = self.scaler.fit_transform(spectra_imputed)

            # Create NDWI – Normalized Difference Water Index
            # Create NDVI – Normalized Difference Vegetation Index
            NDWI = np.zeros(shape=results.shape, dtype=float)
            NDVI = np.zeros(shape=results.shape, dtype=float)

            for i in range(len(NDWI)):
                NDWI[i] = self.calc_NDWI(spectra_imputed[i])
                # (B2 - B4) / (B2 + B4), B2 = green, B4 = NIR
                NDVI[i] = self.calc_NDVI(spectra_imputed[i])
                # (B4 - B1) / (B4 + B1), B1 = red

            # reshape for later hstack call
            NDWI = NDWI.reshape(-1,1)
            NDVI = NDVI.reshape(-1,1)

            # add on missing_mask, NDWI, NDVI
            spectra_finished = np.hstack([NDWI, NDVI])

            # split into train and test data
            self.spectra_train, self.spectra_test, self.results_train, self.results_test = train_test_split(spectra_finished, results, test_size=0.2)
            # can add random_state=x as param to have repeated results

            # train model on normalized data
            self.model = RandomForestClassifier(class_weight="balanced")
            self.model.fit(self.spectra_train, self.results_train)

            return True
        
    def get_stats(self):
        # test model on remaining test data
        if self.model == None:
            return None
        results_prediction = self.model.predict(self.spectra_test)

        accuracy = accuracy_score(self.results_test, results_prediction)
        result = f"""Accuracy: {accuracy: .2f}\n
Confusion Matrix:\n
{confusion_matrix(self.results_test, results_prediction)}\n
Classification Report:\n {classification_report(self.results_test, results_prediction)}"""
        
        return result
    
    def get_importances(self):
        if self.model == None:
            return None
        
        importances = self.model.feature_importances_
        feature_names = []
        feature_names.append("NDWI")
        feature_names.append("NDVI")

        importances_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)

        return importances_df
    
    def predict_for_whole_source(self, bands: list[np.ndarray]):
        height, width = bands[0].shape
        feature_vectors = []
        missing_mask = []

        # create a 2D vector representing all pixels in the source image
        # this is required for imputing and predictions
        for i in range(height):
            for j in range(width):
                pixel = []
                mask = []

                for band in bands:
                    val = band[i][j]
                    if val < -100:
                        pixel.append(np.nan)
                        mask.append(1)
                    else:
                        pixel.append(val)
                        mask.append(0)

                feature_vectors.append(pixel)
                missing_mask.append(mask)

        feature_vectors = np.array(feature_vectors, dtype=float)
        missing_mask = np.array(missing_mask, dtype=int)

        # impute
        spectra_imputed = self.global_Imputer.transform(feature_vectors)

        # normalize
        spectra_imputed = self.scaler.transform(spectra_imputed)

        # calculate NDWI, NDVI
        NDWI = np.zeros(shape=(len(spectra_imputed),), dtype=float)
        NDVI = np.zeros(shape=(len(spectra_imputed),), dtype=float)

        for i in range(len(spectra_imputed)):
            NDWI[i] = self.calc_NDWI(spectra_imputed[i])
            NDVI[i] = self.calc_NDVI(spectra_imputed[i])

        # reshape for later hstack call
        NDWI = NDWI.reshape(-1,1)
        NDVI = NDVI.reshape(-1,1)

        # concat missingness mask
        spectra_finished = np.hstack((NDWI, NDVI))

        # predict
        predictions = self.model.predict(spectra_finished)
        probabilities = self.model.predict_proba(spectra_finished)

        # reshape to 2D image
        predictions = predictions.reshape((height, width))

        flat_probs = np.zeros(len(probabilities))
        for i in range(len(probabilities)):
            flat_probs[i] = probabilities[i][1]
        flat_probs = flat_probs.reshape((height), width)

        return predictions, flat_probs


    def predict_for_file(self, file_path: str):
        """Must contain all datapoints of the source hdf file"""
        with open(file_path, 'r') as data_file:
            reader = csv.reader(data_file)
            data = list(reader)

            # remove header if there is one
            if not any(char.isdigit() for char in data[0]):
                data = data[1:]

            # record lat/long of first and last points and convert back to float
            coords = [data[0][5:7], data[len(data) - 1][5:7]]
            for i in range(len(coords)):
                for j in range(len(coords[i])):
                    coords[i][j] = float(coords[i][j])

            for i in range(len(data)):
                # strip data down to only bands
                data[i] = data[i][:4]
                for j in range(4):
                    data[i][j] = int(data[i][j])

        # convert to np array and set up to impute
        data = np.array(data)
        data = np.where(data < -100, np.nan, data)
        # record missingness for later use
        missing_mask = np.isnan(data)

        # impute, avoiding re-fitting the imputer
        data_imputed = self.global_Imputer.transform(data)
        # normalize
        data_imputed = self.scaler.transform(data_imputed)

        # concat missingness mask
        data_finished = np.hstack([data_imputed, missing_mask.astype(int)])

        # obtain results
        classifications = self.model.predict(data_finished)
        probs = self.model.predict_proba(data_finished)
        # reshape classifications
        classifications = classifications.reshape((2400, 2400))
        
        # reshape probabilities
        # probs come in shape (5,760,000 x 2) and so reshaping is less straightforward.
        probs_index = 0
        probs_2D = np.zeros(shape=(2400,2400), dtype=float)

        for i in range(2400):
            for j in range(2400):
                probs_2D[i][j] = probs[probs_index][1]
                probs_index += 1

        return (classifications, probs_2D, coords)
    
    def predict_one_line(self, line: list):
        """Calling this repeatedly is extremely slow."""
        # remove bad data if any
        one_line = np.array(line)
        one_line = np.where(one_line < -100, np.nan, one_line)
        # record missingness
        missing_mask = np.isnan(one_line)

        # impute, do not refit imputer
        line_imputed = self.global_Imputer.transform(one_line)

        # normalize
        line_imputed = self.scaler.transform(line_imputed)

        # concat missingness mask
        line_finished = np.hstack([line_imputed, missing_mask.astype(int)])

        # predict
        return (self.model.predict(line_finished), self.model.predict_proba(line_finished))
    
    def predict_subset(self, file_path: str):
        """Does not create normal shape for output\n
        Useful for creating 5 param training data\n
        Requires a csv file path as input"""
        with open(file_path, 'r') as data_file:
            reader = csv.reader(data_file)
            data = list(reader)

            # remove header if there is one
            if not any(char.isdigit() for char in data[0]):
                data = data[1:]

            # record lat/long of first and last points and convert back to float
            coords = [data[0][5:7], data[len(data) - 1][5:7]]
            for i in range(len(coords)):
                for j in range(len(coords[i])):
                    coords[i][j] = float(coords[i][j])

            for i in range(len(data)):
                # strip data down to only bands
                data[i] = data[i][:4]
                for j in range(4):
                    data[i][j] = int(data[i][j])

        # convert to np array and set up to impute
        data = np.array(data)
        data = np.where(data < -100, np.nan, data)

        # impute, avoiding re-fitting the imputer
        data_imputed = self.global_Imputer.transform(data)
        # normalize
        data_imputed = self.scaler.transform(data_imputed)

        # Create NDWI – Normalized Difference Water Index
        # Create NDVI – Normalized Difference Vegetation Index
        NDWI = np.zeros(shape=data_imputed.shape, dtype=float)
        NDVI = np.zeros(shape=data_imputed.shape, dtype=float)

        for i in range(len(NDWI)):
            NDWI[i] = self.calc_NDWI(data_imputed[i])
            # (B2 - B4) / (B2 + B4), B2 = green, B4 = NIR
            NDVI[i] = self.calc_NDVI(data_imputed[i])
            # (B4 - B1) / (B4 + B1), B1 = red

        # reshape for later hstack call
        NDWI = NDWI.reshape(-1,1)
        NDVI = NDVI.reshape(-1,1)

        # add on missing_mask, NDWI, NDVI
        data_finished = np.hstack([NDWI, NDVI])

        # obtain results
        classifications = self.model.predict(data_finished)
        probs = self.model.predict_proba(data_finished)

        return (classifications, probs, coords)
    
    def create_5param_train_data(self, input_path: str, output_path):
        # [band1, band2, band3, band4, isWater, result, lat, long] 
        probs = self.predict_subset(input_path)[1]

        with open(input_path, 'r') as data_file:
            reader = csv.reader(data_file)
            data = list(reader)

            # remove header if there is one
            if not any(char.isdigit() for char in data[0]):
                data = data[1:]

            with open(output_path, 'w', newline='') as output_file:
                writer = csv.writer(output_file)

                for i in range(len(data)):
                    row = [
                        data[i][0],
                        data[i][1],
                        data[i][2],
                        data[i][3],
                        probs[i][1],
                        data[i][4],
                        data[i][5],
                        data[i][6]]
                    writer.writerow(row)

    
    def save_model_to_disk(self, file_path):
        dump(self, file_path)
        return True

class FiveParamForest:
    def __init__(self):
        self.model: RandomForestClassifier = None
        self.global_Imputer: SimpleImputer = None
        self.scaler: StandardScaler = None
        self.spectra_train, self.spectra_test, self.results_train, self.results_test = (None, None, None, None)

    def train_model(self, training_data_path: str):
        # read in data
        with open(training_data_path, mode='r') as file:
            csv_data = csv.reader(file)
            # convert to list
            csv_data = list(csv_data)
            # remove header if there is one
            if not any(char.isdigit() for char in csv_data[0]):
                csv_data = csv_data[1:]

            # separate spectral data and results
            spectra: np.array = np.zeros((len(csv_data), 4), dtype=int)
            isWater: np.array = np.zeros((len(csv_data), 1), dtype=float) # represents probability that pixel is water
            results: np.array = np.zeros(len(csv_data), dtype=int)

            # fill spectra, isWater, and results
            # [band1, band2, band3, band4, isWater, result, lat, long]
            for i in range(len(csv_data)):
                for j in range(4):
                    spectra[i][j] = int(csv_data[i][j])
                results[i] = int(csv_data[i][5])
                isWater[i] = float(csv_data[i][4])

            # replace bad data with np.nan. 
            # see MOD09 user guide for details
            spectra = np.where(spectra < -100, np.nan, spectra)

            # create mask to show where values were bad
            # this allows missingness to be accounted for in predictions
            missing_mask = np.isnan(spectra)

            # impute both targets and non-targets together
            # imputer will not be usable for predictions otherwise
            self.global_Imputer = SimpleImputer(strategy='mean')
            spectra_imputed = self.global_Imputer.fit_transform(spectra)

            # normalize data
            self.scaler = StandardScaler()
            spectra_imputed = self.scaler.fit_transform(spectra_imputed)

            # add on missing_mask and isWater
            spectra_finished = np.hstack([spectra_imputed, missing_mask.astype(int), isWater])

            # split into train and test data
            self.spectra_train, self.spectra_test, self.results_train, self.results_test = train_test_split(spectra_finished, results, test_size=0.2)
            # can add random_state=x as param to have repeated results

            # train model on normalized data
            self.model = RandomForestClassifier()
            self.model.fit(self.spectra_train, self.results_train)

            return True
        
    def get_stats(self):
        # test model on remaining test data
        if self.model == None:
            return None
        results_prediction = self.model.predict(self.spectra_test)

        accuracy = accuracy_score(self.results_test, results_prediction)
        result = f"""Accuracy: {accuracy: .2f}\n
Confusion Matrix:\n
{confusion_matrix(self.results_test, results_prediction)}\n
Classification Report:\n {classification_report(self.results_test, results_prediction)}"""
        
        return result
    
    def get_importances(self):
        if self.model == None:
            return None
        
        importances = self.model.feature_importances_
        feature_names = [f"band {i}" for i in range(1, 5)]
        for i in range(1, 5):
            feature_names.append(f"band {i} missingness")
        feature_names.append("isWater")

        importances_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)

        return importances_df
    
    def predict_for_whole_source(self, dPoints: list[DataPoint5Point]):
        # this is passed a 2D array of DataPoint5Points
        height, width = dPoints.shape
        feature_vectors = []
        missing_mask = []
        isWater = []

        for i in range(height):
            for j in range(width):
                pixel = []
                mask = []
            
                dp: DataPoint5Point = dPoints[i][j]
                if dp.band1 >= 100:
                    pixel.append(dp.band1)
                    mask.append(0)
                else:
                    pixel.append(np.nan)
                    mask.append(1)
                if dp.band2 >= 100:
                    pixel.append(dp.band1)
                    mask.append(0)
                else:
                    pixel.append(np.nan)
                    mask.append(1)
                if dp.band3 >= 100:
                    pixel.append(dp.band1)
                    mask.append(0)
                else:
                    pixel.append(np.nan)
                    mask.append(1)
                if dp.band4 >= 100:
                    pixel.append(dp.band1)
                    mask.append(0)
                else:
                    pixel.append(np.nan)
                    mask.append(1)

                feature_vectors.append(pixel)
                missing_mask.append(mask)

                isWater.append(dp.isWater)
                
        feature_vectors = np.array(feature_vectors, dtype=float)
        missing_mask = np.array(missing_mask, dtype=int)
        isWater = np.array(isWater, dtype=float)

        # impute
        spectra_imputed = self.global_Imputer.transform(feature_vectors)
        # normalize
        spectra_normalized = self.scaler.transform(spectra_imputed)

        isWater = isWater.reshape(-1,1)

        # concat missingness mask, and isWater
        spectra_finished = np.hstack((spectra_normalized, missing_mask, isWater))

        # predict
        predictions = self.model.predict(spectra_finished)
        probabilities = self.model.predict_proba(spectra_finished)

        # reshape to 2D image
        predictions = predictions.reshape((height, width))

        flat_probs = np.zeros(len(probabilities))
        for i in range(len(probabilities)):
            flat_probs[i] = probabilities[i][1]
        flat_probs = flat_probs.reshape((height), width)

        return predictions, flat_probs
    
    def predict_one_line(self, line: list):
        # remove bad data if any
        one_line = np.array(line)
        one_line = np.where(one_line < -100, np.nan, one_line)
        # record missingness
        missing_mask = np.isnan(one_line)

        # impute, do not refit imputer
        line_imputed = self.global_Imputer.transform(one_line)

        # normalize
        line_imputed = self.scaler.transform(line_imputed)

        # concat missingness mask and isWater
        line_finished = np.hstack([line_imputed, missing_mask.astype(int), float(one_line[4])])

        # predict
        return (self.model.predict(line_finished), self.model.predict_proba(line_finished))
    
    def save_model_to_disk(self, file_path):
        dump(self, file_path)
        return True