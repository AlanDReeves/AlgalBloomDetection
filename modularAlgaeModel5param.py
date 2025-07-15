import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

class Algae_Bloom_Model5param:
    """Modular version of the Algal Bloom Model factory.\n
    This version requires a float value indicating the probability that a given pixel is water.\n
    Requires a path to the training data file as a parameter\n"""
    def __init__(self, training_data_csv):
        """Requires a path to the training data file as a parameter\n
        Training data is expected to have a header but not pixel size"""
        self.model: RandomForestClassifier = None
        self.globalImputer: SimpleImputer = None
        self.scaler: StandardScaler = None
        self.pixel_width = 0
        self.pixel_height = 0

        # read in training data
        with open(training_data_csv, mode='r') as file:
            csvData = csv.reader(file)
            # change from iterator to list
            csvData = list(csvData)
            # remove header
            csvData = csvData[1:]

            # separate spectral data from result
            spectra = []
            results = []
            for line in csvData:
                entry = line[:4]
                entry.append(line[7]) # this is the probability of isWater
                spectra.append(entry)
                results.append(line[4])

            # convert string values to ints
            for i in range(len(spectra)):
                spectra[i] = [int(spectra[i][0]), int(spectra[i][1]), int(spectra[i][2]), int(spectra[i][3]), float(spectra[i][4])]
            results = [[int(cell) for cell in row] for row in results]

            spectra = np.array(spectra)
            results = np.array(results).flatten()

            # set up to impute bad data
            spectra = np.where(spectra >= 60000, np.nan, spectra)

            # create mask to show where values were bad or missing
            # this allows missingness to be accounted for in predictions
            missing_mask = np.isnan(spectra)

            # impute positive and negative together
            self.globalImputer = SimpleImputer(strategy='mean')
            spectra_imputed = self.globalImputer.fit_transform(spectra)

            # normalize data
            self.scaler = StandardScaler()
            spectra_imputed = self.scaler.fit_transform(spectra_imputed)

            # add in the missing_mask
            spectra_finished = np.hstack([spectra_imputed, missing_mask.astype(int)])
            
            # split into train and test data
            self.spectra_train, self.spectra_test, self.results_train, self.results_test = train_test_split(spectra_finished, results, test_size=0.2)
            # random_state = x for repeated results

            # train model on normalized data
            self.model = RandomForestClassifier()
            self.model.fit(self.spectra_train, self.results_train)


    def print_stats(self):
        # test model on remaining data
        results_prediction = self.model.predict(self.spectra_test)

        accuracy = accuracy_score(self.results_test, results_prediction)
        print(f"Accuracy: {accuracy: .2f}")
        print("Confusion Matrix:\n", confusion_matrix(self.results_test, results_prediction))
        print("Classification Report:\n", classification_report(self.results_test, results_prediction))

    def print_importances(self):
        importances = self.model.feature_importances_
        feature_names = [f"band {1 + i}" for i in range(4)]
        feature_names.append("prob isWater")
        for i in range(4):
            feature_names.append(f"band {1 + i} missingness")
        feature_names.append("prob isWater missingness")

        print(f"length of feature_names = {len(feature_names)}")
        print(f"length of importances for model = {len(importances)}")

        # Print or display as DataFrame
        importances_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
            }).sort_values(by="Importance", ascending=False)

        print(importances_df)

    def predict_for_data(self, data_file_name: str) -> tuple:
        """data_file_name: the filename of the data to make predictions for\n
        Returns a tuple containing the predictions and their probabilities"""
        with open(data_file_name, 'r') as data_file:
            reader = csv.reader(data_file)
            data = list(reader)[1:] # converts to list and cut off header

            # read pixel size and then remove from list
            self.pixel_width = data[0][0]
            self.pixel_height = data[0][1]
            data = data[1:]

            # cut off location data
            coordinates = []
            for row in data:
                coordinates.append([float(row[5]), float(row[6])])

            for i in range(len(data)):
                entry = data[i][:4]
                entry = [int(cell) for cell in entry]
                entry.append(float(data[i][7]))
                data[i] = entry

            data = [[int(cell) for cell in row] for row in data] # turns string values into ints
            # convert to array and prepare to impute
            data = np.array(data)
            data = np.where(data >= 60000, np.nan, data)
            # record missingness for later use
            missing_mask = np.isnan(data)

            # impute
            data_imputed = self.globalImputer.transform(data)

            # normalize
            data_imputed = self.scaler.transform(data_imputed)

            # concat missingness mask
            data_finished = np.hstack([data_imputed, missing_mask.astype(int)])

            return (self.model.predict(data_finished), self.model.predict_proba(data_finished), coordinates)
        
    def predict_one_line(self, line: list):
        # remove bad data
        one_line = np.array(line)
        one_line = np.where(one_line >= 60000, np.nan, one_line)
        # record missingness
        missing_mask = np.isnan(one_line)

        # impute
        line_imputed = self.globalImputer.transform(one_line)

        # normalize
        line_imputed = self.scaler.transform(line_imputed)

        # concat missingness mask
        line_finished = np.hstack([line_imputed, missing_mask.astype(int)])

        # predict
        return self.model.predict_proba(line_finished)
    
    def save_model_to_disk(self, filename):
        dump(self, filename)