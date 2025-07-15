import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Algae_Bloom_Model:
    """Modular version of the Algal Bloom Model factory.\n
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
                spectra.append(line[:9])
                results.append(line[9])

            # convert string values to ints
            spectra = [[int(cell) for cell in row] for row in spectra]
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
        feature_names = [f"band {8 + i}" for i in range(7)]
        for i in range(9):
            feature_names.append(f"band {8 + i} missingness")

        print(f"length of feature_names = {len(feature_names)}")
        print(f"length of importances for model = {len(importances)}")

        # Print or display as DataFrame
        importances_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
            }).sort_values(by="Importance", ascending=False)

        print(importances_df)

    def predict_preset_test_vals(self):
        extraTestData = [
        [11427, 17008, 22209, 26214, 65533, 65533, 65533, 65533, 65533], # expected positive
        [10465, 15348, 19800, 23884, 31165, 65533, 65533, 65533, 65533], # expected positive
        [21422, 65533, 65533, 65533, 65533, 65533, 65533, 65533, 65533], # cloudy pixel over expected negative
        [9997, 14387, 17969, 21532, 26681, 31879, 65533, 30501, 65533] # expected negative
        ]

        # remove bad data
        extraTestData = np.array(extraTestData)
        extraTestData = np.where(extraTestData >= 60000, np.nan, extraTestData)
        # record missingness
        extra_missing_mask = np.isnan(extraTestData)

        # impute
        extras_imputed = self.globalImputer.fit_transform(extraTestData)

        # normalize
        extras_imputed = self.scaler.fit_transform(extras_imputed)

        # concat missingness mask
        extras_finished = np.hstack([extras_imputed, extra_missing_mask.astype(int)])

        # predict
        for element in extras_finished:
            print(self.model.predict([element]))

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
                coordinates.append([float(row[10]), float(row[11])])
            data = [row[:9] for row in data]

            data = [[int(cell) for cell in row] for row in data] # turns string values into ints
            # convert to array and prepare to impute
            data = np.array(data)
            data = np.where(data >= 60000, np.nan, data)
            # record missingness for later use
            missing_mask = np.isnan(data)

            # impute
            data_imputed = self.globalImputer.fit_transform(data)

            # normalize
            data_imputed = self.scaler.fit_transform(data_imputed)

            # concat missingness mask
            data_finished = np.hstack([data_imputed, missing_mask.astype(int)])

            return (self.model.predict(data_finished), self.model.predict_proba(data_finished), coordinates)
