# AlgalBloomDetection
## General Overview
The purpose of this repo is to detect algal blooms using NASA's Moderate Resolution Imaging Spectroradiometer (MODIS) data.
It contains mostly scripts which are intended to be used in sequence.
This project is still in progress and this is not the most up to date branch, though updates are planned for it.

## Dependencies
Besides the Python standard library, the following libraries are required:

Gdal
Numpy
Pandas
Rastorio
Scikit Learn

## Workflow
Convert a MODIS 1km MYD021KM product hdf format file to a tif file using algaeWarpCorrector_v2.py
Define a box in the format of band_box.csv
Use algaeBandBoxStripper to retrieve data from the converted tif files and write it to a csv file
Train a water detection model with algaeModelControler.py and save the model to disk
Use the saved model and 4paramBandBoxStripper.py to create training data for a 5 parameter model
Train the 5 parameter model using algaeModelControler.py
View the resulting tif file of predictions and probabilities

Training data has been provided to skip some of these steps. It was gathered around western Lake Erie.

## 4paramBandBoxStripper.py
A script which strips MODIS data from already converted .tif files and feeds it through an already trained land/water classification model to create training or test data for a 5 parameter model.
In the current version, the tif_folder, output_csv, and input bandbox file path must be entered by manually editing the code.
A pre-generated model has been included in WaterDetectionModel.joblib.

## algaeBandBoxTifStripper.py
A script which strips MODIS data from already converted .tif files.
This is useful for training a land/water classification model required for the 5 parameter model to function.
In the current version, the tif_folder, output_csv, and input bandbox file path must be entered by manually editing the code.

## algaeModelController.py
This is the controller for training or using any of the classification models and creating visualizations.
It prompts the user for data to train an algae detection model, and when the user is satisfied with the model results it continues to ask for data to process.
It prints results to the terminal and creates a .tif raster file to visualize the results, complete with GPS metadata for use in GIS software.

## algaeWarpCorrector_v2.py
This script prompts the user for a file path to a MODIS MYD021KM 1km radiance data hdf file and MYD03 1km geolocation hdf file.
It then corrects warping in the radiance data and produces tif conversions representing bands 1-4 with GPD metadata intact.
Conversions are saved to the user's selected directory and are all named based off a convention set by the user.

## modelVisualization.py
Defines a class called AlgaeModelVisualizer which is used by algaeModelController.py.
### Methods:
make_prediction_tif(source_tif_path, pred_coordinates, predictions, probabilities)
    source_tif_path - the path to the tif file on which the predictions provided are derived.
    pred_coordinates - a list of the lat/long coordinates for the predictions made.
    predictions - a list of binary class predictions.
    probabilities - a 2D list of float probabilities representing the likelyhood that each pixel sampled belongs to either class.
    This method is still under development and is not currently used.
make_prediction_tif_from_bandbox(source_tif_path, source_bandbox_path, predictions, probabilities)
    source_tif_path - the path to the tif file on which the predictions provided are derived.
    source_bandbox_path - the path to the band box used to generate the predictions provided.
    predictions - a list of binary class predictions.
    probabilities - a 2D list of float probabilities representing the likelyhood that each pixel sampled belongs to either class.

## modularAlgaeModel.py
Defines a class called AlgalBloomModel. 
This version uses MODIS bands 8-16 to determine the presence of an algal bloom.
It has been superceded by modularAlgaeModelBands4.py
### Methods:
__init__(training_data_csv)
    training_data_csv - the path to a csv of training data. Formatting is significant.
    This method instantiates the AlgalBloomModel object and trains its ML model.
print_stats()
    Prints to terminal the model's computed accuracy, precision, recall, and f1 scores along with its confusion matrix.
print_importances()
    Prints to terminal the model's weights for each parameter.
predict_preset_test_vals()
    Prints to terminal predictions for a collection of points for which the answer is known. This is intended for debugging.
predict_for_data(self, data_file_name)
    data_file_name - the path to a csv file of data to make predictions for.
    This method creates a prediction for each datapoint included in the csv file provided to it.
    Returns a touple containing the list of predictions, a list of associated probabilities, and coordinates stripped from the original file.
    This method expects the pixel size to be listed immediately under the header in the CSV file.

## modularAlgaeModel5param.py
Defines a class called Algae_Bloom_Model5param.
This version takes MODIS bands 1-4 and a probability value indicating the likelyhood that the pixel examined contains water to predict if the pixel given contains an algal bloom.
### Methods:
__init__(training_data_csv)
    training_data_csv - the path to a csv of training data. Formatting is significant.
    This method instantiates the Algae_Bloom_Model5param object and trains its ML model.
print_stats()
    Prints to terminal the model's computed accuracy, precision, recall, and f1 scores along with its confusion matrix.
print_importances()
    Prints to terminal the model's weights for each parameter.
predict_preset_test_vals()
    Prints to terminal predictions for a collection of points for which the answer is known. This is intended for debugging.
predict_for_data(self, data_file_name)
    data_file_name - the path to a csv file of data to make predictions for.
    This method creates a prediction for each datapoint included in the csv file provided to it.
    Returns a touple containing the list of predictions, a list of associated probabilities, and coordinates stripped from the original file.
    This method expects the pixel size to be listed immediately under the header in the CSV file.
predict_one_line(line)
    line - a list of values including int values of bands 1-4 and a float value indicating the probability that the pixel examined contains water.
    Returns a probability prediction for the one datapoint given to it.

## modularAlgaeModelBands4.py
Defines a class called Algae_Bloom_Model4.
This version takes MODIS bands 1-4 and predicts if the pixel given contained an algal bloom.
### Methods:
__init__(training_data_csv)
    training_data_csv - the path to a csv of training data. Formatting is significant.
    This method instantiates the Algae_Bloom_Model5param object and trains its ML model.
print_stats()
    Prints to terminal the model's computed accuracy, precision, recall, and f1 scores along with its confusion matrix.
print_importances()
    Prints to terminal the model's weights for each parameter.
predict_preset_test_vals()
    Prints to terminal predictions for a collection of points for which the answer is known. This is intended for debugging.
    This method is currently waiting for adjustments and does not work.
predict_for_data(self, data_file_name)
    data_file_name - the path to a csv file of data to make predictions for.
    This method creates a prediction for each datapoint included in the csv file provided to it.
    Returns a touple containing the list of predictions, a list of associated probabilities, and coordinates stripped from the original file.
    This method expects the pixel size to be listed immediately under the header in the CSV file.
predict_one_line(line)
    line - a list of values including int values of bands 1-4 and a float value indicating the probability that the pixel examined contains water.
    Returns a probability prediction for the one datapoint given to it.

## warpCorrection.py
This script is used to fix warping in MODIS MYD021KM hdf files by applying the MYD03 geolocation data to it.
It creates tif files based on the original hdf file and saves them to the disk a the user's chosen location.
### Functions:
create_vrt_and_warp(MYD021KM_path, MYD03_path, band_index, output_tif, swath_type_indicator)
    MYD021KM_path - a path to the desired MYD021KM file.
    MYD03_path - a path to the desired MYD03 file.
    band_index - the band within the desired dataset to convert. Defaults to 1. 
    output_tif - the desired output filename of the converted tif. Defaults to 'output.tif'
    swath_type_indicator - an int indicating the user's desired dataset to use. 1 = 1km refsb, 2 = 500m refsb aggrigate, 3 = 250m refsb aggrigate, 4 = 1km emissive. Defaults to 1.

    Example:  create_vrt_and_warp('1km_path', 'geoloc_path', 1, 'output_tif_file', 2)
        This results in a warp corrected tif file containing only MODIS band 3 which will be called 'output_tif_file.tif'

    This function works by creating a vrt file called 'temp_modis.vrt' which is later used by Gdal. 
    All subsequent runs will rewrite this file and recreate it if it was deleted.

This script is used by algaeWarpCorrector_v2.py

## band_box.csv
An example of the format a band_box file is expected to use.

## erie_algal_bloom_with_isWater.csv
An example of training data for a 5 parameter algal bloom detection model.

## Florida_isWater_test.csv
An example of the format for data to be processed by a 5 parameter algal bloom detection model.
Line 2 represents the pixel size.
Taken from around lake Okeechobee

## isWaterTrainingData.csv
An example of training data for a 4 parameter algal bloom detection model, here reused to detect the presence of water rather than algae.

