# AlgalBloomDetection
## General Overview
The purpose of this repo is to detect algal blooms using NASA's Moderate Resolution Imaging Spectroradiometer (MODIS) data.
It contains mostly scripts which are intended to be used in sequence.
This project is still in progress and this is not the most up to date branch, though updates are planned for it.

## Workflow
Convert a MODIS 1km MYD021 product hdf format file to a tif file using algaeWarpCorrector_v2.py
Define a box in the format of band_box.csv
Use algaeBandBoxStripper to retrieve data from the converted tif files and write it to a csv file
Train a water detection model with algaeModelControler.py and save the model to disk
Use the saved model and 4paramBandBoxStripper.py to create training data for a 5 parameter model
Train the 5 parameter model using algaeModelControler.py
View the resulting tif file of predictions and probabilities

Training data has been provided to skip some of these steps. It was gathered around western Lake Erie.

## 4paramBandBoxStripper.py

## algaeBandBoxTifStripper.py

## algaeModelController.py

## algaeWarpCorrector.py

## modelVisualization.py

## modularAlgaeModel.py

## modularAlgaeModel5param.py

## modularAlgaeModelBands4.py

## warpCorrection.py
