import tkinter as tk
from tkinter import filedialog, simpledialog
from HalfKMAlgaeModel import FourParamForest, FiveParamForest
from HalfKMStripper import HalfKMStripper
from HalfKMVisualization import HalfKMVis
from joblib import load


def show_menu():
    print("HalfKM - MYD09GA Model Controller Main Menu:\n")
    print("Options:")
    print("1: Read in a hdf file")
    print("2: Strip a lat/long box from hdf file")
    print("3: Write lat/long box from memory to disk")
    print("4: Convert current hdf file to a tif file")
    print("5: Train a 4 band classification model")
    print("6: Make predictions for current hdf file")
    print("7: Load previously trained model")
    print("8: Save current model to disk")
    print("9: Generate 5 param model training data from csv")
    print("10: Train 5 param model")
    print("q: Quit\n")

    user_input = input()
    return user_input




if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)

    data_stripper = None
    band_box = None
    visualizer = HalfKMVis()
    four_band_model = None
    five_param_model = None

    # show menu
    user_input = 'x'
    # check user input
    while user_input != 'q':
        user_input = show_menu().lower()

        if user_input == '1': # Strip hdf file in box
            file_path = filedialog.askopenfilename(defaultextension=".hdf", title="Select the MYD09GA file to open")
            print("Opening file and reading. This may take a minute")
            data_stripper = HalfKMStripper(file_path)

        elif user_input == '2': # Strip from lat/long box
            if data_stripper is None:
                print("No source file")
            else:
                # These are asked in this order because the first two make up the top left corner and second two make up the bottom right corner
                # assuming the location given is in the global North West.
                lat_high = simpledialog.askfloat(title="Lat/Long", prompt="Enter Maximum Latitude (Northing)")
                lon_low = simpledialog.askfloat(title="Lat/Long", prompt="Enter Minimum Longitude (Easting)")
                lat_low = simpledialog.askfloat(title="Lat/Long", prompt="Enter Minimum Latitude (Northing)")
                lon_high = simpledialog.askfloat(title="Lat/Long", prompt="Enter Maximum Longitude (Easting)")
                isTarget = simpledialog.askinteger(title="Lat/Long", prompt="Are the pixels in this box targets? 1 - yes, 0 - no\n If this box is for test data, either answer is fine.")

                band_box = data_stripper.strip_by_box(lat_high, lat_low, lon_high, lon_low, isTarget)

        elif user_input == '3': # write box to disk
            if band_box == None:
                print("No box selected")
            else:
                output_name = filedialog.asksaveasfilename(defaultextension=".csv", title="Enter name for output csv file")

                data_stripper.write_to_csv(output_name, band_box)
                print(f"CSV file created at {output_name}")

        elif user_input == '4': # convert hdf to tif
            if data_stripper is None:
                print("No source file")
            else:
                output_path = filedialog.asksaveasfilename(title="Enter name for output tif file")
                output_path = f"{output_path}.tif"
                projection = data_stripper.get_projection()
                transform = data_stripper.get_transform()
                visualizer.write_tif(data_stripper.dPoints, output_path, projection, transform)
                print(f"Tif file created at {output_path}")

        elif user_input == '5': # Train 4 band model
            four_band_model = FourParamForest()
            training_data = filedialog.askopenfilename(defaultextension=".csv", title="Select the training data file to open")

            four_band_model.train_model(training_data)
            print(four_band_model.get_stats())
            print(four_band_model.get_importances())
        
        elif user_input == '6': # make predictions for hdf file:
            print("NOTE: tif files produced will use the currently loaded HDF file's transform")
            model_select = input("Enter 4 for 4 param model\nEnter 5 for 5 param model\n")
            if model_select == '4': # 4 param model
                results = four_band_model.predict_for_whole_source(data_stripper.bands)
                output_path = filedialog.asksaveasfilename(title="Enter name for output tif file")
                visualizer.write_predictions_tif_whole(
                    results[0],  
                    f"{output_path}.tif", 
                    data_stripper.get_projection(), 
                    data_stripper.get_transform()
                    )
                visualizer.write_probs_tif_whole(
                    results[1],  
                    f"{output_path}_probs.tif", 
                    data_stripper.get_projection(), 
                    data_stripper.get_transform()
                    )
                print("Results saved to disk")
            else: # 5 param model
                FourParamPath = filedialog.askopenfilename(defaultextension=".joblib", title="Select 4 param model to use")
                print("Applying model to current hdf")
                data_stripper.gen_5th_param(FourParamPath)
                print("Starting 5 param predictions. This may take a while.")
                results = five_param_model.predict_for_whole_source(data_stripper.dPointsFive)
                output_path = filedialog.asksaveasfilename(title="Enter name for output tif file")
                visualizer.write_predictions_tif_whole(
                    results[0],  
                    f"{output_path}.tif", 
                    data_stripper.get_projection(), 
                    data_stripper.get_transform()
                    )
                visualizer.write_probs_tif_whole(
                    results[1],  
                    f"{output_path}_probs.tif", 
                    data_stripper.get_projection(), 
                    data_stripper.get_transform()
                    )
                print("Results saved to disk")
        
        elif user_input == "7": # load previous model
            model_path = filedialog.askopenfilename(defaultextension=".joblib", title="Select the model file to open")
            four_band_model = load(model_path)

        elif user_input == "8":
            model_path = filedialog.asksaveasfilename(title="Enter filename for model")
            four_band_model.save_model_to_disk(f"{model_path}.joblib")

        elif user_input == "9":
            print("This will use the currently loaded 4 param model to make predictions")
            input_path = filedialog.askopenfilename(defaultextension=".csv", title="Select csv to process")
            output_path = filedialog.asksaveasfilename(title="Enter name for output csv file")
            output_path += ".csv"
            four_band_model.create_5param_train_data(input_path, output_path)

        elif user_input =="10":
            five_param_model = FiveParamForest()
            input_path = filedialog.askopenfilename(defaultextension=".csv", title="Select training data csv")
            five_param_model.train_model(input_path)
            print(five_param_model.get_stats())
            print(five_param_model.get_importances())
