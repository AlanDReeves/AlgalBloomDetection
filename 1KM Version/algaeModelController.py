from modularAlgaeModel import Algae_Bloom_Model
from modularAlgaeModelBands4 import Algae_Bloom_Model4
from modularAlgaeModel5param import Algae_Bloom_Model5param
from modelVisualization import AlgalModelVisualizer
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.lift()
root.attributes("-topmost", True)

userInput = 0
done: bool = False
while not done:
    if userInput != 2:
        training_data_path = filedialog.askopenfilename(defaultextension='.csv', title='Select training data file')

    # model = Algae_Bloom_Model(training_data_path)
    # model = Algae_Bloom_Model4(training_data_path)
    model = Algae_Bloom_Model5param(training_data_path)
    model.print_stats()

    print(f"""If the above results are acceptable to continue, enter 1.\n
          To train again on the same data, enter 2\n
          Input anything else to select different data.""")
    try:
        userInput = int(input())
    except:
        pass

    if userInput == 1:
        done = True


# ask user for data to process
evaluation_data = filedialog.askopenfilename(defaultextension='.csv', title="Select data to process")
predictions, probabilities, coordinates = model.predict_for_data(evaluation_data)

# temporarily just printing for testing purposes
print(predictions)
print(probabilities)
print(len(coordinates))

origin_tif_file = filedialog.askopenfilename(defaultextension='.tif', title="Select one of the source tif files for raster data import")
origin_band_box = filedialog.askopenfilename(defaultextension='.csv', title="Select the original band box coordinate file used to collect data to evaluate")

visualizer = AlgalModelVisualizer()
# visualizer.make_prediction_tif(origin_tif_file, coordinates, predictions, probabilities)
visualizer.make_prediction_tif_from_bandbox(origin_tif_file, origin_band_box, predictions, probabilities)

userInput = input("Would you like to save the model to disk?\n Y/N? ")
if userInput == 'y' or userInput == 'Y':
    filename = filedialog.asksaveasfilename(defaultextension='.joblib')
    model.save_model_to_disk(filename)
