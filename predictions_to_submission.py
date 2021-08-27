import os
import shutil

PREDICTIONS_FOLDER = os.path.join("results", "make_predictions", "35")
PREDICTIONS_FOLDER = 'predictions'


for filename in os.listdir(PREDICTIONS_FOLDER):
    task = filename.split('.')[0]
    target = os.path.join("gpt3-baselines", "data", task, "predictions.csv")
    shutil.copy(os.path.join(PREDICTIONS_FOLDER, filename), target)

