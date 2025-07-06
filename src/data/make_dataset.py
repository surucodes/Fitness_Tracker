import pandas as pd
from glob import glob
# Understanding the dataset:
# - The dataset contains data from a writst watch that has an accelerometer and a gyroscope. The accelarometer measures the acceleration in three axes (x, y, z)(in m/s2 or in terms of g) and the gyroscope measures the angular velocity in three axes (x, y, z)(in degrees per second).
# Imagine a person doing a bicep curl. As they lift the weight (watch on wrist), the accelerometer’s Z-axis will detect upward acceleration against gravity, followed by a downward acceleration as the arm lowers. The X-axis and Y-axis might show smaller changes if the wrist tilts or rotates slightly during the motion.
# Rotational change mainly around the elbow (y-axis rotation) as the forearm swings : The gyroscope readings.
#so based on the above, the accelerometer and gyroscope data can be used to make the model learn the patterns associated with different exercises.
# This is a multilabel classification problem where the model will learn to classify different exercises based on the accelerometer and gyroscope data.
# The dataset is structured as follows:
# - Each CSV file contains data from a single session and a single metric(accelerator sensor / gyro).
# - A Heavy set, a light set, and a medium set are performed in each session.
# Every two alternating csv files are from the same session which have 5 reps recorded but the first file is from the accelerometer and the second file is from the gyroscope.






# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# the glob function is used to find and list all files that match the pattern
# in this case, it will find all CSV files in the data/raw/MetaMotion directory
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)  # Number of files found in the directory

#--------------------------------------------------------------


# --------------------------------------------------------------
# Extract features from filename
# We extract pieces from the filename to create a DataFrame with metadata.
files[0]
# Example filename:
# '../../data/raw/MetaMotion/B-ohp-heavy2-rpe7_MetaWear_2019-01-11T16.42.43.398_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv'
# The metadata includes:
# - Exercise type (e.g., bench press, squat)
# - Weight category (e.g., heavy, light, medium)
# - Date and time of the session
# - Device ID (e.g., C42732BE255C)
# - Sensor type (e.g., accelerometer, gyroscope)
# - Sampling frequency (e.g., 12.500Hz, 25.000Hz)
# - Version of the data collection software (e.g., 1.4.4)
# - Session ID (e.g., A-bench-heavy2-rpe8)
# - Session type (e.g., bench, squat)
# - Session number (e.g., 2)    

# f = files[0]
# participant=f.split("-")[0].replace(data_path, "")  
# # we used the split method to split the filename by the hyphen (-) and then used indexing to access the first part of the split result, then replaced the data_path with an empty string to get the participant ID. eg B
# label = f.split("-")[1]  # e.g., ohp
# category = f.split("-")[2].rstrip("123").rstrip()  # e.g., remove 2 form heavy2. this denotes the set of the exercise heave / light set

# df =pd.read_csv(f)
# df["participant"] = participant
# df["label"] = label
# df["category"] = category

# # --------------------------------------------------------------


# # --------------------------------------------------------------
# # Read all files
# # --------------------------------------------------------------
# acc_df = pd.DataFrame()
# gyr_df = pd.DataFrame()

# acc_set = 1 
# gyr_set = 1

# for f in files: 
    
#     participant=f.split("-")[0].replace(data_path, "")  
#     label = f.split("-")[1]  # e.g., ohp
#     category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

#     df = pd.read_csv(f)
    
#     df["participant"] = participant
#     df["label"] = label
#     df["category"] = category
    
#     if "Accelerometer" in f:
#         df["set"] = acc_set
#         acc_df = pd.concat([acc_df, df], ignore_index=True)
#         acc_set += 1
#     if "Gyroscope" in f:
#         df["set"] = gyr_set
#         gyr_df = pd.concat([gyr_df, df], ignore_index=True)
#         gyr_set += 1
 
 

#     # print(label , category, participant)
#     # | 🌞 Summer | Clocks go forward (e.g., UTC+2) |
#     # |❄️ Winter | Clocks go back (e.g., UTC+1) |
#     # |📦 DST | Exists to match working hours with daylight |
# # --------------------------------------------------------------
# # Working with datetimes
# # --------------------------------------------------------------
# acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
# gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# del acc_df["epoch (ms)"]
# del acc_df["time (01:00)"]
# del acc_df["elapsed (s)"]

# del gyr_df["epoch (ms)"]
# del gyr_df["time (01:00)"]
# del gyr_df["elapsed (s)"]

# # --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
data_path= "../../data/raw/MetaMotion/"
files = glob("../../data/raw/MetaMotion/*.csv")
def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1 
    gyr_set = 1

    for f in files: 
        
        participant=f.split("-")[0].replace(data_path, "")  
        label = f.split("-")[1]  # e.g., ohp
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_df = pd.concat([acc_df, df], ignore_index=True)
            acc_set += 1
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_df = pd.concat([gyr_df, df], ignore_index=True)
            gyr_set += 1
            
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
# we have two datasets, one for accelerometer and one for gyroscope
# we will merge them based on the index (time) and the participant, label, category
data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)
data_merged.columns = ["acc_x", "acc_y", "acc_z","gyr_x", "gyr_y", "gyr_z","participant", "label", "category", "set"]
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz : Gyro was measuring 25 times per second(every 0.04 seconds , one record) and the accelerometer was measuring 12.5 times per second(every 0.08 second one record ). Hence the gyroscope data is more frequent than the accelerometer data.

# we have to sync up the frequencies of the two datasets.
# If you're trying to combine gyroscope and accelerometer readings into one sample per timestamp (for model input), their time steps must match.

sampling = {    
       "acc_x":"mean",
        "acc_y":"mean",
        "acc_z":"mean",
        "gyr_x":"mean",
        "gyr_y":"mean",
        "gyr_z":"mean",
        "participant":"last",
        "label":"last",
        "category":"last",
        "set":"last" 
}
# This is a custom sampling dictionary that defines how to aggregate the data when resampling.
# For the accelerometer and gyroscope data, we take the mean of the x, y, and z axes.
# For the participant, label, category, and set columns, we take the last value in each resampled period, assuming that these values do not change within the resampling.
days = [g for n,g in data_merged.groupby(pd.Grouper(freq='D'))]
# This groups the data by day, allowing us to resample each day's data separately. to save the computational resources.

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
data_resampled.info()

data_resampled["set"] = data_resampled["set"].astype(int)

data_resampled.info()




# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
# A pickle file is a binary file format used in Python to serialize and deserialize objects. It allows you to save complex data structures, such as DataFrames, to disk and load them back into memory later. No conversions are needed when loading the pickle file, as it retains the original data types and structure of the DataFrame. Helpful in timeseries.