# First filter subtle noise (not outliers) and identify parts of the data that explain most of the variance. Then add numerical, temporal, frequency, and cluster features.
# Feature engineering is the process of transforming raw data into meaningful features that can be used for machine learning. This involves selecting the most useful features from the raw data and using them to build or create new features that represent the information in the dataset in a more effective way. These new, engineered features can make it easier for machine learning models to learn from the data and make more accurate predictions when done right.
# Examples
# Alter existing data
# Use domain knowledge
# Numerical features
# Temporal features
# Frequency features
# Principle component analysis

# Why do we remove noise and what does this actually mean in this context ?
# 🎧 1. Noise Is Everywhere
# Sensors pick up unwanted vibrations, electrical noise, and jitter.

# Your wrist trembles, the watch shakes slightly

# Sensor hardware itself introduces random voltage fluctuations

# Small movements (like a finger twitch) get over-amplified

# Sampling or transmission errors cause sudden spikes

# Without smoothing, these small, meaningless fluctuations confuse the model.

# 📈 2. Smoothing Helps Reveal the True Signal
# Take this example:

# Imagine you're detecting a bicep curl from gyroscope data (rotation).

# The actual motion is smooth: curl up, pause, lower

# But the raw signal may show:

# Spikes from wrist flicks

# Vibration when weights touch

# Jitter from sensor bounce

# ✅ After applying a low-pass filter or moving average,
# ➡️ You’ll see a cleaner wave that matches the true motion of the lift.

# 🤖 3. Improves Machine Learning Model Accuracy
# Raw data → too noisy → poor pattern learning
# Smoothed data → clear trends → better features → better model

# Example: If you're feeding rolling averages or peak angles into a classifier (e.g., "bicep curl", "push-up", "plank"), smooth data ensures the features reflect actual movements, not noise artifacts.

# 🔍 4. Makes Peak Detection and Segmentation Easier
# If you're using:

# Peak detection (e.g., count pushups or reps)

# Windowing / segmentation (e.g., separate each movement)

# Fourier Transform / Frequency analysis

# → all of these perform MUCH better when data is smoothed
# (no false peaks, no micro-segments, no distorted frequencies)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df= pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()

subset = df[df["set"] == 35]["gyr_y"]
# If we have enough data, we can just drop the rows with missing values.
# If we have too many missing values, we can impute them using the mean or median
# or we can use interpolation to fill in the missing values.
# For this we can use the pandas function interpolate.
for col in predictor_columns:
   df[col] = df[col].interpolate()
df.info() 
# No missing values now as they are interpolated.  
# 
# 
# --------------------------------------------------------------
# Calculating set duration
# This is a prerequisite for the butterworth lopass filter(a filter for subtle noise data)--------------------------------------------------------------
df[df["set"]==24]["acc_y"].plot()
df[df["set"]==50]["acc_y"].plot()

for s in df["set"].unique():
    
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start 
    df.loc[(df["set"] == s),"duration"] = duration.seconds 

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5
# For the heavy set, the average time for a single rep was about 3 sesonds.(total mean duration of the heavy set / the number of reps in that set = 5.)

duration_df.iloc[1] / 10




# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy() 
LowPass = LowPassFilter()
# we have to specify the sampling frequency and the cutoff frequency. Sampling freq means the number of samples taken per second and the cutoff frequency is used to cut the signals above that freq and below will be accepted.
fs = 1000/200 
# step size = 200ms and divide by one second, we get 5 samples per second. 
cutoff = 1.2 
# trial and error
# Higher the cutoff frequency, the more high frequency noise will be accepted. and the more it will look like the rawe data/
# lesser the cutoff, smoother the data will be.
# We dont want it to be too smooth, as we want to keep the peaks and troughs of the data.
# We need to preserve the characterstics of the data/pattern and at the same time we need to get rid of the noise fluctuations. choose cutoff frequency accordingly.

df_lowpass = LowPass.low_pass_filter(df_lowpass,"acc_y",fs,cutoff , order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"])
# visualizing the differnce betweeen the original dataframe and a lowpass version:
fig,ax = plt.subplots(nrows = 2 , sharex=True , figsize =(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[0].set_title("Raw data")
ax[0].set_ylabel("acc_y")
ax[0].set_xlabel("sample")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.15), fancybox=True, shadow=True)
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter", color="orange")
ax[1].set_title("Lowpass data")
ax[1].set_ylabel("acc_y_lowpass")
ax[1].set_xlabel("sample")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.15), fancybox=True, shadow=True)
# As we can see, the lowpass filter has removed the high frequency noise from the data, making it smoother and easier to analyze.
# Now what is the right cutoff frequency?
# This is a trial and error process, we can try different values and see which one works. 


# We overwrite the original columns with the lowpass filtered columns.
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
# PCA is a technique used in machine learning to reduce the complexity of data by transforming the data into a new set of variables called principal components. This transformation is done in such a way that the new set of variables captures the most amount of information from the original data set, while reducing the number of variables necessary. This helps to reduce the complexity of the data and make it easier to analyze and make predictions from.

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
#The function below helps us to determine the n principal components as it returns the value of variance captured by each principal components. Remember that PCA is a linear transformation technique that transforms the data into a new set of variables called principal components. These principal components are linear combinations of the original variables and are orthogonal to each other. The first principal component captures the most variance in the data, the second principal component captures the second most variance, and so on. The number of principal components is determined by the number of original variables in the data set.
pc_values  = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
# we determine the optimal number of principal components by elbow method, where we plot the explained variance and look for the point where the explained variance starts to level off.

# The elbow technique is a method used to determine the optimal number of components to use when conducting a PCA. It works by testing multiple different component numbers and then evaluating the variance captured by each component number. The optimal component number is then chosen as the number of components that capture the most variance while also not incorporating too many components. This is done by plotting the variance captured against the component number and then selecting the point at which the rate of change in variance diminishes (the "elbow"), as this is typically the point at which adding more components does not significantly improve the analysis.

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(pc_values) + 1), pc_values, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance")
plt.title("Elbow Method for PCA")
plt.xticks(range(1, len(pc_values) + 1))
plt.show()
# 3 is the optinal principal components as seen from the elbow method.
# now we apply pca using the number of components we have selected.

df_pca = PCA.apply_pca(df_pca, predictor_columns,3)
# We basically have summarised the 6 predictor columns into 3 principal components, which are linear combinations of the original columns. This helps to reduce the complexity of the data and make it easier to analyze and make predictions from.

# We check during feature selection whether the principal components are useful or not compared to the initial 6 columns. If they are not useful, we can drop them from the dataset.

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1","pca_2","pca_3"]].plot()



# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]

subset[["acc_r", "gyr_r"]].plot(subplots = True)
# r is the scalar magnitude of the three combined data points: x, y, and z. The advantage of using r versus any particular data direction is that it is impartial to device orientation and can handle dynamic re-orientations. r is calculated by:
# r = sqrt(x^2 + y^2 + z^2)     

# Here , we have scalar magnitudes of the accelerometer and gyroscope data, which can be used as features in the model. This is useful as it captures the overall movement of the body in a single value, rather than having to look at each axis separately. This can help to reduce the complexity of the data and make it easier to analyze and make predictions from.
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
# We are going to compute the rolling mean, rolling standard deviation of the data. This is done to capture the temporal patterns in the data and to create features that can be used in the model. The rolling mean is the average of the data over a certain window size, the rolling standard deviation is the standard deviation of the data over a certain window size.
# Temporal abstraction through rolling statistics captures essential dynamic patterns from time series data, enabling machine learning models to reason over windows of time rather than isolated points, thereby improving recognition performance in sequential tasks like human activity recognition.

df_temporal = df_squared.copy()
df_temporal.drop(columns = "duration", inplace=True)

NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
# Finding an optimal window size is a trial and error process, we can try different values and see which one works.
ws = int(1000 / 200)

# we are taking the window size = 1 second. i.e # 1000 ms / 200 ms = 5 samples per second, so we take the rolling mean over 5 samples.
# And if we keep the window size to be too large, we will lose the temporal patterns in the data, as we will be averaging over too many samples and we will lose the initial n number of data samples as well. If we keep it too small, we will not capture the temporal patterns in the data. So we have to find a balance between the two.

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(
        df_temporal, [col], window_size = ws, aggregation_function = "mean"
    )
    df_temporal = NumAbs.abstract_numerical(
        df_temporal, [col], window_size = ws, aggregation_function = "std"
    )

df_temporal
predictor_columns
# Here as we take the rolling mean and standard deviation of the data, we are looking at the previous window size to compute present data. now if my previous set were a squat and now im performing a bench press, The value now will be influenced by the previous set, which is not what we want. 
# So we make subsets based on sets and then compute the rolling mean and standard deviation for each set separately. This way, we are not influenced by the previous set and we are only looking at the current set.

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset= df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(
            subset, [col], window_size=ws, aggregation_function="mean"
        )
        subset = NumAbs.abstract_numerical(
            subset, [col], window_size=ws, aggregation_function="std"
        )
    df_temporal_list.append(subset)
    # This is a list of dataframes, where each dataframe is a subset of the original dataframe based on the set.
    # We can now concatenate these dataframes to get a single dataframe with all the temporal features.
df_temporal = pd.concat(df_temporal_list)
# For the starting of each set , we will have the first ws samples as NaN.
# We have ensured that no data spill over happens, i.e. the rolling mean and standard deviation is not influenced by the previous set.

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)
# we set the ws to the average length of a repetition. 
df_freq = FreqAbs.abstract_frequency(
    df_freq, ["acc_y"], window_size=ws, sampling_rate=fs
)

subset= df_freq[df_freq["set"] == s].copy()
subset[['acc_y','acc_y_max_freq',
       'acc_y_freq_weighted', 'acc_y_pse']].plot( figsize=(20, 10))

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation on set:{s} ")
    subset = df_freq[df_freq["set"] == s].reset_index(drop = True).copy()
    subset = FreqAbs.abstract_frequency(
        subset, predictor_columns, window_size=ws, sampling_rate=fs
    )
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop = True)
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]  # We drop every second row to avoid overlapping windows.
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
# K-means clustering is an unsupervised machine learning algorithm used to group data into clusters based on similarity. It randomly initializes k points (centroids) in the data space, calculates the distance between each data point and each centroid, and assigns each point to its closest centroid. K-means clustering is a popular feature engineering technique used in many applications such as identifying the most important features in a dataset, segmenting customers into different clusters based on their purchase behaviors, anomaly detection, and image compression.

# The Elbow Method, also known as the “elbow curve technique”, is used to determine the optimal number of clusters for a given dataset. It plots the Inertia (sum of squared distances of samples to their closest cluster center) for each k (number of clusters) against k and looks for the “elbow” point in the plot; this point represents the optimal number of clusters. The elbow method works by looping through different values of k and calculating the sum of squared errors for each iteration.
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k,n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, marker="o")    
plt.xlabel("Number of Clusters (k)")        
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for K-Means Clustering")
plt.xticks(k_values)
plt.show()
# The optinal numer of clusters looks like 5 as seen in the elow method plot.

kmeans = KMeans(n_clusters=5,n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)
# this retuns a new column in the dataframe with the cluster labels for each data point.
# cluster lables are arbitrary and can be used to identify the clusters in the data.

# plot clusters in 3d:
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot (projection='3d')
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis") 
ax.set_zlabel("Z-axis")
ax.set_title("3D Scatter Plot of Clusters")
plt.legend()    
plt.show()
# we can compare the original and kmeans clusters(this plot and the below plot) and we can come to the conclusion that k means did a good job in clustering the data based on the accelerometer data as the cluster 1 captures most of the bench press and ohp data and similarly it has made clusters according to differnet ecxercises/

# plot accelerometer data with cluster labels:
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot (projection='3d')
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Exercise {l}")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis") 
ax.set_zlabel("Z-axis")
ax.set_title("3D Scatter Plot of Exercises divided into differnt clusters")
plt.legend()    
plt.show()

# As we can see, bench press and ohp almost are simiular clusters as they need similar movements and are performed in a similar way.
# In the same way, we can see that row and deadlift are also similar clusters as they need similar movements and are performed in a similar way.




















# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")