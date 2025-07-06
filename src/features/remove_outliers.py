import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
# All the numerical columns in the dataframe are the ones we check for outliers.
outlier_columns = list(df.columns[:6])
# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
# An outlier is an extreme value in a dataset that is much higher or lower than the majority of the data points.
# Box plots can be used to visualize outliers, as they show the distribution of the data and highlight any points that fall outside of the expected range.


plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout = (1, 3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout = (1, 3))
plt.title("Boxplot of the first three outlier columns grouped by label")
plt.show()

# The boxplot only helps us with the visualisation of the outliers, but not with the actual detection.
# We will use different methods to detect outliers in the dataset.

# We instert a function to plot outliers in case of a binary outlier score.
# This function takes a dataset, a column to plot, an outlier column with binary values(true if. outlier flase if not) and a boolean to reset the index for plotting.
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------
# The range between the 25th percentile (Q1) and 75th percentile (Q3) of a dataset.
# So:
# IQR = Q3 - Q1
# Term	Meaning
# Q1	25% of data falls below this value
# Q3	75% of data falls below this value
# IQR	The middle 50% of the data
# Insert IQR function
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
# The IQR method is a common way to detect outliers in a dataset.
# It calculates the interquartile range (IQR) of the data, which is the range between the 25th and 75th percentiles (Q1 and Q3).
# The .quantile() funciton is used to calculate the Q1 and Q3 values.
# Any data point that falls outside of 1.5 times the IQR from either Q1 or Q3 is considered an outlier. 
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
# makes a new column in the dataset with the name of the original column + "_outlier"
# and marks the outliers with True and the non-outliers with False. 
    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Plot a single column

col = "acc_x"
dataset = mark_outliers_iqr(df, col)
# one extra outlier column is added that is a binary value (True if outlier, False if not).
plot_binary_outliers(dataset= dataset, col=col, outlier_col=col + "_outlier", reset_index=True)
# Loop over all columns

for col in outlier_columns : 
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset= dataset, col=col, outlier_col=col + "_outlier", reset_index=True)

#The problem now is that we are looking at the entire dataset rather than based on the type of exercise which makes some of the patters underrepresented 
# So we need to split the data by label/exercise and then apply this outlier detection method. 

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# According to Chauvenet’s criterion we reject a measurement (outlier) from a dataset of size N when it’s probability of observation is less than 1/2N. A generalization is to replace the value 2 with a parameter C.and only a normally distributed dataset ,can be applied with this criterion.

# We can check if a dataset is normally distributed using the Shapiro-Wilk test or the Kolmogorov-Smirnov test.
# The Shapiro-Wilk test is a statistical test that checks if a dataset is normally distributed
# The Kolmogorov-Smirnov test is a non-parametric test that compares the empirical distribution function of a sample with the cumulative distribution function of a reference probability distribution (in this case, the normal distribution).

# Histogram — Do you see a bell shaped curve?
# Boxplot — Is the box symmetrical?
#--------------------------------------------------------------

# Check for normal distribution
df[outlier_columns[:3] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout = (3, 3))
df[outlier_columns[3:] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout = (3, 3))
# As we can see, the data is normally distributed(other than rest), so we can apply Chauvenet's criterion to detect outliers.
# Calculation is complex but it works the same as the IQR method, but it is based on the assumption of a normal distribution.

# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset= dataset, col=col, outlier_col=col + "_outlier", reset_index=True)

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function
# LOF is an unsupervised outlier detection method that identifies anomalies by measuring the local density deviation of a data point with respect to its neighbors.
# density here means how close the data points are to each other in the feature space.
# The Local Outlier Factor (LOF) algorithm compares the local density of a point with the local densities of its neighbors.
# If a point has a significantly lower density than its neighbors,
# it is considered an outlier. The LOF score is a measure of how much a point deviates from its neighbors in terms of density.
# A LOF score greater than 1 indicates that a point is an outlier, while a score close to 1 indicates that the point is similar to its neighbors.
# The Local Outlier Factor (LOF) algorithm is particularly useful for detecting outliers in datasets with varying densities, as it takes into account the local structure of the data.

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_
# It creates a new column in your DataFrame called "outlier_lof" and fills it with:

# True if the row is an outlier (as predicted by LOF)=? -1

# False otherwise


    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

# Loop over all columns
# This works on all of the outlier columns at once, so we can use it to detect outliers in all of the columns.
dataset, outliers , X_score = mark_outliers_lof(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset= dataset, col=col, outlier_col= "outlier_lof", reset_index=True)
# Now we can see that the outliers are maked within the what looks to be normal datapoints. That is the differnece between distance based and distribution based outlier detection methods.
# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------
label = "dead"
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(dataset= dataset, col=col, outlier_col= col + "_outlier", reset_index=True)
# This doesnt look proper as many of the data points are being marked as outliers, so we cannot sacrifice that much of data
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(dataset= dataset, col=col, outlier_col=col + "_outlier", reset_index=True)

dataset, outliers , X_score = mark_outliers_lof(df[df["label"] == label], outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset= dataset, col=col, outlier_col= "outlier_lof", reset_index=True)
# after visual inspection , this method looks good as it makes sense to have some outliers in the data, but not too many. and it is marking the outliers within the bulk of the data as well.
# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------
# now decide what to do with the outliers, we can either remove them or replace them with the mean or median of the column./ impute them etc



# Test on single column
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col)
#  Based on the boolean column of the outliers, we have to transform the actual column!
dataset[dataset["gyr_z_outlier"]]
# We can see that this returns the df where the outliers are marked as True, so we can use this to replace the outliers with the mean or median of the column.

# We first replace the values with NaN.
dataset.loc[dataset["gyr_z_outlier"], "gyr_z"] = np.nan
# df.loc[row_selector, column_selector]
# row_selector: tells pandas which rows you want

# column_selector: tells pandas which columns you want

# You can then read from or assign to this selection


# df.loc[<boolean mask for rows>, <column name>] = new_value



# Create a loop
outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
        dataset.loc[dataset[col + "_outlier"], col] = np.nan
       # df.loc[<boolean mask for rows>, <column name>] = new_value

        # update the original dataframe with the new values
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[col]
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} outliers from {col} for  {label}")
        
outliers_removed_df.info()
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")