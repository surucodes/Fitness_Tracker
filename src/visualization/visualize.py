import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
df.head(5)
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"]==label]
    fig , ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()
    
for label in df["label"].unique():
    subset = df[df["label"]==label]
    fig , ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams ["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100
# this parameter makes the plot nice and crisp when exported.
# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig , ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_xlabel("sample")
ax.set_ylabel("acc_y")
ax.set_title("Accelerometer data for squat exercise, grouped by category (medium vs. heavy)")
ax.legend(title="Category")
# We can see that for the medium set, the acc_y is higher than for the heavy set, which is expected as the medium set is performed with less weight and it is performed faster.
# This will plot the accelerometer data for the squat exercise, grouped by category (medium vs. heavy) for participant A.

fig , ax = plt.subplots()
# We can also compare the bench exercise for participant B.
# This will plot the accelerometer data for the bench exercise, grouped by category (medium vs. heavy) for participant B.
category2 = df.query("label == 'ohp'").query("participant == 'B'").reset_index()
category2.groupby(["category"])["acc_y"].plot()
ax.set_xlabel("sample")
ax.set_ylabel("acc_y")
ax.set_title("Accelerometer data for bench exercise, grouped by category (medium vs. heavy)")
ax.legend(title="Category")
# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
# We can also compare the accelerometer data for the squat exercise for both participants A and B.as we want the model to gerneralize well, we need to compare the data for both participants.
participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
fig , ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_xlabel("sample")
ax.set_ylabel("acc_y")
ax.set_title("Accelerometer data for squat exercise, grouped by category (medium vs. heavy)")
ax.legend(title="Category")
# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
all_axes_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig , ax = plt.subplots()
all_axes_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
ax.set_xlabel("sample")
ax.set_ylabel("acc_y")
ax.set_title(f"Accelerometer data for {label} exercise, participant {participant}")
plt.legend()
# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
    # no need to concatenate the dataframes, as we are plotting them separately.
        if len(all_axis_df)>0:
          
            fig,ax = plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
            ax.set_xlabel("sample")
            ax.set_ylabel("acc_y")
            plt.title(f"Accelerometer data for {label} exercise, participant {participant}")
            plt.legend()

for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
    # no need to concatenate the dataframes, as we are plotting them separately.
        if len(all_axis_df)>0:
          
            fig,ax = plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            ax.set_xlabel("sample")
            ax.set_ylabel("gyr_y")
            plt.title(f"Accelerometer data for {label} exercise, participant {participant}")
            plt.legend()
# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_plot_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)

fig, ax = plt.subplots(2, figsize=(20, 10), sharex=True)
# this will create a figure with two subplots, one for the accelerometer data and one for the gyroscope data.
# The `sharex=True` parameter will make sure that the x-axis is shared between the two subplots.
# The `figsize` parameter will set the size of the figure to (20, 10) inches.
# The `reset_index(drop=True)` will reset the index of the dataframe and drop the old index, so that the x-axis is not affected by the index

combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
ax[0].set_title(f"Accelerometer data for {label} exercise, participant {participant}")

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,0.15), ncol=3,fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,0.15), ncol=3,fancybox=True, shadow=True)
ax[1].set_xlabel("Sample")

ax[0].set_ylabel("Accelerometer (m/s^2)")
ax[1].set_title(f"Gyroscope data for {label} exercise, participant {participant}")
ax[1].set_ylabel("Gyroscope (rad/s)")
ax[1].set_xlabel("Sample")
plt.tight_layout()
plt.show()
# This will plot the accelerometer and gyroscope data for the row exercise for participant A in a single figure with two subplots, one for the accelerometer data and one for the gyroscope
# data. The x-axis is shared between the two subplots, and the y-axis is labeled accordingly. The `plt.tight_layout()` function is used to adjust the spacing between the subplots
# to prevent overlapping of labels and titles.





# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
    # no need to concatenate the dataframes, as we are plotting them separately.
        if len(combined_plot_df)>0:
          
            fig,ax = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)
    
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
            
            ax[0].set_title(f"Accelerometer data for {label} exercise, participant {participant}")

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,0.15), ncol=3,fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,0.15), ncol=3,fancybox=True, shadow=True)
            ax[1].set_xlabel("Sample")

            ax[0].set_ylabel("Accelerometer (m/s^2)")
            ax[1].set_title(f"Gyroscope data for {label} exercise, participant {participant}")
            ax[1].set_ylabel("Gyroscope (rad/s)")
            ax[1].set_xlabel("Sample")
            plt.tight_layout()
            
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()  