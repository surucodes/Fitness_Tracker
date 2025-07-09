import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pickle

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")
# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_train = df.drop(["participant","category","set"],axis =1)

x = df_train.drop(["label"], axis=1)
y = df_train["label"]

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=42 , stratify = y)
# The stratify parameter ensures that the class distribution in the training and test sets is similar to the original dataset.In simple words, it ensures that the proportion of each class label in the training and test sets is similar to the proportion in the original dataset. This is particularly useful when dealing with imbalanced datasets, where some classes have significantly more samples than others.for example, if you have a dataset with 80% of samples belonging to class A and 20% belonging to class B, using stratified sampling will ensure that the training set has approximately 80% of samples from class A and 20% from class B, and the test set will have the same distribution.
# visualizing the distribution of labels in train and test sets to make sure stratify has done its job :
fig,ax= plt.subplots(figsize = (10,5))
df_train["label"].value_counts().plot(kind='bar', ax=ax, color='skyblue', label ="total")
y_train.value_counts().plot(kind='bar', ax=ax, color='orange', label ="train")
y_test.value_counts().plot(kind='bar', ax=ax, color='green', label ="test")
ax.set_title("Distribution of Labels in Train and Test Sets")
ax.set_xlabel("Labels")
ax.set_ylabel("Count")
ax.legend()
plt.show()





# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
# This section is to check whether the features we added in the feature engineering part actually has any influence on the predictive modelling/ model performance or not.
# We will split the features into different subsets and compare the performance of the models trained on these subsets.

basic_features = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z"]
square_features = ["acc_r","gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

print("Basic features:",len(basic_features))
print("Square features: ",len(square_features)) 
print("PCA features: ",len(pca_features)) 
print("Time features: ",len(time_features)) 
print("Frequeny features: ",len(freq)) 
print("Cluster features: ",len(cluster_features)) 

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features+ pca_features))
feature_set_3 = list(set(feature_set_2 + time_features ))
feature_set_4 = list(set(feature_set_3 + freq+cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
#  So How Does This Help in Feature Selection?
# If a feature is used early in the tree (closer to the root) or many times, it’s important, it has the highest purity — it contributes the most to reducing uncertainty.


# What Is Forward Selection?
# Forward selection is a wrapper-based feature selection technique.

# You start with no features.

# Then, one-by-one, you add the feature that gives the best performance (here: accuracy) when added to the existing selected set.

# You repeat until you've selected the desired number of features (max_features).

# This helps you build an optimal feature subset for your model.
# Iteration 1:
# Try each one individually:

# acc_x → 72% accuracy

# acc_y → 68%

# gyr_z → 80% ✅

# angle → 65%
# → Select gyr_z

# Iteration 2:
# Try adding each of the remaining to ['gyr_z']:

# acc_x + gyr_z → 83%

# acc_y + gyr_z → 76%

# angle + gyr_z → 79%
# → Select acc_x


learner = ClassificationAlgorithms()

max_features = 10 

selected_features , ordered_features , ordered_scores = learner.forward_selection(max_features,x_train,y_train)

selected_features = ["acc_z_freq_0.0_Hz_ws_14",
"acc_x_freq_0.0_Hz_ws_14",
"gyr_r_pse",
"acc_y_freq_0.0_Hz_ws_14",
"gyr_z_freq_0.714_Hz_ws_14",
"gyr_r_freq_1.071_Hz_ws_14",
"gyr_z_freq_0.357_Hz_ws_14",
"gyr_x_freq_1.071_Hz_ws_14",
"acc_x_max_freq",
"gyr_z_max_freq",
]






plt.figure(figsize = (10,5))
plt.plot(np.arange(1,max_features + 1 ,1), ordered_scores)
plt.xlabel("Number of features")
plt.xticks(np.arange(1,max_features + 1 ,1))
plt.grid()
plt.show()

# The ordered scores show that [0.885556704584626,0.9886246122026887,... the first feature alone derived from the fourier transform gives an 88% accuracy and the second feature+first feature gives 98% accuracy! 
# This shows how important feature engineering is! 



# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# Hyperparameter tuning is always done with cross validation as that is the purpose of cross validation 
# --------------------------------------------------------------
# You train the model on the say 70% train split during cross validation (k fold)

# Then you try different model types or parameters (e.g. 5 vs 10 principal components in PCA) 

# After each change, you check the validation accuracy that is with the validation data in its respective iteration.

# Once you’ve found the best version of your model, you freeze everything

# Then finally, you test it on the test set (which you haven't looked at at all)

# We are gonna train 5 models now, along with grid search 

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features 
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features" 
]

iterations = 1 
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = x_train[possible_feature_sets[i]]
    selected_test_X = x_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    # Non-Deterministic Classifiers:
    # A non-deterministic classifier involves randomness in its algorithm, so even with the same input and same data, it may give slightly different results on different runs (unless you fix the randomness with random_state).

    # Same data → Can give different results (unless seed is fixed)
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    #  Deterministic Classifiers:
    # A deterministic classifier will always give you the same output for the same input, every time you run it — no randomness involved.

    # Same data → Same model → Same predictions
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])
#--------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values(by="accuracy" , ascending= False)

plt.figure(figsize=(10,10))
sns.barplot(x="model", y="accuracy" , hue="feature_set", data = score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7,1)
plt.legend(loc = "lower right")
plt.show()

# We can see that feature set 4 and selected features i.e the freq features and features selected through decision trees really did the trick and hence the great results.

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
(
    class_train_y , 
    class_test_y,
    class_train_prob_y,
    class_test_prob_y
) = learner.random_forest( x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch= True)

accuracy = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test , class_test_y, labels = classes)
# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(["set", "category"], axis =1)

x_train = participant_df[participant_df["participant"] != "A"].drop("label" , axis =1 )
y_train = participant_df[participant_df["participant"] != "A"]["label"]


x_test = participant_df[participant_df["participant"] == "A"].drop("label" , axis =1 )
y_test = participant_df[participant_df["participant"] == "A"]["label"]
# just a way to test 
fig,ax= plt.subplots(figsize = (10,5))
df_train["label"].value_counts().plot(kind='bar', ax=ax, color='skyblue', label ="total")
y_train.value_counts().plot(kind='bar', ax=ax, color='orange', label ="train")
y_test.value_counts().plot(kind='bar', ax=ax, color='green', label ="test")
ax.set_title("Distribution of Labels in Train and Test Sets")
ax.set_xlabel("Labels")
ax.set_ylabel("Count")
ax.legend()
plt.show()
# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
# trying to see if the model has generalised well. 
(class_train_y ,class_test_y,class_train_prob_y,class_test_prob_y,fmodel) = learner.random_forest( x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch= True)

accuracy = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test , class_test_y, labels = classes)


# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# save the model
# Define the path where you want to save the model
save_path = "../../data/interim/modelrandom_forest_model.pkl"

# Ensure the directory exists
import os
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the model
with open(save_path, "wb") as file:
    pickle.dump(fmodel, file)
# 

# My doubt solved : 
# class_test_prob_y contains the outputs for each of the exercises. since this is a multilabel classification , the model outputs with the probability that each exercise belongs to a certain class. 












# ✅ Why Only a single Row Works of test data works — Even Though It's Just a “Single Point in Time”
# It looks like one point, but it’s actually a whole window of time that has been pre-encoded into that row during feature engineering.

# 🔁 What Actually Happened:
# During preprocessing, you did not feed raw accelerometer/gyroscope values directly. Instead, you applied:

# Rolling statistics (mean, std) → over 1–2 second windows

# Fourier Transform (frequency analysis) → on a window

# PCA (dimensionality reduction) on windows

# K-means → clustering motion patterns in a window

# Other aggregations and abstractions → like signal entropy (pse)

# ⚡ So each row in your model input already represents:
# “How the sensor data behaved over a short period of time (e.g., 2 seconds)”

# This means:

# Your one row isn’t one timestamp

# It’s actually a summary of a time window — a compressed representation of motion














# --------------------------------------------------------------
# Try a complex model with the less features
# --------------------------------------------------------------
(
    class_train_y , 
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
    
) = learner.feedforward_neural_network( x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch= False)

accuracy = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test , class_test_y, labels = classes)
# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
