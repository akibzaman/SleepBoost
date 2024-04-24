import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import os
import glob

# read the path
# os.path.expanduser('~/Documents/Data/*.csv')
# file_list = glob.glob("/Users/akibzaman/Codes/SleepBoost/Data/alldataretest/fs/*.{}".format("csv"))
# file_list = glob.glob("/Users/akibzaman/Codes/SleepBoost/Data/alldataretest/train/*.{}".format("csv"))
file_list = glob.glob("/Users/akibzaman/Codes/SleepBoost/Git Code/SleepBoost/data/alldatacsv/*.{}".format("csv"))

# print(file_list)
nFiles = len(file_list)
data = pd.DataFrame()
# append all files together
for file in file_list:
    df_temp = pd.read_csv(file)
    data = pd.concat(
        [data, df_temp], ignore_index=True
    )  # data.append(df_temp, ignore_index=True)

print(data.shape)

# # data = pd.read_csv("alldatacsv/sample04.csv")
labelencoder = LabelEncoder()
data["class"] = labelencoder.fit_transform(data["class"])
X, y = data.loc[:, "mean":"D19"], data["class"]
# nunique = data.nunique()
# cols_to_drop = nunique[nunique == 1].index
# data.drop(cols_to_drop, axis=1)
feature_names = X.columns.values.tolist()
test = y


# SETTING THE THRESHOLD
# THRESHOLD = 0.1
# THRESHOLD = 0.2
THRESHOLD = 0.3
# THRESHOLD = 0.23



####### SCATTER PLOT #############

# Assuming 'X', 'test', and 'class_to_stage' are already defined
# class_to_stage dictionary mapping
class_to_stage = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

# Step 1: Find the unique classes in 'test'
unique_classes = np.unique(test)

# Rename the first fourteen features in X to T1 to T14
new_feature_names = [f"T{i}" for i in range(1, 15)] + list(X.columns[14:])
X_renamed = X.copy()
X_renamed.columns = new_feature_names

# Preparing the data structure to store MI scores
mi_scores = pd.DataFrame(index=X_renamed.columns)

# Calculate MI for each class individually and store in DataFrame
for cls in unique_classes:
    stage_name = class_to_stage.get(cls, f"Class {cls}")  # Get stage name
    binary_target = (test == cls).astype(int)  # Binary target for the current class
    mi_scores[stage_name] = mutual_info_classif(X_renamed, binary_target)

# Calculate and add MI scores for all classes together
mi_scores['All Classes'] = mutual_info_classif(X_renamed, test)
print(mi_scores['All Classes'])

# Assuming 'THRESHOLD' and 'mi_scores' are predefined
# Define the plot size, style, and palette
plt.figure(figsize=(20, 10))
sns.set(style="whitegrid")
palette = sns.color_palette("tab10")  # 'tab10' offers a set of distinct colors

# Iterate over each class for plotting
for i, column in enumerate(mi_scores.columns[:-1]):  # Skip "All Classes"
    plt.scatter(x=mi_scores.index, y=mi_scores[column], color=palette[i % len(palette)], s=100, alpha=0.9, label=column)

# Highlight "All Classes" differently
plt.scatter(x=mi_scores.index, y=mi_scores['All Classes'], color='grey', s=100, alpha=0.9, label='All Classes', marker='x')

# Highlight features above the threshold in "All Classes"
above_threshold = mi_scores['All Classes'] > THRESHOLD
plt.scatter(mi_scores.index[above_threshold], mi_scores['All Classes'][above_threshold], color='magenta', s=150, label='Above Threshold', zorder=5, marker='*')

# Add a horizontal line for the MI threshold
plt.axhline(y=THRESHOLD, color='r', linestyle='--', linewidth=2, label='Threshold Line')

# Legend, titles, and labels
plt.legend(title='Sleep Stages & Highlights', loc='upper right', fontsize=16, title_fontsize=18)
plt.title('Mutual Information Scores per Feature Across Sleep Stages', fontsize=24, color='black')
plt.xlabel('Features', fontsize=20)
plt.ylabel('Mutual Information Score', fontsize=20)

# Customizing tick labels for clarity
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()

# Save and optionally display the plot
plt.savefig('Non-Scaled Class_Feature MI comparison Scatter.png', dpi=300, bbox_inches='tight')
# plt.show()  # Uncomment if you wish to display the plot directly




# Assuming 'mi_scores_scaled', 'scaled_threshold', and class_names are predefined

# Step 1: Find the Min and Max MI for "All Classes"
min_mi_all_classes = mi_scores['All Classes'].min()
max_mi_all_classes = mi_scores['All Classes'].max()

# Determining THRESHOLD as a percentage of the range
range_all_classes = max_mi_all_classes - min_mi_all_classes
threshold_percentage = (THRESHOLD - min_mi_all_classes) / range_all_classes

# Step 2: Scale all the values per class to that range
scaler = MinMaxScaler(feature_range=(min_mi_all_classes, max_mi_all_classes))

# Dropping 'All Classes' column to avoid scaling it
mi_scores_scaled = mi_scores.drop(columns=['All Classes'])

# Applying scaling
mi_scores_scaled = pd.DataFrame(scaler.fit_transform(mi_scores_scaled), index=mi_scores_scaled.index, columns=mi_scores_scaled.columns)

# Inserting 'All Classes' back to scaled DataFrame
mi_scores_scaled['All Classes'] = mi_scores['All Classes']

# Now adjust the threshold to be a specific value within the new scaled range
scaled_threshold = min_mi_all_classes + threshold_percentage * range_all_classes

# Set Seaborn style for plotting
sns.set(style="whitegrid")

# Initialize the plot with 5 subplots arranged vertically
fig, axes = plt.subplots(5, 1, figsize=(20, 30), sharex=True)

# # Class names - assuming these are the names in 'mi_scores_scaled'
class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

for i, cls in enumerate(class_names):
    # Scatter plot for the specific class's scaled MI scores
    axes[i].scatter(mi_scores_scaled.index, mi_scores_scaled[cls], color='blue', s=100, alpha=0.5, label=f"{cls} (Scaled)")
    
    # Scatter plot for "All Classes" scaled MI scores for comparison
    axes[i].scatter(mi_scores_scaled.index, mi_scores_scaled['All Classes'], color='grey', s=100, alpha=0.5, marker='x', label="All Classes (Scaled)")
    
    # Highlight points above the threshold in the individual class
    above_threshold_class = mi_scores_scaled[cls] > scaled_threshold
    axes[i].scatter(mi_scores_scaled.index[above_threshold_class], mi_scores_scaled[cls][above_threshold_class], color='red', s=150, label='Above Threshold (This Class)', zorder=5)
    
    # Highlight points above the threshold in "All Classes"
    above_threshold_all = mi_scores_scaled['All Classes'] > scaled_threshold
    axes[i].scatter(mi_scores_scaled.index[above_threshold_all], mi_scores_scaled['All Classes'][above_threshold_all], color='purple', s=150, marker='*', label='Above Threshold (All Classes)', zorder=5)
    
    # Solid red line for the threshold
    axes[i].axhline(y=scaled_threshold, color='red', linestyle='-', linewidth=2, label='Scaled Threshold')
    
    # Setting titles, labels, and customizing ticks and legend
    axes[i].set_title(f"Class {cls} vs. All Classes (Scaled MI Scores)", fontsize=16)
    axes[i].set_xlabel("Features", fontsize=16)
    axes[i].set_ylabel("Scaled MI Score", fontsize=16)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()

plt.tight_layout()

# Save the plot as an image file
plt.savefig('Scaled Class_Feature (Individual) MI Comparison with Highlight Scatter.png', dpi=300, bbox_inches='tight')
# plt.show()  # Uncomment to display the plot directly



####### SCATTER PLOT #############


# # Assuming 'X', 'test', and 'class_to_stage' are already defined
# # class_to_stage dictionary mapping
# class_to_stage = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

# # Step 1: Find the unique classes in 'test'
# unique_classes = np.unique(test)

# # Rename the first fourteen features in X to T1 to T14
# new_feature_names = [f"T{i}" for i in range(1, 15)] + list(X.columns[14:])
# X_renamed = X.copy()
# X_renamed.columns = new_feature_names

# # Preparing the data structure to store MI scores
# mi_scores = pd.DataFrame(index=X_renamed.columns)

# # Calculate MI for each class individually and store in DataFrame
# for cls in unique_classes:
#     stage_name = class_to_stage.get(cls, f"Class {cls}")  # Get stage name
#     binary_target = (test == cls).astype(int)  # Binary target for the current class
#     mi_scores[stage_name] = mutual_info_classif(X_renamed, binary_target)

# # Calculate and add MI scores for all classes together
# mi_scores['All Classes'] = mutual_info_classif(X_renamed, test)
# print(mi_scores['All Classes'])



#  ####### Visualisation 01 :  Non-Scaled Class_Feature MI comparison
# sns.set(style="whitegrid")

# # Generate the plot with enhancements
# plt.figure(figsize=(20, 10))

# # Set a visually appealing palette with distinct colors
# palette = sns.color_palette("tab10")  # 'tab10' offers a set of distinct colors

# for i, column in enumerate(mi_scores.columns[:-1]):  # Skip "All Classes" for individual plotting
#     plt.plot(mi_scores.index, mi_scores[column], marker='o', color=palette[i % len(palette)], linewidth=2, alpha=0.9, label=column, markersize=5)

# # Highlight features above threshold in "All Classes"
# above_threshold = mi_scores['All Classes'] > THRESHOLD
# plt.scatter(mi_scores.index[above_threshold], mi_scores['All Classes'][above_threshold], color='magenta', s=100, label='Above Threshold', zorder=5)

# # Plot "All Classes" with a neutral color and markers
# plt.plot(mi_scores.index, mi_scores['All Classes'], marker='o', color='grey', linewidth=2, alpha=0.9, label='All Classes', markersize=5)

# # Add a horizontal line for the MI threshold at THRESHOLD
# plt.axhline(y=THRESHOLD, color='r', linestyle='--', linewidth=2, label='Threshold Line')

# # Add legend in the upper right
# plt.legend(title='Sleep Stages & Highlights', loc='upper right', fontsize=12, title_fontsize=14)

# # Add titles and labels with increased font size for readability and clarity
# plt.title('Mutual Information Scores per Feature Across Sleep Stages', fontsize=24, color='black')
# plt.xlabel('Features', fontsize=20)
# plt.ylabel('Mutual Information Score', fontsize=20)

# # Customize tick labels for clarity
# plt.xticks(rotation=45, fontsize=14)
# plt.yticks(fontsize=14)

# plt.tight_layout()

# # Save the plot as an image file
# plt.savefig('Non-Scaled Class_Feature MI comparison.png', dpi=300, bbox_inches='tight')
# # plt.show()  # Uncomment if you wish to display the plot directly


# ######### Visualisation 02: Scaled Class_Feature (Individual) MI Comparison with Highlight

# # Assuming 'mi_scores' DataFrame is already defined and contains MI scores for all classes and individual classes

# # Step 1: Find the Min and Max MI for "All Classes"
# min_mi_all_classes = mi_scores['All Classes'].min()
# max_mi_all_classes = mi_scores['All Classes'].max()

# # Determining THRESHOLD as a percentage of the range
# range_all_classes = max_mi_all_classes - min_mi_all_classes
# threshold_percentage = (THRESHOLD - min_mi_all_classes) / range_all_classes

# # Step 2: Scale all the values per class to that range
# scaler = MinMaxScaler(feature_range=(min_mi_all_classes, max_mi_all_classes))

# # Dropping 'All Classes' column to avoid scaling it
# mi_scores_scaled = mi_scores.drop(columns=['All Classes'])

# # Applying scaling
# mi_scores_scaled = pd.DataFrame(scaler.fit_transform(mi_scores_scaled), index=mi_scores_scaled.index, columns=mi_scores_scaled.columns)

# # Inserting 'All Classes' back to scaled DataFrame
# mi_scores_scaled['All Classes'] = mi_scores['All Classes']

# # Now adjust the threshold to be a specific value within the new scaled range
# scaled_threshold = min_mi_all_classes + threshold_percentage * range_all_classes

# # Set Seaborn style for the plotting
# sns.set(style="whitegrid")

# # Initialize the plot with 5 subplots (for 5 classes) arranged vertically
# fig, axes = plt.subplots(5, 1, figsize=(20, 30))

# # Class names - assuming these are the names in 'mi_scores_scaled'
# class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

# for i, cls in enumerate(class_names):
#     # Plot MI scores for the specific class
#     axes[i].plot(mi_scores_scaled.index, mi_scores_scaled[cls], label=f"{cls} (Scaled)", color='blue', alpha=0.5, linestyle='-', marker='o', markersize=5)
    
#     # Plot MI scores for "All Classes" for comparison
#     axes[i].plot(mi_scores_scaled.index, mi_scores_scaled['All Classes'], label="All Classes (Scaled)", color='grey', alpha=0.5, linestyle='--', marker='o', markersize=5)
    
#     # Identify points above the threshold in the individual class
#     above_threshold_class = mi_scores_scaled[cls] > scaled_threshold
#     # Identify points below the threshold for "All Classes"
#     below_threshold_all = mi_scores_scaled['All Classes'] <= scaled_threshold
    
#     # Highlight points that are only above in individual class with RED
#     individual_only_above = above_threshold_class & below_threshold_all
#     axes[i].scatter(mi_scores_scaled.index[individual_only_above], mi_scores_scaled[cls][individual_only_above], color='red', s=100, label='Neglected Individually Significant Feature', zorder=5)
    
#     # Highlight remaining points above the threshold in the individual class with GREEN dots (excluding those marked red)
#     axes[i].scatter(mi_scores_scaled.index[above_threshold_class & ~individual_only_above], mi_scores_scaled[cls][above_threshold_class & ~individual_only_above], color='green', s=100, label='Above Threshold (This Class)', zorder=5)
    
#     # Highlight points above the threshold in "All Classes" with PURPLE dots
#     above_threshold_all = mi_scores_scaled['All Classes'] > scaled_threshold
#     axes[i].scatter(mi_scores_scaled.index[above_threshold_all], mi_scores_scaled['All Classes'][above_threshold_all], color='purple', s=100, label='Above Threshold (All Classes)', zorder=5)
    
#     # Draw a SOLID RED LINE for the threshold
#     axes[i].axhline(y=scaled_threshold, color='red', linestyle='-', linewidth=2, label='Scaled Threshold')
    
#     # Set titles and labels
#     axes[i].set_title(f"Class {cls} vs. All Classes (Scaled MI Scores)", fontsize=16)
#     axes[i].set_xlabel("Features", fontsize=14)
#     axes[i].set_ylabel("Scaled MI Score", fontsize=14)
    
#     # Customize ticks and legend
#     axes[i].tick_params(axis='x', rotation=45)
#     axes[i].legend()

# plt.tight_layout()

# # Save the plot as an image file
# plt.savefig('Scaled Class_Feature (Individual) MI Comparison with Highlight.png', dpi=300, bbox_inches='tight')
# # plt.show()  # Uncomment to display the plot directly


# #####################  CREATE SELECTED FEATURE DATASET  #####################
# MI_threshold = 0.23
# fs = SelectKBest(score_func=mutual_info_classif, k="all")
# fs.fit(X, test)
# # what are scores for the features
# sum = 0
# for i in range(len(fs.scores_)):
#     sum = sum + fs.scores_[i]
#     print("Feature %d: %f" % (i, fs.scores_[i]))

# mean = sum / 40
# print("Mean: {0}".format(mean))
# # what are scores for the features
# selected_feature = []
# for i in range(len(fs.scores_)):
#     if fs.scores_[i] > MI_threshold:
#         selected_feature.append(fs.feature_names_in_[i])
#         print("Feature %d %s: %f" % (i, fs.feature_names_in_[i], fs.scores_[i]))

# # #print(selected_feature)
# data_selected = data[selected_feature]

# ####### IMPORT AS CSV (SELECTED DATA)

# y = data["class"]
# data_frame = [data_selected, y]
# data_selected = pd.concat(data_frame, axis=1)

# # data_selected.to_csv("/Users/akibzaman/Codes/SleepBoost/Data/alldataretest/selected/selected_data_fs.csv")
# # data_selected.to_csv("/Users/akibzaman/Codes/SleepBoost/Data/alldataretest/selected/selected_data_train.csv")
# data_selected.to_csv("/Users/akibzaman/Codes/SleepBoost/Data/alldataretest/selected/selected_data_test.csv")
