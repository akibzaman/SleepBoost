# from sklearn.ensemble import RandomForestClassifier
# #from boruta import BorutaPy
import numpy as np
import pandas as pd
# from sklearn import datasets
# from yellowbrick.target import FeatureCorrelation
from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder
# from matplotlib import pyplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
#import the modules
import os
import glob
#read the path
file_list = glob.glob('alldatacsv/*.{}'.format('csv'))
nFiles = len(file_list)
data = pd.DataFrame()
#append all files together
for file in file_list:
    df_temp = pd.read_csv(file)
    data = pd.concat([data, df_temp], ignore_index=True)#data.append(df_temp, ignore_index=True)

print(data.shape)

# # data = pd.read_csv("alldatacsv/sample04.csv")
labelencoder = LabelEncoder()
data['class']=labelencoder.fit_transform(data['class'])
X, y = data.loc[:,"mean":"D19"], data['class']
# nunique = data.nunique()
# cols_to_drop = nunique[nunique == 1].index
# data.drop(cols_to_drop, axis=1)
feature_names = X.columns.values.tolist()
test=y
#########Normalization:
# scaler=MinMaxScaler()
# #scaler=StandardScaler()
# scaler.fit(X)
# X=scaler.transform(X)



#######################BORUTAPY-FEATURE-RANKING##########################

# #model=xgb.XGBClassifier()
# model = RandomForestClassifier(n_estimators = 100, random_state=30)

# # define Boruta feature selection method
# feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)

# # find all relevant features
# feat_selector.fit(X, y)

# # check selected features
# print(feat_selector.support_)

# # check ranking of features
# print(feat_selector.ranking_)

# # zip feature names, ranks, and decisions
# feature_ranks = list(zip(feature_names,
#                          feat_selector.ranking_,
#                          feat_selector.support_))
# ranked_data = pd.DataFrame (feature_ranks, columns = ['feature_name','rank','support'])
# # print the results
# for feat in feature_ranks:
#     print('Feature: {:<30} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))


# # Create a list of the feature names
# features = np.array(data['avg_red'])

##### Instantiate the visualizer
# visualizer = FeatureCorrelation(labels=None)

# visualizer.fit(X, y)        # Fit the data to the visualizer
# visualizer.show()           # Finalize and render the figure



##################################FEATURE RANKING########################

# ######################  ANOVA f-test Feature Selection  ##############
# fs = SelectKBest(score_func=f_classif, k='all')
# fs.fit(X, test)
# # what are scores for the features
# for i in range(len(fs.scores_)):
#     print('Feature %d: %f' % (i, fs.scores_[i]))
#
# # what are scores for the features
# selected_feature = []
# for i in range(len(fs.pvalues_)):
#     if(fs.pvalues_[i]<0.05):
#         selected_feature.append(fs.feature_names_in_[i])
#         print('Feature %d : %f' % (i,fs.pvalues_[i]),fs.feature_names_in_[i])


######################  Mutual Information  #####################
fs = SelectKBest(score_func=mutual_info_classif, k='all')
fs.fit(X, test)
# what are scores for the features
sum = 0
for i in range(len(fs.scores_)):
    sum = sum + fs.scores_[i]
    print('Feature %d: %f' % (i, fs.scores_[i]))

mean = sum /40
print("Mean: {0}".format(mean))
# what are scores for the features
selected_feature = []
for i in range(len(fs.scores_)):
    if(fs.scores_[i]>0.1):
        selected_feature.append(fs.feature_names_in_[i])
        print('Feature %d %s: %f' % (i,fs.feature_names_in_[i],fs.scores_[i]))

# #print(selected_feature)
data_selected = data[selected_feature]

####### IMPORT AS CSV (SELECTED DATA)

y = data['class']
data_frame=[data_selected, y]
data_selected=pd.concat(data_frame, axis=1)
data_selected.to_csv("alldatacsv/selected/selected_data_1.csv")

# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()



# ########## Multi-Colinearity Test #########
#
# vif_data = pd.DataFrame()
# vif_data["feature"] = data_selected.columns
# vif_data["VIF"] = [variance_inflation_factor(data_selected.values, i)
#                    for i in range(len(data_selected.columns))]
#
# vif_data = vif_data[vif_data['VIF'] <= 2.5]
# #vif_data = vif_data[vif_data['VIF'] <= 10]





# # ##Mutual-Information Classification

# visualizer = FeatureCorrelation(
#     method='mutual_info-classification', feature_names=None, sort=True)

# visualizer.fit(X, y)     # Fit the data to the visualizer
# visualizer.show()


# from yellowbrick.features import JointPlotVisualizer
# visualizer = JointPlotVisualizer(columns="cement")

# visualizer.fit_transform(X, y)        # Fit and transform the data
# visualizer.show()


# from yellowbrick.features import Rank1D
# visualizer = Rank1D(algorithm='shapiro')

# visualizer.fit(X, y)           # Fit the data to the visualizer
# a=visualizer.transform(X)        # Transform the data
# visualizer.show()
















#############https://www.scikit-yb.org/en/latest/api/target/feature_correlation.html