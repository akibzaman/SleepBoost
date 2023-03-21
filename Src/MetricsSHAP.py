import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from numpy import interp
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from collections import Counter
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d
import shap


feature_names=['Actual', 'Predicted']
actual_class = ['W','N1', 'N2','N3','REM']

###############		Raw Data	#################

# trainDataFiles = glob.glob('trainData/*.{}'.format('csv'))
# testDataFiles = glob.glob('testData/*.{}'.format('csv'))
# trainData = pd.DataFrame()
# testData = pd.DataFrame()
# #append all files together
# for file in trainDataFiles:
#     df_temp = pd.read_csv(file)
#     trainData = trainData.append(df_temp, ignore_index=True)
#
# for file in testDataFiles:
# 	df_temp = pd.read_csv(file)
# 	testData = testData.append(df_temp, ignore_index=True)
#
# print(trainData.shape)
# print(testData.shape)

# labelencoder = LabelEncoder()
# data['class']=labelencoder.fit_transform(data['class'])
# X_train, y_train = trainData.loc[:, "mean":"D19"], trainData['class']
# X_test, y_test = testData.loc[:, "mean":"D19"], testData['class']

################# Selected Data	###############


############## Mean ###############
# data = pd.read_csv("alldatacsv/selected/selected_data.csv")
# X, y = data.loc[:,"std":"D18"], data['class']

# ############# 0.1 ################
data = pd.read_csv("alldatacsv/selected/selected_data_1.csv")
X, y = data.loc[:,"std":"D19"], data['class']
#
# ############# 0.2 ################
# data = pd.read_csv("alldatacsv/selected/selected_data_2.csv")
# X, y = data.loc[:,"std":"D19"], data['class']
#
# ############# 0.3 ################
# data = pd.read_csv("alldatacsv/selected/selected_data_3.csv")
# X, y = data.loc[:,"std":"D6"], data['class']


X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=32) #stratify=y
X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=32)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)

# scaler=MinMaxScaler()
# scaler.fit(X_train)
# X_train=scaler.transform(X_train)
# scaler.fit(X_test)
# X_test=scaler.transform(X_test)

def showDistribution(y):
	counter = Counter(y)
	for k,v in counter.items():
		per = v / len(y) * 100
		print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
	pyplot.bar(counter.keys(), counter.values())
	pyplot.show()

showDistribution(y_train)
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)
# showDistribution(y_train)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

######## HYPERPARAMETER TUNING ##############
# from sklearn.model_selection import RandomizedSearchCV
#
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]


# random_grid = {'n_estimators': n_estimators,
# 			   'max_features': max_features,
# 			   'max_depth': max_depth,
# 			   'min_samples_split': min_samples_split,
# 			   'min_samples_leaf': min_samples_leaf,
# 			   'bootstrap': bootstrap}
# print(random_grid)
#
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3,
# 							   verbose=2, random_state=42, n_jobs = -1)
# try:
# 	rf_random.fit(X_train,y_train)
# 	print(rf_random.best_params_)
# except:
# 	print("An exception occurred")
# best_random = rf_random.best_estimator_
#
# y_pred_train=best_random.predict(X_train)
# y_pred_test = best_random.predict(X_test)
# print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y_pred_test))
# pickle.dump(model_RF, open('trained_model\model_RF.pkl','wb'))

# ### Support Vector Machine (SVM)
# print("SVM ------------------------------->")
# model_SVM_RBF = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
# # model_SVM_RBF = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
# model_SVM_RBF.fit(X_train,y_train)
# y_pred_train=model_SVM_RBF.predict(X_train)
# y_pred_test_SVM = model_SVM_RBF.predict(X_test)
# print(metrics.classification_report(y_test, y_pred_test_SVM))
# # pickle.dump(model_RF, open('trained_model\model_RF.pkl','wb'))
#
# ##Confusion Matrix - verify accuracy of each class
# cm = confusion_matrix(y_test, y_pred_test_SVM)
# # print(cm)
# test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
# sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
# 			square=True,
# 			annot=True,
# 			fmt='d',cmap='BuGn', annot_kws={"size": 16})
# pyplot.savefig('EvaluationFigures/LATEST/SVM.jpg',dpi=300)
# pyplot.show()
#
#
# ##### Random Forest #####
# print("RF ------------------------------->")
# model_RF = RandomForestClassifier( n_estimators=200
# 								   ,
# 								   criterion="entropy", #gini
# 								   max_depth=None,
# 								   min_samples_split=9,
# 								   min_samples_leaf=1,
# 								   min_weight_fraction_leaf=0.0,
# 								   max_features="auto",
# 								   max_leaf_nodes=None,
# 								   min_impurity_decrease=0.0,
# 								   bootstrap=True,
# 								   oob_score=False,
# 								   n_jobs=None,
# 								   random_state=None,
# 								   verbose=0,
# 								   warm_start=False,
# 								   class_weight=None,
# 								   ccp_alpha=0.0,
# 								   max_samples=None,
# 								   )
# model_RF.fit(X_train,y_train)
# y_pred_train=model_RF.predict(X_train)
# y_pred_test_RF = model_RF.predict(X_test)
# # print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y_pred_test_RF))
# # pickle.dump(model_RF, open('trained_model\model_RF.pkl','wb'))
#
# ##Confusion Matrix - verify accuracy of each class
# cm = confusion_matrix(y_test, y_pred_test_RF)
# # print(cm)
# test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
# sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
# 			square=True,
# 			annot=True,
# 			fmt='d',cmap='BuGn', annot_kws={"size": 16})
# pyplot.savefig('EvaluationFigures/LATEST/RF.jpg',dpi=300)
# pyplot.show()
#
#
# ##### CatBoost #####
# print("CatBoost Train Starts------------------------------->")
# model_CBC = CatBoostClassifier()
# model_CBC = model_CBC.fit(X_train,y_train)
# y_pred_train= model_CBC.predict(X_train)
# y_pred_test_CAT = model_CBC.predict(X_test)
# print("CatBoost------------------------------->")
# # print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y_pred_test_CAT))
#
# ##Confusion Matrix - verify accuracy of each class
# cm = confusion_matrix(y_test, y_pred_test_CAT)
# # print(cm)
# test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
# sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
# 			square=True,
# 			annot=True,
# 			fmt='d',cmap='BuGn', annot_kws={"size": 16})
# pyplot.savefig('EvaluationFigures/LATEST/CatBoost.jpg',dpi=300)
# pyplot.show()
#
#
# ##### AdaBoost #####
# print("ADB------------------------------->")
# model_adb = AdaBoostClassifier(n_estimators=100,learning_rate=0.5,
# 							   random_state=0)
# model_adb = model_adb.fit(X_train,y_train)
# y_pred_train= model_adb.predict(X_train)
# y_pred_test_ADB = model_adb.predict(X_test)
# # print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y_pred_test_ADB))
#
# ##Confusion Matrix - verify accuracy of each class
# cm = confusion_matrix(y_test, y_pred_test_ADB)
# # print(cm)
# test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
#
# # pyplot.figure(figsize = (10,7))
# sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
# 			square=True,
# 			annot=True,
# 			fmt='d',cmap='BuGn', annot_kws={"size": 16})
# pyplot.savefig('EvaluationFigures/LATEST/AdaBoost.jpg',dpi=300)
# pyplot.show()
#
##### LGBoost #####
print("LGBM------------------------------->")
lgb_model=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=25,
							 min_data_in_leaf=6,
							 learning_rate=0.4, n_estimators=200,
							 objective='multiclass', class_weight='balanced',
							 min_split_gain=0.0,
							 min_child_weight=0.03, min_child_samples=20,
							 subsample=1.0,
							 subsample_freq=500, colsample_bytree=1.0,
							 reg_lambda=0.65, random_state=0,num_classes=3)
lgb_model= lgb_model.fit(X_train,y_train)
y_pred_train=lgb_model.predict(X_train)
y_pred_test_LGB = lgb_model.predict(X_test)
print(metrics.classification_report(y_test, y_pred_test_LGB))
#
# ##Confusion Matrix - verify accuracy of each class
# cm = confusion_matrix(y_test, y_pred_test_LGB)
# # print(cm)
# test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
# sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
# 			square=True,
# 			annot=True,
# 			fmt='d',cmap='BuGn', annot_kws={"size": 16})
# pyplot.savefig('EvaluationFigures/LATEST/LGBoost.jpg',dpi=300)
# pyplot.show()



# #### SleepBoost
#
# ###############     ENSEMBLE     ########################
#
# # ########CatBoost Model
# print("CatBoost------------------------------->")
# model_Cat = CatBoostClassifier()
# model_Cat.fit(X_train, y_train)
# C1_knn_pred = model_Cat.predict(X_valid)
# C1_knn_pred = pd.DataFrame(C1_knn_pred, columns=['C1KNN'])
# C1_knn_pred = C1_knn_pred.reset_index()
# C1_knn_pred = C1_knn_pred.drop(['index'], axis=1)
#
# ###LGBM
# print("LGBM------------------------------->")
# lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=25,
# 							   min_data_in_leaf=6,
# 							   learning_rate=0.4, n_estimators=200,
# 							   objective='multiclass', class_weight='balanced',
# 							   min_split_gain=0.0,
# 							   min_child_weight=0.03, min_child_samples=20,
# 							   subsample=1.0,
# 							   subsample_freq=500, colsample_bytree=1.0,
# 							   reg_lambda=0.65, random_state=0, num_classes=3)
# lgb_model.fit(X_train, y_train)
# C1_lgbm_pred = lgb_model.predict(X_valid)
# C1_lgbm_pred = pd.DataFrame(C1_lgbm_pred, columns=['C2LGBM'])
# C1_lgbm_pred = C1_lgbm_pred.reset_index()
# C1_lgbm_pred = C1_lgbm_pred.drop(['index'], axis=1)
#
# # RandonForest
# print("--------------- RF ------------------------------->")
# model_RF = RandomForestClassifier(n_estimators=200,
# 								  criterion="entropy",  # gini
# 								  max_depth=None,
# 								  min_samples_split=9,
# 								  min_samples_leaf=1,
# 								  min_weight_fraction_leaf=0.0,
# 								  max_features="auto",
# 								  max_leaf_nodes=None,
# 								  min_impurity_decrease=0.0,
# 								  bootstrap=True,
# 								  oob_score=False,
# 								  n_jobs=None,
# 								  random_state=None,
# 								  verbose=0,
# 								  warm_start=False,
# 								  class_weight=None,
# 								  ccp_alpha=0.0,
# 								  max_samples=None, )
# model_RF.fit(X_train, y_train)
# C1_rf_pred = model_RF.predict(X_valid)
# C1_rf_pred = pd.DataFrame(C1_rf_pred, columns=['C3RF'])
# C1_rf_pred = C1_rf_pred.reset_index()
# C1_rf_pred = C1_rf_pred.drop(['index'], axis=1)
#
# actual = pd.DataFrame(y_valid, columns=['class'])
# actual = actual.reset_index()
# actual = actual.drop(['index'], axis=1)
#
# C_Final = pd.concat([C1_knn_pred, C1_lgbm_pred, C1_rf_pred, actual], axis=1)
#
# VALUE1 = 0.33
# VALUE2 = 0.67
#
#
# def adaptiveWeightAllocation(df):
# 	weight = [0.0] * 3
# 	for i in range(len(df)):
# 		instance = df.iloc[i]
# 		correct = 0
# 		# print(instance[3])
# 		flag = [0] * 3
# 		if (instance[0] == instance[3]):
# 			correct += 1
# 			flag[0] = 1
# 		if (instance[1] == instance[3]):
# 			correct += 1
# 			flag[1] = 1
# 		if (instance[2] == instance[3]):
# 			correct += 1
# 			flag[2] = 1
# 		# print(correct)
# 		# print(flag)
# 		if (correct > 0 and correct <= 3):
# 			# print("inside")
# 			if (correct == 2):
# 				for k in range(len(flag)):
# 					if (flag[k] == 1):
# 						weight[k] += VALUE1
# 			# else:
# 			# 	weight[k]-=(VALUE2/3)
# 			if (correct == 1):
# 				for k in range(len(flag)):
# 					if (flag[k] == 1):
# 						weight[k] += VALUE2
# 	# else:
# 	# 	weight[k]-=(VALUE1/3)
# 	# print(weight)
# 	# sum = weight[1]+weight[0]+weight[2]
# 	# weight[0]/= sum
# 	# weight[1]/= sum
# 	# weight[2]/= sum
# 	return weight
#
#
# weight = adaptiveWeightAllocation(C_Final)
# print(weight)
#
# ######TEST
#
# ######KNN
# C1_cat_test_pred = model_Cat.predict(X_test)
# C1_cat_test_pred = pd.DataFrame(C1_cat_test_pred, columns=['C1KNN'])
# C1_cat_test_pred = C1_cat_test_pred.reset_index()
# C1_cat_test_pred = C1_cat_test_pred.drop(['index'], axis=1)
#
# ###LGBM
# C1_lgbm_test_pred = lgb_model.predict(X_test)
# C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=['C2LGBM'])
# C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
# C1_lgbm_test_pred = C1_lgbm_test_pred.drop(['index'], axis=1)
#
# # RandonForest
# C1_rf_test_pred = model_RF.predict(X_test)
# C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=['C3RF'])
# C1_rf_test_pred = C1_rf_test_pred.reset_index()
# C1_rf_test_pred = C1_rf_test_pred.drop(['index'], axis=1)
#
# C_test_Final = pd.concat([C1_cat_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)
#
#
# def SleepBoost(df, weight):
# 	prdicted_classes = [0] * len(df)
# 	for i in range(len(df)):
# 		matrix = [[0.0, 0.0, 0.0, 0.0, 0.0]] * 3  # Number of Classes
# 		# print(matrix)
# 		# break
# 		instance = df.iloc[i]
# 		for j in range(len(instance)):
# 			if (instance[j] == 0):
# 				matrix[j] = [weight[j], 0.0, 0.0, 0.0, 0.0]
# 			elif (instance[j] == 1):
# 				matrix[j] = [0.0, weight[j], 0.0, 0.0, 0.0]
# 			elif (instance[j] == 2):
# 				matrix[j] = [0.0, 0.0, weight[j], 0.0, 0.0]
# 			elif (instance[j] == 3):
# 				matrix[j] = [0.0, 0.0, 0.0, weight[j], 0.0]
# 			else:
# 				matrix[j] = [0.0, 0.0, 0.0, 0.0, weight[j]]
# 		# print(matrix)
# 		sum = [0.0] * 5
# 		for k in range(5):
# 			sum[k] = matrix[0][k] + matrix[1][k] + matrix[2][k]
# 		max_value = max(sum)
# 		index = sum.index(max_value)
# 		prdicted_classes[i] = index
# 	# print(sum)
# 	# break
# 	return prdicted_classes
#
# sleepBoost_test_pred = SleepBoost(C_test_Final, weight)
# cm = confusion_matrix(y_test, sleepBoost_test_pred)
# test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
# sns.heatmap(test_cm, robust=True, linewidths=0.1, linecolor='grey',
# 			square=True,
# 			annot=True,
# 			fmt='d', cmap='BuGn', annot_kws={"size": 16})
# pyplot.savefig('EvaluationFigures/LATEST/SleepBoost.jpg',dpi=300)
# pyplot.show()

# print("SVM")
# print(metrics.classification_report(y_test, y_pred_test_SVM))
# print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_SVM)))
# print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_SVM, average='macro')))
# print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_SVM, average='macro')))
# print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_SVM, average='macro')))
# print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_SVM)))
# print("\n")
#
#
# print("AdaBoost")
# print(metrics.classification_report(y_test, y_pred_test_ADB))
# print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_ADB)))
# print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_ADB, average='macro')))
# print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_ADB, average='macro')))
# print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_ADB, average='macro')))
# print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_ADB)))
# print("\n")
#
# print("Random Forest")
# print(metrics.classification_report(y_test, y_pred_test_RF))
# print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_RF)))
# print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_RF, average='macro')))
# print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_RF, average='macro')))
# print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_RF, average='macro')))
# print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_RF)))
# print("\n")
#
# print("CatBoost")
# print(metrics.classification_report(y_test, y_pred_test_CAT))
# print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_CAT)))
# print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_CAT, average='macro')))
# print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_CAT, average='macro')))
# print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_CAT, average='macro')))
# print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_CAT)))
# print("\n")
#
# print("LGBoost")
# print(metrics.classification_report(y_test, y_pred_test_LGB))
# print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_LGB)))
# print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_LGB, average='macro')))
# print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_LGB, average='macro')))
# print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_LGB, average='macro')))
# print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_LGB)))
# print("\n")

# print("SLEEPBOOST")
# print(metrics.classification_report(y_test, sleepBoost_test_pred))
# print("Accuracy: {0}".format(metrics.accuracy_score(y_test, sleepBoost_test_pred)))
# print("Precision: {0}".format(metrics.precision_score(y_test, sleepBoost_test_pred, average='macro')))
# print("Recall: {0}".format(metrics.recall_score(y_test, sleepBoost_test_pred, average='macro')))
# print("F1: {0}".format(metrics.f1_score(y_test, sleepBoost_test_pred, average='macro')))
# print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, sleepBoost_test_pred)))
# print("\n")


explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X)

shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])
