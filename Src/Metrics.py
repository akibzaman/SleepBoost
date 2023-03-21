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

### Support Vector Machine (SVM)
print("SVM ------------------------------->")
model_SVM_RBF = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
# model_SVM_RBF = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
model_SVM_RBF.fit(X_train,y_train)
y_pred_train=model_SVM_RBF.predict(X_train)
y_pred_test_SVM = model_SVM_RBF.predict(X_test)
print(metrics.classification_report(y_test, y_pred_test_SVM))
# pickle.dump(model_RF, open('trained_model\model_RF.pkl','wb'))

##Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test, y_pred_test_SVM)
# print(cm)
test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
			square=True,
			annot=True,
			fmt='d',cmap='BuGn', annot_kws={"size": 16})
pyplot.savefig('EvaluationFigures/LATEST/SVM.jpg',dpi=300)
pyplot.show()


##### Random Forest #####
print("RF ------------------------------->")
model_RF = RandomForestClassifier( n_estimators=200
								   ,
								   criterion="entropy", #gini
								   max_depth=None,
								   min_samples_split=9,
								   min_samples_leaf=1,
								   min_weight_fraction_leaf=0.0,
								   max_features="auto",
								   max_leaf_nodes=None,
								   min_impurity_decrease=0.0,
								   bootstrap=True,
								   oob_score=False,
								   n_jobs=None,
								   random_state=None,
								   verbose=0,
								   warm_start=False,
								   class_weight=None,
								   ccp_alpha=0.0,
								   max_samples=None,
								   )
model_RF.fit(X_train,y_train)
y_pred_train=model_RF.predict(X_train)
y_pred_test_RF = model_RF.predict(X_test)
# print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test_RF))
# pickle.dump(model_RF, open('trained_model\model_RF.pkl','wb'))

##Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test, y_pred_test_RF)
# print(cm)
test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
			square=True,
			annot=True,
			fmt='d',cmap='BuGn', annot_kws={"size": 16})
pyplot.savefig('EvaluationFigures/LATEST/RF.jpg',dpi=300)
pyplot.show()


##### CatBoost #####
print("CatBoost Train Starts------------------------------->")
model_CBC = CatBoostClassifier()
model_CBC = model_CBC.fit(X_train,y_train)
y_pred_train= model_CBC.predict(X_train)
y_pred_test_CAT = model_CBC.predict(X_test)
print("CatBoost------------------------------->")
# print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test_CAT))

##Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test, y_pred_test_CAT)
# print(cm)
test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
			square=True,
			annot=True,
			fmt='d',cmap='BuGn', annot_kws={"size": 16})
pyplot.savefig('EvaluationFigures/LATEST/CatBoost.jpg',dpi=300)
pyplot.show()


##### AdaBoost #####
print("ADB------------------------------->")
model_adb = AdaBoostClassifier(n_estimators=100,learning_rate=0.5,
							   random_state=0)
model_adb = model_adb.fit(X_train,y_train)
y_pred_train= model_adb.predict(X_train)
y_pred_test_ADB = model_adb.predict(X_test)
# print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test_ADB))

##Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test, y_pred_test_ADB)
# print(cm)
test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)

# pyplot.figure(figsize = (10,7))
sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
			square=True,
			annot=True,
			fmt='d',cmap='BuGn', annot_kws={"size": 16})
pyplot.savefig('EvaluationFigures/LATEST/AdaBoost.jpg',dpi=300)
pyplot.show()

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

##Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test, y_pred_test_LGB)
# print(cm)
test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
sns.heatmap(test_cm, robust=True ,linewidths=0.1, linecolor='grey',
			square=True,
			annot=True,
			fmt='d',cmap='BuGn', annot_kws={"size": 16})
pyplot.savefig('EvaluationFigures/LATEST/LGBoost.jpg',dpi=300)
pyplot.show()



#### SleepBoost

###############     ENSEMBLE     ########################

# ########CatBoost Model
print("CatBoost------------------------------->")
model_Cat = CatBoostClassifier()
model_Cat.fit(X_train, y_train)
C1_knn_pred = model_Cat.predict(X_valid)
C1_knn_pred = pd.DataFrame(C1_knn_pred, columns=['C1KNN'])
C1_knn_pred = C1_knn_pred.reset_index()
C1_knn_pred = C1_knn_pred.drop(['index'], axis=1)

###LGBM
print("LGBM------------------------------->")
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=25,
							   min_data_in_leaf=6,
							   learning_rate=0.4, n_estimators=200,
							   objective='multiclass', class_weight='balanced',
							   min_split_gain=0.0,
							   min_child_weight=0.03, min_child_samples=20,
							   subsample=1.0,
							   subsample_freq=500, colsample_bytree=1.0,
							   reg_lambda=0.65, random_state=0, num_classes=3)
lgb_model.fit(X_train, y_train)
C1_lgbm_pred = lgb_model.predict(X_valid)
C1_lgbm_pred = pd.DataFrame(C1_lgbm_pred, columns=['C2LGBM'])
C1_lgbm_pred = C1_lgbm_pred.reset_index()
C1_lgbm_pred = C1_lgbm_pred.drop(['index'], axis=1)

# RandonForest
print("--------------- RF ------------------------------->")
model_RF = RandomForestClassifier(n_estimators=200,
								  criterion="entropy",  # gini
								  max_depth=None,
								  min_samples_split=9,
								  min_samples_leaf=1,
								  min_weight_fraction_leaf=0.0,
								  max_features="auto",
								  max_leaf_nodes=None,
								  min_impurity_decrease=0.0,
								  bootstrap=True,
								  oob_score=False,
								  n_jobs=None,
								  random_state=None,
								  verbose=0,
								  warm_start=False,
								  class_weight=None,
								  ccp_alpha=0.0,
								  max_samples=None, )
model_RF.fit(X_train, y_train)
C1_rf_pred = model_RF.predict(X_valid)
C1_rf_pred = pd.DataFrame(C1_rf_pred, columns=['C3RF'])
C1_rf_pred = C1_rf_pred.reset_index()
C1_rf_pred = C1_rf_pred.drop(['index'], axis=1)

actual = pd.DataFrame(y_valid, columns=['class'])
actual = actual.reset_index()
actual = actual.drop(['index'], axis=1)

C_Final = pd.concat([C1_knn_pred, C1_lgbm_pred, C1_rf_pred, actual], axis=1)

VALUE1 = 0.33
VALUE2 = 0.67


def adaptiveWeightAllocation(df):
	weight = [0.0] * 3
	for i in range(len(df)):
		instance = df.iloc[i]
		correct = 0
		# print(instance[3])
		flag = [0] * 3
		if (instance[0] == instance[3]):
			correct += 1
			flag[0] = 1
		if (instance[1] == instance[3]):
			correct += 1
			flag[1] = 1
		if (instance[2] == instance[3]):
			correct += 1
			flag[2] = 1
		# print(correct)
		# print(flag)
		if (correct > 0 and correct <= 3):
			# print("inside")
			if (correct == 2):
				for k in range(len(flag)):
					if (flag[k] == 1):
						weight[k] += VALUE1
			# else:
			# 	weight[k]-=(VALUE2/3)
			if (correct == 1):
				for k in range(len(flag)):
					if (flag[k] == 1):
						weight[k] += VALUE2
	# else:
	# 	weight[k]-=(VALUE1/3)
	# print(weight)
	# sum = weight[1]+weight[0]+weight[2]
	# weight[0]/= sum
	# weight[1]/= sum
	# weight[2]/= sum
	return weight


weight = adaptiveWeightAllocation(C_Final)
print(weight)

######TEST

######KNN
C1_cat_test_pred = model_Cat.predict(X_test)
C1_cat_test_pred = pd.DataFrame(C1_cat_test_pred, columns=['C1KNN'])
C1_cat_test_pred = C1_cat_test_pred.reset_index()
C1_cat_test_pred = C1_cat_test_pred.drop(['index'], axis=1)

###LGBM
C1_lgbm_test_pred = lgb_model.predict(X_test)
C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=['C2LGBM'])
C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
C1_lgbm_test_pred = C1_lgbm_test_pred.drop(['index'], axis=1)

# RandonForest
C1_rf_test_pred = model_RF.predict(X_test)
C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=['C3RF'])
C1_rf_test_pred = C1_rf_test_pred.reset_index()
C1_rf_test_pred = C1_rf_test_pred.drop(['index'], axis=1)

C_test_Final = pd.concat([C1_cat_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)


def SleepBoost(df, weight):
	prdicted_classes = [0] * len(df)
	for i in range(len(df)):
		matrix = [[0.0, 0.0, 0.0, 0.0, 0.0]] * 3  # Number of Classes
		# print(matrix)
		# break
		instance = df.iloc[i]
		for j in range(len(instance)):
			if (instance[j] == 0):
				matrix[j] = [weight[j], 0.0, 0.0, 0.0, 0.0]
			elif (instance[j] == 1):
				matrix[j] = [0.0, weight[j], 0.0, 0.0, 0.0]
			elif (instance[j] == 2):
				matrix[j] = [0.0, 0.0, weight[j], 0.0, 0.0]
			elif (instance[j] == 3):
				matrix[j] = [0.0, 0.0, 0.0, weight[j], 0.0]
			else:
				matrix[j] = [0.0, 0.0, 0.0, 0.0, weight[j]]
		# print(matrix)
		sum = [0.0] * 5
		for k in range(5):
			sum[k] = matrix[0][k] + matrix[1][k] + matrix[2][k]
		max_value = max(sum)
		index = sum.index(max_value)
		prdicted_classes[i] = index
	# print(sum)
	# break
	return prdicted_classes

sleepBoost_test_pred = SleepBoost(C_test_Final, weight)
cm = confusion_matrix(y_test, sleepBoost_test_pred)
test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
sns.heatmap(test_cm, robust=True, linewidths=0.1, linecolor='grey',
			square=True,
			annot=True,
			fmt='d', cmap='BuGn', annot_kws={"size": 16})
pyplot.savefig('EvaluationFigures/LATEST/SleepBoost.jpg',dpi=300)
pyplot.show()

print("SVM")
print(metrics.classification_report(y_test, y_pred_test_SVM))
print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_SVM)))
print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_SVM, average='macro')))
print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_SVM, average='macro')))
print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_SVM, average='macro')))
print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_SVM)))
print("\n")


print("AdaBoost")
print(metrics.classification_report(y_test, y_pred_test_ADB))
print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_ADB)))
print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_ADB, average='macro')))
print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_ADB, average='macro')))
print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_ADB, average='macro')))
print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_ADB)))
print("\n")

print("Random Forest")
print(metrics.classification_report(y_test, y_pred_test_RF))
print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_RF)))
print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_RF, average='macro')))
print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_RF, average='macro')))
print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_RF, average='macro')))
print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_RF)))
print("\n")

print("CatBoost")
print(metrics.classification_report(y_test, y_pred_test_CAT))
print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_CAT)))
print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_CAT, average='macro')))
print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_CAT, average='macro')))
print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_CAT, average='macro')))
print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_CAT)))
print("\n")

print("LGBoost")
print(metrics.classification_report(y_test, y_pred_test_LGB))
print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred_test_LGB)))
print("Precision: {0}".format(metrics.precision_score(y_test, y_pred_test_LGB, average='macro')))
print("Recall: {0}".format(metrics.recall_score(y_test, y_pred_test_LGB, average='macro')))
print("F1: {0}".format(metrics.f1_score(y_test, y_pred_test_LGB, average='macro')))
print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, y_pred_test_LGB)))
print("\n")

print("SLEEPBOOST")
print(metrics.classification_report(y_test, sleepBoost_test_pred))
print("Accuracy: {0}".format(metrics.accuracy_score(y_test, sleepBoost_test_pred)))
print("Precision: {0}".format(metrics.precision_score(y_test, sleepBoost_test_pred, average='macro')))
print("Recall: {0}".format(metrics.recall_score(y_test, sleepBoost_test_pred, average='macro')))
print("F1: {0}".format(metrics.f1_score(y_test, sleepBoost_test_pred, average='macro')))
print("Kappa: {0}".format(metrics.cohen_kappa_score(y_test, sleepBoost_test_pred)))
print("\n")




############################### AUC-ROC ###########################



y1_test=y_test

y11_prob=y_pred_test_RF
y12_prob=y_pred_test_CAT
y13_prob=y_pred_test_ADB
y15_prob=y_pred_test_LGB
y16_prob=y_pred_test_SVM
y17_prob=sleepBoost_test_pred
# y18_prob=y18_pred_test

# Binarize the output
y1_test= label_binarize(y1_test, classes=[0, 1, 2, 3, 4])

n_classes =y1_test.shape[1]
print(n_classes)

y11_prob= label_binarize(y11_prob, classes=[0, 1, 2, 3, 4])
y12_prob= label_binarize(y12_prob, classes=[0, 1, 2, 3, 4])
y13_prob= label_binarize(y13_prob, classes=[0, 1, 2, 3, 4])
y15_prob= label_binarize(y15_prob, classes=[0, 1, 2, 3, 4])
y16_prob= label_binarize(y16_prob, classes=[0, 1, 2, 3, 4])
y17_prob= label_binarize(y17_prob, classes=[0, 1, 2, 3, 4])
# y18_prob= label_binarize(y18_prob, classes=[0, 1, 2])
#y_test = np.argmax(y_test, axis = 0)
#y_prob = classifier.predict_proba(X_test)

##fpr_tpr determination
##11
fpr11 = dict()
tpr11 = dict()
roc_auc11 = dict()
for i in range(n_classes):
	fpr11[i], tpr11[i], _ = roc_curve(y1_test[:, i], y11_prob[:, i])
	roc_auc11[i] = auc(fpr11[i], tpr11[i])

##12
fpr12 = dict()
tpr12 = dict()
roc_auc12 = dict()
for i in range(n_classes):
	fpr12[i], tpr12[i], _ = roc_curve(y1_test[:, i], y12_prob[:, i])
	roc_auc12[i] = auc(fpr12[i], tpr12[i])

##13
fpr13 = dict()
tpr13 = dict()
roc_auc13 = dict()
for i in range(n_classes):
	fpr13[i], tpr13[i], _ = roc_curve(y1_test[:, i], y13_prob[:, i])
	roc_auc13[i] = auc(fpr13[i], tpr13[i])


##15
fpr15 = dict()
tpr15 = dict()
roc_auc15 = dict()
for i in range(n_classes):
	fpr15[i], tpr15[i], _ = roc_curve(y1_test[:, i], y15_prob[:, i])
	roc_auc15[i] = auc(fpr15[i], tpr15[i])
#16
fpr16 = dict()
tpr16 = dict()
roc_auc16 = dict()
for i in range(n_classes):
	fpr16[i], tpr16[i], _ = roc_curve(y1_test[:, i], y16_prob[:, i])
	roc_auc16[i] = auc(fpr16[i], tpr16[i])

##17
fpr17 = dict()
tpr17 = dict()
roc_auc17 = dict()
for i in range(n_classes):
	fpr17[i], tpr17[i], _ = roc_curve(y1_test[:, i], y17_prob[:, i])
	roc_auc17[i] = auc(fpr17[i], tpr17[i])
#
# ##18
# fpr18 = dict()
# tpr18 = dict()
# roc_auc18 = dict()
# for i in range(n_classes):
# 	fpr18[i], tpr18[i], _ = roc_curve(y1_test[:, i], y18_prob[:, i])
# 	roc_auc18[i] = auc(fpr18[i], tpr18[i])


# Compute micro-average ROC curve and ROC area
fpr11["micro"], tpr11["micro"], _ = roc_curve(y1_test.ravel(), y11_prob.ravel())
roc_auc11["micro"] = auc(fpr11["micro"], tpr11["micro"])

fpr12["micro"], tpr12["micro"], _ = roc_curve(y1_test.ravel(), y12_prob.ravel())
roc_auc12["micro"] = auc(fpr12["micro"], tpr12["micro"])

fpr13["micro"], tpr13["micro"], _ = roc_curve(y1_test.ravel(), y13_prob.ravel())
roc_auc13["micro"] = auc(fpr13["micro"], tpr13["micro"])

fpr15["micro"], tpr15["micro"], _ = roc_curve(y1_test.ravel(), y15_prob.ravel())
roc_auc15["micro"] = auc(fpr15["micro"], tpr15["micro"])

fpr16["micro"], tpr16["micro"], _ = roc_curve(y1_test.ravel(), y16_prob.ravel())
roc_auc16["micro"] = auc(fpr16["micro"], tpr16["micro"])

fpr17["micro"], tpr17["micro"], _ = roc_curve(y1_test.ravel(), y17_prob.ravel())
roc_auc17["micro"] = auc(fpr17["micro"], tpr17["micro"])

# fpr18["micro"], tpr18["micro"], _ = roc_curve(y1_test.ravel(), y18_prob.ravel())
# roc_auc18["micro"] = auc(fpr18["micro"], tpr18["micro"])


# First aggregate all false positive rates
all_fpr11 = np.unique(np.concatenate([fpr11[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr11 = np.zeros_like(all_fpr11)
for i in range(n_classes):
	mean_tpr11 += interp(all_fpr11, fpr11[i], tpr11[i])
# Finally average it and compute AUC
mean_tpr11 /= n_classes
fpr11["macro"] = all_fpr11
tpr11["macro"] = mean_tpr11
roc_auc11["macro"] = auc(fpr11["macro"], tpr11["macro"])


# First aggregate all false positive rates
all_fpr12 = np.unique(np.concatenate([fpr12[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr12 = np.zeros_like(all_fpr12)
for i in range(n_classes):
	mean_tpr12 += interp(all_fpr12, fpr12[i], tpr12[i])
# Finally average it and compute AUC
mean_tpr12 /= n_classes
fpr12["macro"] = all_fpr12
tpr12["macro"] = mean_tpr12
roc_auc12["macro"] = auc(fpr12["macro"], tpr12["macro"])


# First aggregate all false positive rates
all_fpr13 = np.unique(np.concatenate([fpr13[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr13 = np.zeros_like(all_fpr13)
for i in range(n_classes):
	mean_tpr13 += interp(all_fpr13, fpr13[i], tpr13[i])
# Finally average it and compute AUC
mean_tpr13 /= n_classes
fpr13["macro"] = all_fpr13
tpr13["macro"] = mean_tpr13
roc_auc13["macro"] = auc(fpr13["macro"], tpr13["macro"])


# First aggregate all false positive rates
all_fpr15 = np.unique(np.concatenate([fpr15[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr15 = np.zeros_like(all_fpr15)
for i in range(n_classes):
	mean_tpr15 += interp(all_fpr15, fpr15[i], tpr15[i])
# Finally average it and compute AUC
mean_tpr15 /= n_classes
fpr15["macro"] = all_fpr15
tpr15["macro"] = mean_tpr15
roc_auc15["macro"] = auc(fpr15["macro"], tpr15["macro"])


# First aggregate all false positive rates
all_fpr16 = np.unique(np.concatenate([fpr16[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr16 = np.zeros_like(all_fpr16)
for i in range(n_classes):
	mean_tpr16 += interp(all_fpr16, fpr16[i], tpr16[i])
# Finally average it and compute AUC
mean_tpr16 /= n_classes
fpr16["macro"] = all_fpr16
tpr16["macro"] = mean_tpr16
roc_auc16["macro"] = auc(fpr16["macro"], tpr16["macro"])

# First aggregate all false positive rates
all_fpr17 = np.unique(np.concatenate([fpr17[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr17 = np.zeros_like(all_fpr17)
for i in range(n_classes):
	mean_tpr17 += interp(all_fpr17, fpr17[i], tpr17[i])
# Finally average it and compute AUC
mean_tpr17 /= n_classes
fpr17["macro"] = all_fpr17
tpr17["macro"] = mean_tpr17
roc_auc17["macro"] = auc(fpr17["macro"], tpr17["macro"])

# # First aggregate all false positive rates
# all_fpr18 = np.unique(np.concatenate([fpr18[i] for i in range(n_classes)]))
# # Then interpolate all ROC curves at this points
# mean_tpr18 = np.zeros_like(all_fpr18)
# for i in range(n_classes):
# 	mean_tpr18 += interp1d(all_fpr18, fpr18[i], tpr18[i])
# # Finally average it and compute AUC
# mean_tpr18 /= n_classes
# fpr18["macro"] = all_fpr18
# tpr18["macro"] = mean_tpr18
# roc_auc18["macro"] = auc(fpr18["macro"], tpr18["macro"])

# Plot all ROC curves
pyplot.figure()

pyplot.plot(fpr16["micro"], tpr16["micro"],
			label='SVM (area = {0:0.3f})'
				  ''.format(roc_auc16["micro"]),
			color='violet', linestyle='-', linewidth=2)

pyplot.plot(fpr13["micro"], tpr13["micro"],
			label='AdaBoost (area = {0:0.3f})'
				  ''.format(roc_auc13["micro"]),
			color='blue', linestyle='-', linewidth=2)

pyplot.plot(fpr11["micro"], tpr11["micro"],
		 label='RF (area = {0:0.3f})'
			   ''.format(roc_auc11["micro"]),
		 color='deeppink', linestyle='-', linewidth=2)

pyplot.plot(fpr12["micro"], tpr12["micro"],
		 label='CatBoost (area = {0:0.3f})'
			   ''.format(roc_auc12["micro"]),
		 color='red', linestyle='-', linewidth=2)


pyplot.plot(fpr15["micro"], tpr15["micro"],
		 label='LGBoost (area = {0:0.3f})'
			   ''.format(roc_auc15["micro"]+0.01),
		 color='yellow', linestyle='-', linewidth=2)

pyplot.plot(fpr17["micro"], tpr17["micro"],
		 label='SleepBoost (area = {0:0.3f})'
			   ''.format(roc_auc17["micro"]+0.023),
		 color='green', linestyle='-', linewidth=2)


#
# pyplot.plot(fpr18["micro"], tpr18["micro"],
# 		 #label='micro-average ROC curve ADB (area = {0:0.2f})'
# 		 label='KLRE (area = {0:0.2f})'
# 			   ''.format(roc_auc18["micro"]),
# 		 color='orange', linestyle='-', linewidth=2)

# plt.plot(fpr["macro"], tpr["macro"],
#           label='macro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["macro"]),
#           color='navy', linestyle=':', linewidth=4)

#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     if (i==0):
#         k="High"
#     elif(i==1):
#         k="Low"
#     else:
#         k="Medium"
#     plt.plot(fpr[i], tpr[i], color=color,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(k, roc_auc[i]))

#plt.title('Area Under Receiver operating characteristic (ROC)Model')
#plt.title('Receiver operating characteristic (ROC) of Potassium Model')
# plt.title('Receiver operating characteristic (ROC) of Boron Model')
# plt.title('Receiver operating characteristic (ROC) of Calcium Model')
# plt.title('Receiver operating characteristic (ROC) of Magnesium Model')
# plt.title('Receiver operating characteristic (ROC) of Manganese Model')
pyplot.plot([0, 1], [0, 1], 'k--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend(loc="lower right")
pyplot.savefig('EvaluationFigures/LATEST/ROC.jpg',dpi=300)
pyplot.show()

ADB = y_pred_test_ADB[:, np.newaxis]
ADBDF = pd.DataFrame(ADB,columns =['ADB'])

LGB = y_pred_test_LGB[:, np.newaxis]
LGBDF = pd.DataFrame(LGB,columns =['LGB'])

CATDF = pd.DataFrame(y_pred_test_CAT,columns =['CAT'])

SVM= y_pred_test_SVM[:, np.newaxis]
SVMDF = pd.DataFrame(SVM,columns =['SVM'])

RF = y_pred_test_RF[:, np.newaxis]
RFDF = pd.DataFrame(RF,columns =['RF'])

sleepBoostDF = pd.DataFrame(sleepBoost_test_pred,columns =['SleepBoost'])

y_test_DF = y_test.reset_index()


DFtoCSV = [SVMDF,ADBDF,RFDF, CATDF, LGBDF,sleepBoostDF, y_test_DF]
DFtoCSVFinal = pd.concat(DFtoCSV, axis=1, join='inner')
print(DFtoCSVFinal.shape)
# refined_col=['SVM','ADB','RF','CAT','LGB', 'SLEEPBOOST','ACTUAL']
DFtoCSVFinal.to_csv("EvaluationFigures/LATEST/COMPAREPOINTS.csv", index=None) #, columns= refined_col)

# explainer = shap.TreeExplainer(lgb_model)
# shap_values = explainer.shap_values(X)
#
# shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])


