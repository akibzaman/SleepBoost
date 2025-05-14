import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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
data = pd.read_csv("alldatacsv/selected/selected_data_1.csv") #Use your own data path
X, y = data.loc[:, "std":"D19"], data["class"]
#
# ############# 0.2 ################
# data = pd.read_csv("alldatacsv/selected/selected_data_2.csv")
# X, y = data.loc[:,"std":"D19"], data['class']
#
# ############# 0.3 ################
# data = pd.read_csv("alldatacsv/selected/selected_data_3.csv")
# X, y = data.loc[:,"std":"D6"], data['class']

X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=32
)  # stratify=y
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=32
)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)

######## Normalization ########


def normalizeData(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    NX_train = scaler.transform(X_train)
    scaler.fit(X_test)
    NX_test = scaler.transform(X_test)
    return NX_train, NX_test


def showDistribution(y):
    counter = Counter(y)
    for k, v in counter.items():
        per = v / len(y) * 100
        print("Class=%d, n=%d (%.3f%%)" % (k, v, per))
    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()

    ### Normalization and OverSampling

    # normalizeData(X_train, X_test)
    # showDistribution(y_train)

    # oversample = SMOTE()
    # X_train, y_train = oversample.fit_resample(X_train, y_train)
    # showDistribution(y_train)

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    ###############     ENSEMBLE     ########################

    # ########CatBoost Model


print("CatBoost------------------------------->")
model_Cat = CatBoostClassifier()
model_Cat.fit(X_train, y_train)
C1_knn_pred = model_Cat.predict(X_valid)
C1_knn_pred = pd.DataFrame(C1_knn_pred, columns=["C1KNN"])
C1_knn_pred = C1_knn_pred.reset_index()
C1_knn_pred = C1_knn_pred.drop(["index"], axis=1)

###LGBM
print("LGBM------------------------------->")
lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    num_leaves=25,
    min_data_in_leaf=6,
    learning_rate=0.4,
    n_estimators=200,
    objective="multiclass",
    class_weight="balanced",
    min_split_gain=0.0,
    min_child_weight=0.03,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=500,
    colsample_bytree=1.0,
    reg_lambda=0.65,
    random_state=0,
    num_classes=3,
)
lgb_model.fit(X_train, y_train)
C1_lgbm_pred = lgb_model.predict(X_valid)
C1_lgbm_pred = pd.DataFrame(C1_lgbm_pred, columns=["C2LGBM"])
C1_lgbm_pred = C1_lgbm_pred.reset_index()
C1_lgbm_pred = C1_lgbm_pred.drop(["index"], axis=1)

# RandonForest
print("--------------- RF ------------------------------->")
model_RF = RandomForestClassifier(
    n_estimators=200,
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
    max_samples=None,
)
model_RF.fit(X_train, y_train)
C1_rf_pred = model_RF.predict(X_valid)
C1_rf_pred = pd.DataFrame(C1_rf_pred, columns=["C3RF"])
C1_rf_pred = C1_rf_pred.reset_index()
C1_rf_pred = C1_rf_pred.drop(["index"], axis=1)

actual = pd.DataFrame(y_valid, columns=["class"])
actual = actual.reset_index()
actual = actual.drop(["index"], axis=1)

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
        if instance[0] == instance[3]:
            correct += 1
            flag[0] = 1
        if instance[1] == instance[3]:
            correct += 1
            flag[1] = 1
        if instance[2] == instance[3]:
            correct += 1
            flag[2] = 1
        # print(correct)
        # print(flag)
        if correct > 0 and correct <= 3:
            # print("inside")
            if correct == 2:
                for k in range(len(flag)):
                    if flag[k] == 1:
                        weight[k] += VALUE1
            # else:
            # 	weight[k]-=(VALUE2/3)
            if correct == 1:
                for k in range(len(flag)):
                    if flag[k] == 1:
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
C1_cat_test_pred = pd.DataFrame(C1_cat_test_pred, columns=["C1KNN"])
C1_cat_test_pred = C1_cat_test_pred.reset_index()
C1_cat_test_pred = C1_cat_test_pred.drop(["index"], axis=1)

###LGBM
C1_lgbm_test_pred = lgb_model.predict(X_test)
C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=["C2LGBM"])
C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
C1_lgbm_test_pred = C1_lgbm_test_pred.drop(["index"], axis=1)

# RandonForest
C1_rf_test_pred = model_RF.predict(X_test)
C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=["C3RF"])
C1_rf_test_pred = C1_rf_test_pred.reset_index()
C1_rf_test_pred = C1_rf_test_pred.drop(["index"], axis=1)

C_test_Final = pd.concat([C1_cat_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)


def SleepBoost(df, weight):
    prdicted_classes = [0] * len(df)
    for i in range(len(df)):
        matrix = [[0.0, 0.0, 0.0, 0.0, 0.0]] * 3  # Number of Classes
        # print(matrix)
        # break
        instance = df.iloc[i]
        for j in range(len(instance)):
            if instance[j] == 0:
                matrix[j] = [weight[j], 0.0, 0.0, 0.0, 0.0]
            elif instance[j] == 1:
                matrix[j] = [0.0, weight[j], 0.0, 0.0, 0.0]
            elif instance[j] == 2:
                matrix[j] = [0.0, 0.0, weight[j], 0.0, 0.0]
            elif instance[j] == 3:
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


ensemble_test_pred = SleepBoost(C_test_Final, weight)
print(metrics.classification_report(y_test, ensemble_test_pred))
print(metrics.accuracy_score(y_test, ensemble_test_pred))
print(metrics.precision_score(y_test, ensemble_test_pred, average="macro"))
print(metrics.recall_score(y_test, ensemble_test_pred, average="macro"))
print(metrics.f1_score(y_test, ensemble_test_pred, average="macro"))
print(metrics.cohen_kappa_score(y_test, ensemble_test_pred))
print("\n")

feature_names = ["Actual", "Predicted"]
actual_class = ["W", "N1", "N2", "N3", "REM"]

cm = confusion_matrix(y_test, ensemble_test_pred)
test_cm = pd.DataFrame(cm, index=actual_class, columns=actual_class)
sns.heatmap(
    test_cm,
    robust=True,
    linewidths=0.1,
    linecolor="grey",
    square=True,
    annot=True,
    fmt="d",
    cmap="BuGn",
    annot_kws={"size": 16},
)
pyplot.show()


# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
#
# # roc curve for models
# fpr1, tpr1, thresh1 = roc_curve(y_test, ensemble_test_pred, pos_label=1)
# # fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred_test_model1, pos_label=1)
#
# # roc curve for tpr = fpr
# random_probs = [0 for i in range(len(y_test))]
# p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
#
# # auc scores
# auc_score1 = roc_auc_score(y_test, ensemble_test_pred)
# # auc_score2 = roc_auc_score(y_test, y_pred_test_model1)
#
# print(auc_score1)  # , auc_score2)
#
# pyplot.style.use('seaborn-pastel')  # ('seaborn-whitegrid')
# # plot roc curves
# pyplot.plot(fpr1, tpr1, linestyle='--', color='green', label='SleepBoost (area = {0:0.2f})'.format(auc_score1))
# # pyplot.plot(fpr2, tpr2, linestyle='--',color='red', label='Non-Calibrated Base Model (area = {0:0.2f})'.format(auc_score2))
# pyplot.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# # title
# # pyplot.title('ROC curve')
# pyplot.plot([0, 1], [0, 1], 'k--')
# pyplot.xlim([0.0, 1.0])
# pyplot.ylim([0.0, 1.02])
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# pyplot.legend(loc="lower right")
# # pyplot.savefig('ROC',dpi=300)
#
# pyplot.show()
