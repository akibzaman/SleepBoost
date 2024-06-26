import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
# import shap
print(plt.style.available)
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

############## 0.1 ################

data = pd.read_csv("/Users/akibzaman/Codes/SleepBoost/Git Code/SleepBoost/data/alldatacsv/selected/selected_data_1.csv") #use your own path link here 
X, y = data.loc[:, "std":"D19"], data["class"]
#
# ############# 0.2 ################
# data = pd.read_csv("alldatacsv/selected/selected_data_2.csv")
# X, y = data.loc[:,"std":"D19"], data['class']
#
# ############# 0.3 ################
# data = pd.read_csv("alldatacsv/selected/selected_data_3.csv")
# X, y = data.loc[:,"std":"D6"], data['class']
FEATURES = [
    "std", "var", "minim", "maxim", "rms", "median", "absDiffSignal", "kurtosis",
    "HM", "HC", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E1",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19"
    ]
RESULTS = {"accuracy": {}, "precision": {}, "recall": {}, "f1_score": {}, "kappa": {}}
CM_AVG = 0
acc = []
pre = []
rec = []
f1 = []
kappa = []
for i in range(10):
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

    # ###### SHAP ANALYSIS
    # # Initialize SHAP explainer
    # explainer = shap.Explainer(lgb_model, X_train)
    # # Calculate SHAP values
    # shap_values = explainer(X_test, )
    # # Choose the class index for which you want to visualize SHAP values
    # # For instance, if you want to visualize the importance for the first class in a multi-class classification, set class_index=0
    # class_index = 0
    # # Visualize the SHAP values for the specified class
    # shap.summary_plot(shap_values[..., class_index].values, X_test, feature_names=FEATURES)

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
        max_features= "sqrt",
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

    # ###### SHAP ANALYSIS
    # # Initialize SHAP explainer
    # explainer = shap.Explainer(model_RF, X_train)
    # # Calculate SHAP values
    # shap_values = explainer(X_test)
    # # Choose the class index for which you want to visualize SHAP values
    # # For instance, if you want to visualize the importance for the first class in a multi-class classification, set class_index=0
    # class_index = 0
    # # Visualize the SHAP values for the specified class
    # shap.summary_plot(shap_values[..., class_index].values, X_test, feature_names=FEATURES)

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

    C_test_Final = pd.concat(
        [C1_cat_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1
    )

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
    print("Fold Number {0}:\n".format(i))
    # print(metrics.classification_report(y_test, ensemble_test_pred))
    print(metrics.accuracy_score(y_test, ensemble_test_pred))
    acc.append(metrics.accuracy_score(y_test, ensemble_test_pred))
    print(metrics.precision_score(y_test, ensemble_test_pred, average="macro"))
    pre.append(metrics.precision_score(y_test, ensemble_test_pred, average="macro"))
    print(metrics.recall_score(y_test, ensemble_test_pred, average="macro"))
    rec.append(metrics.recall_score(y_test, ensemble_test_pred, average="macro"))
    print(metrics.f1_score(y_test, ensemble_test_pred, average="macro"))
    f1.append(metrics.f1_score(y_test, ensemble_test_pred, average="macro"))
    print(metrics.cohen_kappa_score(y_test, ensemble_test_pred))
    kappa.append(metrics.cohen_kappa_score(y_test, ensemble_test_pred))
    print("\n")

    feature_names = ["Actual", "Predicted"]
    actual_class = ["W", "N1", "N2", "N3", "REM"]

    cm = confusion_matrix(y_test, ensemble_test_pred)
    CM_AVG = CM_AVG + cm
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
    # pyplot.show()

    # from sklearn.metrics import roc_curve
    # from sklearn.metrics import roc_auc_score
    #
    # # roc curve for models
    # fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred_test_trail2, pos_label=1)
    # fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred_test_model1, pos_label=1)
    #
    # # roc curve for tpr = fpr
    # random_probs = [0 for i in range(len(y_test))]
    # p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    #
    # # auc scores
    # auc_score1 = roc_auc_score(y_test, y_pred_test_trail2)
    # auc_score2 = roc_auc_score(y_test, y_pred_test_model1)
    #
    # print(auc_score1)#, auc_score2)
    #
    # pyplot.style.use ('seaborn-pastel') #('seaborn-whitegrid')
    # # plot roc curves
    # pyplot.plot(fpr1, tpr1, linestyle='--',color='green', label='XACatNet:Calibrated using SCE (area = {0:0.2f})'.format(auc_score1))
    # pyplot.plot(fpr2, tpr2, linestyle='--',color='red', label='Non-Calibrated Base Model (area = {0:0.2f})'.format(auc_score2))
    # pyplot.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # # title
    # # pyplot.title('ROC curve')
    # pyplot.plot([0, 1], [0, 1], 'k--')
    # pyplot.xlim([0.0, 1.0])
    # pyplot.ylim([0.0, 1.02])
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    # pyplot.legend(loc="lower right")
    # pyplot.savefig('ROC',dpi=300)
    #
    # plt.show()


#### Line Plotting 
# # List of folds
# folds = list(range(1, 11))

# # Plotting
# pyplot.figure(figsize=(10, 6))
# pyplot.plot(folds, acc, label='Accuracy', marker='o')
# pyplot.plot(folds, pre, label='Precision', marker='o')
# pyplot.plot(folds, rec, label='Recall', marker='o')
# pyplot.plot(folds, f1, label='F1 Score', marker='o')
# pyplot.plot(folds, kappa, label='Kappa', marker='o')

# # Adding titles and labels
# pyplot.title('Performance Metrics Across Folds')
# pyplot.xlabel('Fold')
# pyplot.ylabel('Score')
# pyplot.xticks(folds)  # Ensure we have a tick for each fold
# pyplot.legend()
# pyplot.grid(True)

# # Show the plot
# pyplot.show()


###### Box Plot

data = [acc, pre, rec, f1, kappa]
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Kappa']

# Define figure and style
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Create custom lines for the legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#4286f4', lw=1),
                Line2D([0], [0], color='red', lw=1)]

# Iterate through each metric to plot high, low, and average
for i, metric in enumerate(data):
    # Calculate high, low, and average values
    high = np.max(metric)
    low = np.min(metric)
    avg = np.mean(metric)
    
    # Plot high and low with thin blue lines
    plt.plot([i+1-0.1, i+1+0.1], [high, high], color='#4286f4', linewidth=1, label='High' if i == 0 else "")  # High line
    plt.plot([i+1-0.1, i+1+0.1], [low, low], color='#4286f4', linewidth=1, label='Low' if i == 0 else "")  # Low line
    
    # Plot average with a thin red line
    plt.plot([i+1-0.1, i+1+0.1], [avg, avg], color='red', linewidth=1, label='Average' if i == 0 else "")  # Average line

# Customizing the plot
ax = plt.gca()
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set the y-axis range and ticks
ax.set_ylim(0.7, 1.0)
ax.set_yticks(np.arange(0.7, 1.0, 0.05))

# Adding titles, labels, and custom legend
plt.title('Performance Metrics Across Folds', fontsize=14, fontweight='bold')
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(range(1, len(labels) + 1), labels, fontsize=10)

plt.legend(custom_lines, ['High & Low', 'Average'], loc='best')

# Show or save the plot
# plt.show()
plt.savefig('Performance Metrics Across Folds - High, Low, and Average Line with Legend.png', dpi=300, bbox_inches='tight')



accuracy = sum(acc) / len(acc)
precision = sum(pre) / len(pre)
recall = sum(rec) / len(rec)
f1Score = sum(f1) / len(f1)
kohenKappa = sum(kappa) / len(kappa)

print("Average Accuracy: {0}".format(accuracy))
print("Average Precision: {0}".format(precision))
print("Average Recall: {0}".format(recall))
print("Average F1 Score: {0}".format(f1Score))
print("Average Kappa Score: {0}".format(kohenKappa))
