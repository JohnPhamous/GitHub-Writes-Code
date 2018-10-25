# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import statsmodels.api as sm

# import numpy as np

'''
Previously obtained features from the RFE model; empty if we have none
'''
rfeFeatures = []
# pd.set_option('display.expand_frame_repr', True) #Set to false to show more columns

# seed = np.random.randint(1,100) #Get consistent breakdown across models
model = LogisticRegression()  # Only one model for now


def main(filename, keyColumn):
    """
    Calls the other functions
    """
    data = readData(filename)
    viewData(data, keyColumn)
    data = cleanData(data)
    testSize = 0.3  # What percentage of the data is reserved for testing?
    targetFeatures = None  # How many features do we want in the end?
    threshold = 0.09  # Manually set a threshold for the classifier
    features = recursiveFeatureElimination(data, keyColumn, model, targetFeatures)
    xTrain, xValid, yTrain, yValid = classificationSetup(data, testSize, keyColumn, features)
    runAlgorithms(xTrain, xValid, yTrain, yValid, 'roc_auc')
    predictions(xTrain, xValid, yTrain, yValid, model, threshold)


def readData(filename):
    """
    Reads in the data from the CSV file with name filename and returns it
    """
    table = pd.read_csv(filename, header=0)
    if "Unnamed" in table.columns.values[0]:
        table.set_index("Unnamed: 0", inplace=True)
    return table


def showCrosstabFrequency(data, xx, yy):
    """
    Compute and display crosstab frequency between xx and yy
    in dataset data. Show results as 100% stacked bar chart
    """
    ct = pd.crosstab(data[xx], data[yy])
    ct = ct.div(ct.sum(1), axis=0)  # Normalize
    ct.plot(kind='bar', stacked=True)
    plt.title("{0} frequency across {1}".format(xx.title(), yy.title()))
    plt.xlabel(xx)
    plt.ylabel("Frequency")
    plt.show()


def viewData(data, keyVar, cutoff=20):
    """
    Display some information about the data
    and show graphs of variables against the key variable
    """
    print(data[keyVar].value_counts())  # A count of our key variable
    print(data[keyVar].value_counts(normalize=True))  # A percentage of our key variable
    print(data.groupby(keyVar).mean())
    for col in data.columns.values:
        kinds = data[col].unique()
        if col != keyVar and len(kinds) < cutoff:
            showCrosstabFrequency(data, col, keyVar)


def recursiveFeatureElimination(data, keyVar, model, features=None):
    """
    Use RFE to help identify features to use in the model
    """
    if rfeFeatures:
        featureCheck(*separateKeyVar(data, keyVar, rfeFeatures))
        return rfeFeatures
    xx, yy = separateKeyVar(data, keyVar)
    # Optional second argument tells it how many features to pick
    # Default is half
    rfe = RFE(model, features)
    fit = rfe.fit(xx, yy)
    print(fit.support_)
    print(fit.ranking_)
    useCols = xx.columns[fit.support_].values
    print(useCols)

    featureCheck(*separateKeyVar(data, keyVar, useCols))

    return useCols


def featureCheck(xx, yy):
    logModel = sm.Logit(yy, xx)
    results = logModel.fit()
    print(results.summary())


def separateKeyVar(data, keyVar, useCols=None):
    """
    Separates data into two pieces:
        1. Containing all but keyVar
        2. Containing keyVar
    If useCols is not None, returns only those in the xData
    """
    if useCols is None:
        xColumns = data.columns.tolist()  # Get column names
        xColumns.remove(keyVar)  # Remove key column
    else:
        xColumns = useCols
    xData = data.loc[:, xColumns]  # Get non-key columns
    yData = data.loc[:, keyVar]  # Get key column
    return xData, yData


def cleanData(data):
    """
    Cleans up the data, including making dummies out of categorical variables
    """
    return pd.get_dummies(data)


def classificationSetup(data, valSize, keyColumn, features=None):
    """
    Prepares the data for analysis
    """
    xx, yy = separateKeyVar(data, keyColumn, features)
    xData, yData = xx.values, yy.values  # Don't want DataFrames here
    xTrain, xValid, yTrain, yValid = train_test_split(xData, yData, test_size=valSize)  # ,random_state=seed)
    return xTrain, xValid, yTrain, yValid


def runAlgorithms(xTrain, xValid, yTrain, yValid, scoring='accuracy', folds=3):
    """
    Runs an algorithm on the data to do some machine learning analysis
    """
    kfold = KFold(n_splits=folds)  # , random_state=seed)
    cv_results = cross_val_score(model, xTrain, yTrain, cv=kfold, scoring=scoring)
    msg = "{0:5s}: {1:.6f} ({2:.6f})".format("LR", cv_results.mean(), cv_results.std())
    print(msg)


def adjustClasses(yScores, threshold):
    """
    Adjust class predictions to a given threshold
    """
    return [1 if yy >= threshold else 0 for yy in yScores]


def showROC(yy, predict, probs):
    """
    Show the ROC curve for the given data
    """
    rocAuc = roc_auc_score(yy, predict)
    fpr, tpr, thresholds = roc_curve(yy, probs)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % rocAuc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


def predictions(xTrain, xValid, yTrain, yValid, alg, threshold=0):
    """
    Evaluates the predictive power of our algorithm
    """
    fit = alg.fit(xTrain, yTrain)  # Fit the algorithm to our data
    predictions = alg.predict(xValid)  # Make predictions
    probs = fit.predict_proba(xValid)[:, 1]
    if threshold > 0:
        predictions = adjustClasses(probs, threshold)

    showROC(yValid, predictions, probs)
    print(roc_auc_score(yValid, predictions))
    print(confusion_matrix(yValid, predictions))
    print(classification_report(yValid, predictions))


main("diabetes_data.csv", "readmit_30")
