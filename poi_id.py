#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler                                  
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', \
                 'loan_advances', 'bonus', 'restricted_stock_deferred', \
                 'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive', \
                 'exercised_stock_options', 'from_messages', 'other', \
                 'from_poi_to_this_person', 'from_this_person_to_poi', \
                 'deferred_income', 'expenses', 'restricted_stock', 'director_fees']    #all features apart from email address

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

poi_num = 0

for person in data_dict:
    if data_dict[person]['poi'] == True:                                        #count pois
        poi_num += 1

print "\nThere are", len(data_dict), "persons"
print "\nThere are", poi_num, "poi"
print "\nThere are", len(data_dict) - poi_num, "non-poi"
print "\nThere are", len(data_dict.values()[0]), "features for each person"
                        
data_sum = {}

for person in data_dict:
    for feature in data_dict[person]:                                           #set dat_dict to zeros
            data_sum[feature] = 0

for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == 'NaN':                                 #count nans
            data_sum[feature] += 1

print "\nTable of features with NaN count:\n", data_sum                         #to create full features list and see nans
print "\nTotal NaN sum:", sum(data_sum.values())
print "\nTotal number of values:", len(data_dict) * len(data_dict.values()[0])
print "\n", np.round(float(sum(data_sum.values())) /    \
                     float(len(data_dict) * len(data_dict.values()[0])) \
                          * 100, 2), '% of the data is NaN'                     #divide nan over total using flots to 2dp + change to %
        
### Task 2: Remove outliers

data = featureFormat(data_dict, ["salary", "bonus"])

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.title("Salary against Bonus")                                               #plot salary/bonus graph & save
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.savefig("Outliers.png")
plt.show()

df = pd.DataFrame(data_dict)                                                    #load pickle into dataframe
print "\nNames:\n", df.iloc[0]                                                  #print row 0 - names

data_dict.pop("TOTAL",0)                                                        #delete TOTAL in dictionary

data = featureFormat(data_dict, ["salary", "bonus"])

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.title("Salary against Bonus without TOTAL")                                 #plot salary/bonus graph without total and save
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.savefig("OutliersRemoved.png")
plt.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

def CalcPercent(POIemail, ALLemail):
    if POIemail != 'NaN' and ALLemail != 'NaN':                                 #if Nan
        percent = np.round(float(POIemail) / float(ALLemail),3)                 #calc percent to 3dp
    else:
        percent = 0.0                                                           #else return float 0
    return percent

for person in my_dataset:                                                       #create my features
    fromPOI = my_dataset[person]['from_poi_to_this_person']
    TOemail = my_dataset[person]['to_messages']
    my_dataset[person]['PERCENT_from_poi'] = CalcPercent(fromPOI, TOemail)

    toPOI = my_dataset[person]['from_this_person_to_poi']
    FROMemail = my_dataset[person]['from_messages']
    my_dataset[person]['PERCENT_to_poi'] = CalcPercent(toPOI, FROMemail)

    sharedPOI = my_dataset[person]['shared_receipt_with_poi']
    ALLemail = my_dataset[person]['to_messages'] + my_dataset[person]['from_messages']
    my_dataset[person]['PERCENT_shared_poi'] = CalcPercent(sharedPOI, ALLemail)

data2 = featureFormat(data_dict, ["PERCENT_from_poi", "PERCENT_to_poi"])

for point in data2:
    fromPOI = point[0]
    toPOI = point[1]
    plt.scatter(fromPOI, toPOI)

plt.title("Emails to/from POI as a percentage of all emails")                   #plot email to/from graph
plt.xlabel("From POI")
plt.ylabel("To POI")
plt.savefig("POIto-from.png")
plt.show()

print "\nOriginal features list:\n", features_list                              #add my created features
features_list = features_list + ['PERCENT_from_poi']
features_list = features_list + ['PERCENT_to_poi']
features_list = features_list + ['PERCENT_shared_poi']

print "\nFeatures list with % email:\n", features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

select = SelectKBest(f_classif, k='all')                                        #select best features
select.fit(features, labels)
score = zip(features_list[1:], select.scores_)                                  #ignore POI in list
scoreSorted = sorted(score, key=lambda score: score[1], reverse = True)         #sort by score
scoreDF = pd.DataFrame(scoreSorted)                                             #display in pandas
print "\nSelectKBest Sorted Score:\n", scoreDF
scoreDF.to_excel("SelectFeaturesScores.xlsx")                                   #output feature scores to excel

features_list = ['poi'] + [(feature[0]) for feature in scoreSorted[0:8]]
print "\nEdited features list:\n", features_list

processData = featureFormat(my_dataset, features_list, sort_keys = True)        #scale values
labels, features = targetFeatureSplit(processData)                              #split to labels and features
scaler = MinMaxScaler()                                                         #use Min Max scaler
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)


Results = []

def tuneModel(algorithm, Time, gridSearch, features, labels, parameters, iterations = 42):  #tune the models
    accuracy = []
    precision = []
    recall = []
    
    for iteration in range(iterations):
        features_train, features_test, labels_train, \
        labels_test = train_test_split(features, labels, test_size = 0.3, random_state = iteration)
        
        gridSearch.fit(features_train, labels_train)
        prediction = gridSearch.predict(features_test)
        accuracy = accuracy + [accuracy_score(labels_test, prediction)]
        precision = precision + [precision_score(labels_test, prediction)]
        recall = recall + [recall_score(labels_test, prediction)]
        
    print "\nBest parameters for", algorithm                                    #print parameters here as couldn't output to pandas
    best_params = gridSearch.best_estimator_.get_params()
    for param_name in parameters.keys():
        print param_name, ':', best_params[param_name]
    if algorithm == 'GaussianNB':                                               #no parameters passed to naive bays
        print 'None'
    results = {'1 Algorithm':algorithm, '2 Accuracy':np.mean(accuracy), \
    '3 Precision':np.mean(precision), '4 Recall':np.mean(recall), \
    '5 Time(seconds)':str(round(time()-Time, 2))}                               #output result to pandas
    return results


clf = GaussianNB()                                                              #test naive bayes
parameters = {}
gridSearch = GridSearchCV(estimator = clf, param_grid = parameters)
Results.append(tuneModel('GaussianNB', time(), gridSearch, features, labels, parameters)) 


clf = svm.SVC()                                                                 #test svm
parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'C': [1000, 100, 10, 1, 0.1]}
gridSearch = GridSearchCV(estimator = clf, param_grid = parameters)
Results.append(tuneModel('SVM', time(), gridSearch, features, labels, parameters))


clf = KNeighborsClassifier()                                                    #test knn
parameters = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
gridSearch = GridSearchCV(estimator = clf, param_grid = parameters)
Results.append(tuneModel('K Nearest Neigbors', time(), gridSearch, features, labels, parameters))  


compare = pd.DataFrame(Results)                                                 #print results
print "\nCompare algorithms:\n", compare
compare.to_excel("CompareAlgorithms08.xlsx")                                                 #output results to excel

Best = []

def bestModel(algorithm, Time, clf, features, labels, parameters):              #run best algorithm + parameters
    accuracy = 0.0                                                              #set scores as floats
    precision = 0.0
    recall = 0.0
    
    features_train, features_test, labels_train, \
    labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)
        
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, prediction)
    precision = precision_score(labels_test, prediction)
    recall = recall_score(labels_test, prediction)
        
    results = {'1 Algorithm':algorithm, '2 Accuracy':accuracy, \
    '3 Precision':precision, '4 Recall':recall, '5 Time(seconds)':str(round(time()-Time, 2))}
    return results


clf = GaussianNB()                                                              #select naive bays as best algorithm
parameters = {}
Best.append(bestModel('GaussianNB', time(), clf, features, labels, parameters))

final = pd.DataFrame(Best)
print "\nBest model output:\n", final


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)