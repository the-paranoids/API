import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as KM
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import euclidean
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.font_manager
import math

# reading features from file
data = pd.DataFrame.from_csv("../data/DSL-StrongPasswordData3.csv",index_col = 0)

subjects = ['s003'] # which subjects to create a profile for
outliers = ['s027']
rep_training = [i for i in range(1,11)] # total number of repetitions to use for training
session_training = [1,2] # total number of sessions to use for training
rep_test = [i for i in range(1,11)]
session_test = [5,6]

rep_valid = [i for i in range(1,11)]
session_valid = [5,7]

size = len(rep_training) * len(session_training) # total number of training values
size2 = len(rep_test) * len(session_test)
size3 = len(rep_valid) * len(session_valid)

print("\nSize of Training Set 1: ",size)
print("\nSize of Traing Set 2:",size2)
print("\nSize of Traing Set 2:",size3)
#print(data)

training_data = data.loc[:,:][ (data['subject'].isin(subjects)) & (data['sessionIndex'].isin(session_training)) & (data['rep'].isin(rep_training)) ]

test_data = data.loc[:,:][ (data['subject'].isin(subjects)) & (data['sessionIndex'].isin(session_test) ) & (data['rep'].isin(rep_test)) ]

outlier_data = data.loc[:,:][ (data['subject'].isin(outliers)) & (data['sessionIndex'].isin(session_valid) ) & (data['rep'].isin(rep_valid)) ]


#print(test_data)
#training_data = data.loc[:,['subject','sessionIndex','rep']][ (data['subject'] == 's002') & (data['sessionIndex'].isin(session)) & (data['rep'].isin(training)) ]
#print(subjects)

# create a list with all column names
names = [ name for name in training_data.columns ]
names.remove('subject')
names.remove('rep')
names.remove('sessionIndex')

#print(len(names))

# create a dictionary to store mean values for keyboard input
header = {}
for name in names :
    header[name] = 0

print(header)

#print(header)
#print(training_data)
#print(test_data)
training_data = training_data.reset_index()
test_data = test_data.reset_index()
outlier_data = outlier_data.reset_index()


#Calculating SUM
count = 0
for name, item in training_data.iteritems() :
    for i in range(len(item)) :
        if name in header :
            header[name] += item[i]
        if name == 'DD.Shift.r.o' :
            count += item[i]


#Calculating MEAN
print("\n\nSummed values: \n",header)
#print(len(header))
mean_data = {}

for name in header :
    mean_data[name] = header[name] / size

print("\n\nMean value:")
print(mean_data)

#print(len(mean_data))
#print(training_data['H.a'])

#print(training_data)



#print(training_data)

difference_training_data = pd.DataFrame()
sum_ED = {}
sum_temp = {}


array = {} # Euclidean Distance for a Feature
array2 = {}
'''
for name, item in training_data.iteritems() :
    if name in header :
        #print("Train",training_data[name])
        #print("Mean",mean_data[name])
        training_data[name].subtract(mean_data[name])
        difference_training_data[name] = training_data[name].pow(2)
        #print(difference_training_data)
        array[name]= difference_training_data[name].sum()

        array[name] = array[name]**0.5
'''

for name in header :
    array2[name] = euclidean(training_data[name],mean_data[name])
print("EUCLIDEAN DISTANCE",array2)

'''
sum_temp2 = {}

for name in header :
    if(name not in sum_temp) :
        sum_temp[name] = 0

    sum_temp[name] += sum_ED[name]
    sum_temp2[name] = sum_temp[name]/size

#print(sum_temp['H.Return'])


print("\n\nEuclidean Distances for Training Data:")
for name in header :
    print(array[name])
'''

# DELTA value (threshold) for a valid input
test_distance = {}



for name in header :
    if name not in test_distance :
        test_distance[name] = 0
    test_distance[name] = (test_data[name].iloc[0] - mean_data[name])

import collections
#array = OrderedDict(array2)

ED = pd.Series(array2,array2.keys())
#print(ED)


test_data.drop(['index','subject','rep','sessionIndex'],inplace=True,axis=1)

outlier_data.drop(['index','subject','rep','sessionIndex'],inplace=True,axis=1)

training_data.drop(['index','subject','rep','sessionIndex'],inplace=True,axis=1)


scores_train = []
scores_test = []
scores_out = []
for i in range(0,size3) :
    score = euclidean(ED,training_data.iloc[i])
    scores_train.append(score)
    #print(score)

for i in range(0,size) :
    score = euclidean(ED,test_data.iloc[i])
    scores_test.append(score)
    #print(score)

for i in range(0,size) :
    score = euclidean(ED,outlier_data.iloc[i])
    scores_out.append(score)
    #print(score)


flag = {}
a=5
count = 0
max_train = 0
max_test = 0
max_out = 0

for i in range(0,size3) :
    print("Outlier: ", scores_test[i]-scores_train[i])
    print("Valid: ",scores_out[i] -scores_train[i])
    max_train = max(max_train,scores_train[i])
    max_test = max(max_test,scores_test[i])
    max_out = max(max_out,scores_out[i])

for i in range(0,size3) :
    scores_test[i] /= max_test
    scores_train[i] /= max_train
    scores_out[i] /= max_out

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Don't cheat - fit only on training data
scaler.fit(scores_train)
X_train = scaler.transform(scores_train)
print(X_train)
# apply same transformation to test data
X_test = scaler.transform(scores_test)
print(X_test)

X_out = scaler.transform(scores_out)
print(X_out)

count = 0
count2 = 0
for item in range(len(X_out)) :
    if(X_[item] < abs(X_train[item])) :
        count+=1

print(count)
'''

print("\n\nDelta for Euclidean Distance:")
#print(valid_distance)

test_distance = {}
#print(test_data.size,valid_data.size)
mini = 100
count = 0

#for i in range(0,size3) :
for name in header :
    if name not in test_distance :
        test_distance[name] = 0
    test_distance[name] = (sum_temp[name] - valid_data[name].iloc[5])**2
    test_distance[name] **= 0.5
    #print(test_distance)
    #for i in range(0,size3) :

            #print("\n\nValidating with New Data:")

flag = {}
count = 0
for name in header :
    flag[name] = 0
    if (abs(test_distance[name]) >= 0.9*valid_distance[name]) and (abs(test_distance[name] <= 1.1*valid_distance[name])):
        flag[name] = 1
        count += 1

print(count/31)

'''


'''


for index, value in sum.iteritems() :
    sum2 += value**2

sum2 = math.sqrt(sum2)
print(sum2)

#print(difference_training_data.shape)
#print(training_data.shape)


test_data.drop(['index','subject','rep','sessionIndex'],inplace=True,axis=1)

training_data.drop(['index','subject','rep','sessionIndex'],inplace=True,axis=1)

outlier_data.drop(['index','subject','rep','sessionIndex'],inplace=True,axis=1)

# calculating euclidean distance
#for name, item in training_data.iteritems() :
#    if name in header :


#print("Mean value (H.a) \n:",mean_data['H.a'])
#print("\nTraining data (H.a) \n:",training_data['H.a'])

#print("\nTraining Samples: ",size)
#print("\nNumber of Features: ", len(header),"\n\n")




#print(training_data.shape)



#print(training_data)
#print(test_data)

kmeans = KM(n_clusters = 2)
kmeans.fit(training_data)
kmeans.predict(training_data)

print("\n\nK-Means Clustering:\n")

print("\nTest Data:")
for i in range(0,len(test_data)) :
    score = kmeans.score(test_data.iloc[[i]])
    print(score)

print("\nImposter Data:")
for i in range(0,len(outlier_data)) :
    score = kmeans.score(outlier_data.iloc[[i]])
    print(score)

'''

'''
print("\n\nHierachical Clustering")
HC = linkage(training_data,'average')

c, coph_dist = cophenet(HC,pdist(training_data))
print(c)


c, coph_dist = cophenet(HC,pdist(test_data.iloc[[0]]))
print(c)


print("\n\nSupport Vector Classification: ")
clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
clf.fit(training_data)
y_pred_train = clf.predict(training_data)
print("\nTraining Data:")
print(len(y_pred_train))
print(y_pred_train)

y_pred_test = clf.predict(test_data)
print("\nTest Data:")
print(len(y_pred_test))
print(y_pred_test)
print("\nOutlier Data:")
y_pred_outliers = clf.predict(outlier_data)
print(len(y_pred_outliers))
print(y_pred_outliers)


n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
print(n_error_train)
print(n_error_test)
print(n_error_outliers)
'''
'''

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

b1 = plt.scatter(training_data[:, 0], training_data[:, 1], c='white')
b2 = plt.scatter(test_data[:, 0], test_data[:, 1], c='green')
c = plt.scatter(outlier_data[:, 0], outlier_data[:, 1], c='red')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
'''
'''

print("\n\nMean values: \n",average_data)




    for name in header :
        print(item)

#print(header)
#print(training_data)

'''




