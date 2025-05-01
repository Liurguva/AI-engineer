# -*- coding: utf-8 -*-
"""
Created by Leo Liu for the book "Artificial Intelligence for Engineers: Basics and Implementations".
Theory, symbol, and procedure are explained in the book. Major info can be found at wwww.AI-engineer.org.
This script file is for decision tree algorithms - ID3, C4.5, and CART can be selected.
"""

# import operator
from math import log2
import time
import numpy as np

def createDataSet():
    """
    Date Structure is as follows:
    dataSet = [Instance 1; Instance 2;,...]
    Instance 1 = [Feature1_value, Feature2_value, ..., Label 1]
    Features = [Feature1_name, Feature2_name,...]
    """
    """
    # 1. Create data manually
    dataSet=[[1,1,'yes'], # "no surfacing", "flippers", and classification label
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    Features = ['no surfaceing','flippers'] # Features/Attributes corresponding to the first two values in each feature vector
    """
    """
    # 2. Import data from sklearn
    from sklearn.datasets import load_breast_cancer # load_digits; load_iris; load_breast_cancer
    data_bundle = load_breast_cancer(return_X_y=False)
    dataSet = np.ndarray.tolist(data_bundle['data'])
    # dataSet = np.ndarray.tolist(np.hstack((data_bundle['data'],data_bundle['target'].reshape(data_bundle['target'].shape[0],-1))))
    for i in range(len(data_bundle['target'])):
        dataSet[i].append(str(data_bundle['target'][i]))
    # dataSet[:][-1]=str(dataSet[:][-1])
    Features = data_bundle['feature_names']
    """
    # 3. Import data from files
    dataSet = np.ndarray.tolist(np.loadtxt(open("DT_watermelon_data.txt", encoding='utf8'), dtype=str)) # delimiter='\t' may be needed in other languages
    Features = np.ndarray.tolist(np.loadtxt(open("DT_watermelon_feature.txt", encoding='utf8'), dtype=str))

    return dataSet, Features
    # """
    
# Calculate the Information Entropy
def Entropy(dataSet):
    N_instances = len(dataSet)
    Category_sizes = {} # A dictionary to store the numbers of samples for each category like {"yes": 2, "no": 3 }
    for Instance in dataSet:
        currentLabel = Instance[-1]
        if currentLabel not in Category_sizes:
            Category_sizes[currentLabel] = 0
        Category_sizes[currentLabel] += 1
    Ent = 0.0
    Gini = 1.0
    for key in Category_sizes:
        prob = float(Category_sizes[key])/N_instances
        Ent -= prob * log2(prob)
        Gini -= prob**2
    return Ent, Gini

# Select which feature will be used for splitting data at the current node based on information gain
def chooseBestFeatureToSplit(dataSet, method): # method = ID3, C4_5, or CART
    N_features = len(Features) # len(dataSet[0]) - 1 # The last element of every instance is not a feature but a label thus needs to be excluded
    baseEntropy,baseGini = Entropy(dataSet) # BaseGini is not used
    bestMeasure = 0.0
    bestFeature_num = -1
    for i in range(N_features):
        feature_i_values = [example[i] for example in dataSet] # Create a list of all possible values of feature i
        feature_i_unique_values = set(feature_i_values) # Create a list of all unique values of feature i
        newEntropy = 0.0
        IV_a = 1e-5
        Gini_index = 0.0
        for feature_value in feature_i_unique_values:
            subDataSet = splitDataSet(dataSet, i, feature_value)
            prob = len(subDataSet) / float(len(dataSet)) # Dv/D
            Ent, Gini = Entropy(subDataSet) 
            newEntropy += prob * Ent
            IV_a -= prob * np.log2(prob) 
            Gini_index += prob * Gini
        infoGain = baseEntropy - newEntropy # Information gain for ID3
        GainRatio = infoGain / IV_a
        
        if method == 'ID3':
            Measure = infoGain
        elif method == 'C4_5':
            Measure = GainRatio
        elif method == 'CART':
            Measure = Gini_index
        
        if Measure > bestMeasure:
            bestMeasure = Measure
            bestFeature_num = i
        
    return bestFeature_num

# Split a dataset at the current node for a feature and one of its feature values and generate a subset for the sub-node corresponding to the feature value
def splitDataSet(dataSet, feature_num, feature_value): # axis is the feature no.: 0 is the 1st feature, 1 is the 2nd; value is the unique values of the feature
    subDataSet = []
    for Instance in dataSet:
        if Instance[feature_num] == feature_value:
            reducedFeatVec = Instance[:feature_num]
            reducedFeatVec.extend(Instance[feature_num+1:]) # These two lines take off the feature_num feature that has a vlue of feature_value
            subDataSet.append(reducedFeatVec)
    return subDataSet
                
# We select features from the feature pool to grow the tree step by step. In some cases, all features have been used while there are still 
# multiple categories in the dataset. In that case, we will treat this node as a leaf node and vote by majority to determine the label/category.
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)         
    
def createTree(dataSet, Features):
    categoryList = [Instance[-1] for Instance in dataSet] # Create a list of labels/categories/classes
    if categoryList.count(categoryList[0]) == len(categoryList): # Stop data splitting if there is only one category
    # The count() method returns the number of times the specified element appears in the list.
        return categoryList[0]
    dataSet_featuresonly = [j[0:-1] for j in dataSet]
    # print("1 is", dataSet,'\t', "2 is", all(x==dataSet_featuresonly[0] for x in dataSet_featuresonly))
    if len(Features) == 0 or all(x==dataSet_featuresonly[0] for x in dataSet_featuresonly): # No features left or all the samples have identical feature values
        return majorityCnt(categoryList)
    bestFeature_num = chooseBestFeatureToSplit(dataSet, 'ID3')
    bestFeature = Features[bestFeature_num]
    myTree = {bestFeature:{}}
    del(Features[bestFeature_num]) # Use Features to record a list of available features; Here it is updated by removing the selected feature
    feature_values = [Instance[bestFeature_num] for Instance in dataSet]
    feature_unique_values = set(feature_values)
    for value in feature_unique_values:
         myTree[bestFeature][value] = createTree(splitDataSet(dataSet, 
                                        bestFeature_num, value),Features)
    return myTree
    

dataSet,Features = createDataSet()
t1 = time.process_time() #time.clock()
myTree = createTree(dataSet,Features)
t2 = time.process_time() #time.clock()
# print (myTree)
print ('execute for ', t2-t1)

import tree_plot_function
tree_plot_function.main(myTree)