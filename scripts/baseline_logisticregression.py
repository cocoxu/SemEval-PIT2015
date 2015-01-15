# ==============  SemEval-2015 Task 1  ==============
#  Paraphrase and Semantic Similarity in Twitter
# ===================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# Implementation of a baseline system that uses logistic 
# regression model with simple n-gram features, which 
# is originally described in the ACL 2009 paper
# http://www.aclweb.org/anthology/P/P09/P09-1053.pdf
# by Dipanjan Das and Noah A. Smith 
# 
# A few Python packages are needed to run this script:
# http://www.nltk.org/_modules/nltk/classify/megam.html
# http://www.umiacs.umd.edu/~hal/megam/index.html



from __future__ import division

import sys
import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import MaxentClassifier
from nltk.stem import porter

from cPickle import load
from cPickle import dump

from collections import *


# sub-functions for find overlapping n-grams
def intersect_modified (list1, list2) :
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1    
    inter = cnt1 & cnt2
    union = cnt1 | cnt2
    largeinter = Counter()
    for (element, count) in inter.items():
        largeinter[element] = union[element]
    return list(largeinter.elements())

def intersect (list1, list2) :
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1    
    inter = cnt1 & cnt2
    return list(inter.elements())



    
# create n-gram features and stemmed n-gram features
def paraphrase_Das_features(source, target, trend):
    source_words = word_tokenize(source)
    target_words = word_tokenize(target)
	
    features = {}
    
    ###### Word Features ########
	
    s1grams = [w.lower() for w in source_words]
    t1grams = [w.lower() for w in target_words]
    s2grams = []
    t2grams = []
    s3grams = []
    t3grams = []
        
    for i in range(0, len(s1grams)-1) :
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i+1]
            s2grams.append(s2gram)
        if i < len(s1grams)-2:
            s3gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2]
            s3grams.append(s3gram)
            
    for i in range(0, len(t1grams)-1) :
        if i < len(t1grams) - 1:
            t2gram = t1grams[i] + " " + t1grams[i+1]
            t2grams.append(t2gram)
        if i < len(t1grams)-2:
            t3gram = t1grams[i] + " " + t1grams[i+1] + " " + t1grams[i+2]
            t3grams.append(t3gram)

    f1gram = 0        
    precision1gram = len(set(intersect(s1grams, t1grams))) / len(set(s1grams))
    recall1gram    = len(set(intersect(s1grams, t1grams))) / len(set(t1grams))
    if (precision1gram + recall1gram) > 0:
        f1gram = 2 * precision1gram * recall1gram / (precision1gram + recall1gram)
    precision2gram = len(set(intersect(s2grams, t2grams))) / len(set(s2grams))
    recall2gram    = len(set(intersect(s2grams, t2grams))) / len(set(t2grams))
    f2gram = 0
    if (precision2gram + recall2gram) > 0:
        f2gram = 2 * precision1gram * recall2gram / (precision2gram + recall2gram)
    precision3gram = len(set(intersect(s3grams, t3grams))) / len(set(s3grams))
    recall3gram    = len(set(intersect(s3grams, t3grams))) / len(set(t3grams))
    f3gram = 0
    if (precision3gram + recall3gram) > 0:
        f3gram = 2 * precision3gram * recall3gram /(precision3gram + recall3gram)

    features["precision1gram"] = precision1gram
    features["recall1gram"] = recall1gram
    features["f1gram"] = f1gram
    features["precision2gram"] = precision2gram
    features["recall2gram"] = recall2gram
    features["f2gram"] = f2gram
    features["precision3gram"] = precision3gram
    features["recall3gram"] = recall3gram
    features["f3gram"] = f3gram
    
    ###### Stemmed Word Features ########
    
    porterstemmer = porter.PorterStemmer()
    s1stems = [porterstemmer.stem(w.lower()) for w in source_words]
    t1stems = [porterstemmer.stem(w.lower()) for w in target_words]
    s2stems = []
    t2stems = []
    s3stems = []
    t3stems = []
        
    for i in range(0, len(s1stems)-1) :
        if i < len(s1stems) - 1:
            s2stem = s1stems[i] + " " + s1stems[i+1]
            s2stems.append(s2stem)
        if i < len(s1stems)-2:
            s3stem = s1stems[i] + " " + s1stems[i+1] + " " + s1stems[i+2]
            s3stems.append(s3stem)
            
    for i in range(0, len(t1stems)-1) :
        if i < len(t1stems) - 1:
            t2stem = t1stems[i] + " " + t1stems[i+1]
            t2stems.append(t2stem)
        if i < len(t1stems)-2:
            t3stem = t1stems[i] + " " + t1stems[i+1] + " " + t1stems[i+2]
            t3stems.append(t3stem)
                
    precision1stem = len(set(intersect(s1stems, t1stems))) / len(set(s1stems))
    recall1stem    = len(set(intersect(s1stems, t1stems))) / len(set(t1stems))
    f1stem = 0
    if (precision1stem + recall1stem) > 0:
        f1stem = 2 * precision1stem * recall1stem / (precision1stem + recall1stem)
    precision2stem = len(set(intersect(s2stems, t2stems))) / len(set(s2stems))
    recall2stem    = len(set(intersect(s2stems, t2stems))) / len(set(t2stems))
    f2stem = 0
    if (precision2stem + recall2stem) > 0:
        f2stem = 2 * precision2stem * recall2stem / (precision2stem + recall2stem)
    precision3stem = len(set(intersect(s3stems, t3stems))) / len(set(s3stems))
    recall3stem    = len(set(intersect(s3stems, t3stems))) / len(set(t3stems))
    f3stem = 0
    if (precision3stem + recall3stem) > 0:
        f3stem = 2 * precision3stem * recall3stem / (precision3stem + recall3stem)
	
    features["precision1stem"] = precision1stem
    features["recall1stem"] = recall1stem
    features["f1stem"] = f1stem
    features["precision2stem"] = precision2stem
    features["recall2stem"] = recall2stem
    features["f2stem"] = f2stem
    features["precision3stem"] = precision3stem
    features["recall3stem"] = recall3stem
    features["f3stem"] = f3stem

    return features


# read from train/test data files and create features
def readInData(filename):

    data = []
    trends = set([])
    
    (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = (None, None, None, None, None, None, None)
    
    for line in open(filename):
        line = line.strip()
        #read in training or dev data with labels
        if len(line.split('\t')) == 7:
            (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = line.split('\t')
        #read in test data without labels
        elif len(line.split('\t')) == 6:
            (trendid, trendname, origsent, candsent, origsenttag, candsenttag) = line.split('\t')
        else:
            continue
        
        #if origsent == candsent:
        #    continue
        
        trends.add(trendid)
        features = paraphrase_Das_features(origsent, candsent, trendname)
        
        if judge == None:
            data.append((features, judge, origsent, candsent, trendid))
            continue

        # ignoring the training/test data that has middle label 
        if judge[0] == '(':  # labelled by Amazon Mechanical Turk in format like "(2,3)"
            nYes = eval(judge)[0]            
            if nYes >= 3:
                amt_label = True
                data.append((features, amt_label, origsent, candsent, trendid))
            elif nYes <= 1:
                amt_label = False
                data.append((features, amt_label, origsent, candsent, trendid))   
        elif judge[0].isdigit():   # labelled by expert in format like "2"
            nYes = int(judge[0])
            if nYes >= 4:
                expert_label = True
                data.append((features, expert_label, origsent, candsent, trendid))
            elif nYes <= 2:
                expert_label = False
                data.append((features, expert_label, origsent, candsent, trendid))         
                
    return data, trends
    
# Evaluation by Precision/Recall/F-measure
# cut-off at probability 0.5, estimated by the model
def OneEvaluation():
            

    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    

    # read in training/test data with labels and create features
    trainfull, traintrends  = readInData(trainfilename)    
    testfull, testtrends  = readInData(testfilename)
    
    train = [(x[0], x[1]) for x in trainfull]
    test  = [(x[0], x[1]) for x in testfull]
    
    print "Read in" , len(train) , "valid training data ... " 
    print "Read in" , len(test) , "valid test data ...  "
    print 
    if len(test) <=0 or len(train) <=0 :
        sys.exit()

    
    # train the model 
    classifier = nltk.classify.maxent.train_maxent_classifier_with_megam(train, gaussian_prior_sigma=10, bernoulli=True)
    
    # uncomment the following lines if you want to save the trained model into a file
    
    #modelfile = './baseline_logisticregression.model'
    #outmodel = open(modelfile, 'wb')
    #dump(classifier, outmodel)
    #outmodel.close()
    
    # uncomment the following lines if you want to load a trained model from a file
    
    inmodel = open(modelfile, 'rb') 
    classifier = load(inmodel)
    inmodel.close()
    
    
        
    for i, t in enumerate(test):
    	
    	sent1 = testfull[i][2]
    	sent2 = testfull[i][3]
    	
    	guess = classifier.classify(t[0])
        label = t[1]
        if guess == True and label == False:
            fp += 1.0
        elif guess == False and label == True:
            fn += 1.0
        elif guess == True and label == True:
            tp += 1.0
        else:
            tn += 1.0  			

        if guess == True:
             print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
  
    print
    print "PRECISION: %s, RECALL: %s, F1: %s" % (P,R,F)
 
    print "ACCURACY: ", nltk.classify.accuracy(classifier, test)

    print "# true pos:", tp
    print "# false pos:", fp
    print "# false neg:", fn 
    print "# true neg:", tn    



# Evaluation by precision/recall curve
def PREvaluation():
    
    # read in training/test data with labels and create features
    trainfull, traintrends  = readInData(trainfilename)    
    testfull, testtrends  = readInData(testfilename)
    
    
    train = [(x[0], x[1]) for x in trainfull]
    test  = [(x[0], x[1]) for x in testfull]
    
    print "Read in" , len(train) , "valid training data ... " 
    print "Read in" , len(test) , "valid test data ...  "
    print
    if len(test) <=0 or len(train) <=0 :
        sys.exit()

    # train the model
    classifier = nltk.classify.maxent.train_maxent_classifier_with_megam(train, gaussian_prior_sigma=10, bernoulli=True)
    
    # comment the following lines to skip saving the above trained model into a file
    
    modelfile = './baseline_logisticregression.model'
    outmodel = open(modelfile, 'wb')
    dump(classifier, outmodel)
    outmodel.close()
    
    # comment the following lines to skip loading a previously trained model from a file
    
    inmodel = open(modelfile, 'rb') 
    classifier = load(inmodel)
    inmodel.close()
            
    probs = []
    totalpos = 0
        
    for i, t in enumerate(test):
    	prob = classifier.prob_classify(t[0]).prob(True)
    	probs.append(prob)
    	goldlabel = t[1]
    	if goldlabel == True:
            totalpos += 1
    	
    # rank system outputs according to the probabilities predicted
    sortedindex = sorted(range(len(probs)), key = probs.__getitem__)   
    sortedindex.reverse() 
    
    truepos = 0
    falsepos = 0
    
    print "\t\tPREC\tRECALL\tF1\t|||\tMaxEnt\tSENT1\tSENT2"
    
    i = 0
    for sortedi in sortedindex:
    	i += 1
    	strhit = "HIT"
    	
    	sent1 = testfull[sortedi][2]
    	sent2 = testfull[sortedi][3]
        if test[sortedi][1] == True:
            truepos += 1
        else:
            falsepos += 1
            strhit = "ERR"
            
        precision = truepos / (truepos + falsepos)
        recall = truepos / totalpos
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        
        
        print str(i) + "\t" + strhit + "\t" + "{0:.3f}".format(precision) + '\t' + "{0:.3f}".format(recall) + "\t" + "{0:.3f}".format(f1),
        print "\t|||\t" + "{0:.3f}".format(probs[sortedi]) + "\t" + sent1 + "\t" + sent2


# Load the trained model and output the predictions
def OutputPredictions(modelfile, outfile):
    
    # read in test data and create features
    testfull, testtrends  = readInData(testfilename)
    
    test  = [(x[0], x[1]) for x in testfull]
    
    print "Read in" , len(test) , "valid test data ...  "
    print
    if len(test) <=0:
        sys.exit()

	# read in pre-trained model    
    inmodel = open(modelfile, 'rb') 
    classifier = load(inmodel)
    inmodel.close()
           
    # output the results into a file
    outf = open(outfile,'w') 
          
    for i, t in enumerate(test):
        prob = classifier.prob_classify(t[0]).prob(True)
        if prob >= 0.5:
             outf.write("true\t" + "{0:.4f}".format(prob) + "\n")
        else:
             outf.write("false\t" + "{0:.4f}".format(prob) + "\n")
             
    outf.close()
             
if __name__ == "__main__":
    trainfilename = "../data/train.labelled.data"
    testfilename  = "../data/test.labelled.data"
    
    # Training and Testing by precision/recall curve
    #PREvaluation()
    
    # Training and Testing by Precision/Recall/F-measure
    OneEvaluation()
    
    # write results into a file in the SemEval output format
    outputfilename = "./PIT2015_BASELINE_02_lg.output"
    modelfilename = './baseline_logisticregression.model'
    OutputPredictions(modelfilename, outputfilename)
   