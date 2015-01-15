# ==============  SemEval-2015 Task 1  ==============
#  Paraphrase and Semantic Similarity in Twitter
# ===================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# This is the unofficial evaluation script. 
# We may change or add some metrics (e.g. area under P/R curve) for the final evaluation.
#
# To run:
# > python pit2015_eval_single.py ../data/test.label ../systemoutputs/PIT2015_BASELINE_02_LG.output
#
# The output looks like:
#    size_of_test_data|name_of_the_run||F1|Precision|Recall||Pearson|maxF1|mPrecision|mRecall
#
#    F1|Precision|Recall                 are for binary outputs
#    Pearson|maxF1|mPrecision|mRecall    are for degreed outputs
#

from __future__ import division
import sys
import re
import math
from os import listdir
from os.path import isfile, join
from pit2015_checkformat import CheckFileFormat



def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
        
    if xdiff2 == 0 or ydiff2 == 0:
    	return 0.0

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def EvalSingleSystem(testlabelfile, outputfile):

	# read in golden labels
	goldlabels = []
	goldscores = []
	
	hasscore = False
	with open(testlabelfile) as tf:
		for tline in tf:
			tline = tline.strip()
			tcols = tline.split('\t')
			if len(tcols) == 2:
				goldscores.append(float(tcols[1]))
				if tcols[0] == "true":
					goldlabels.append(True)
				elif tcols[0] == "false":
					goldlabels.append(False)
				else:
					goldlabels.append(None)

	
	
	
	# read in system labels
	syslabels = []
	sysscores = []
	
	with open(outputfile) as of:
		for oline in of:
			oline = oline.strip()
			ocols = oline.split('\t')
			if len(ocols) == 2:
				sysscores.append(float(ocols[1]))
				if float(ocols[1]) > 0.001 :
					hasscore = True
				if ocols[0] == "true":
					syslabels.append(True)
				elif ocols[0] == "false":
					syslabels.append(False)

	# evaluation metrics
	
	# system binary labels vs golden binary labels
	# F1 / Precision / Recall
	sysoutputs = []
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for (i,syslabel) in enumerate(syslabels):
	
		if syslabel == True and goldlabels[i] == True:
			tp += 1
		elif syslabel == True and goldlabels[i] == False:
			fp += 1
		elif syslabel == False and goldlabels[i] == False:
			tn += 1
		elif syslabel == False and goldlabels[i] == True:
			fn += 1
		
	P = tp / (tp + fp)
	R = tp / (tp + fn)
	F = 2 * P * R / (P + R)
	
	testsize = str(tp + fn + fp + tn)

	bar = "\t"
	none = "---"
	matches = re.match(r'.*PIT2015_([^_]*)_(.*).output', outputfile)
	teamname = matches.groups(1)[0]
	nameofrun = matches.groups(1)[1]

	if hasscore == False:
		evalresult = testsize + bar + teamname + bar + nameofrun + bar + bar + "{0:.3f}".format(F) + bar + "{0:.3f}".format(P) + bar + "{0:.3f}".format(R) \
			+ bar + bar + none + bar + none + bar + none + bar + none
		return evalresult

	# system degreed scores vs golden binary labels
	# maxF1 / Precision / Recall  
    
	maxF1 = 0
	P_maxF1 = 0
	R_maxF1 = 0
    
    # rank system outputs according to the probabilities predicted
	sortedindex = sorted(range(len(sysscores)), key = sysscores.__getitem__)   
	sortedindex.reverse() 

	truepos  = 0
	falsepos = 0
	
	for sortedi in sortedindex:
		if goldlabels[sortedi] == True:
			truepos += 1
		elif goldlabels[sortedi] == False:
			falsepos += 1
            
		precision = 0
		
		if truepos + falsepos > 0:
			precision = truepos / (truepos + falsepos)
		
		recall = truepos / (tp + fn)
		f1 = 0

		#print precision, recall
		    
		if precision + recall > 0:
			f1 = 2 * precision * recall / (precision + recall)
			if f1 > maxF1:
				maxF1 = f1
				P_maxF1 = precision
				R_maxF1 = recall

	# system degreed scores  vs golden degreed scores 
	# Pearson correlation
	
	
	pcorrelation = pearson(sysscores, goldscores)
		
	evalresult = testsize + bar + teamname + bar + nameofrun + bar + bar + "{0:.3f}".format(F) + bar + "{0:.3f}".format(P) + bar + "{0:.3f}".format(R) \
		+ bar + bar + "{0:.3f}".format(pcorrelation) + bar + "{0:.3f}".format(maxF1) + bar + "{0:.3f}".format(P_maxF1) + bar + "{0:.3f}".format(R_maxF1) 
		
		
	return evalresult
	

def PITEval(labelfile, outfile):
	if CheckFileFormat(labelfile, outfile):
		return EvalSingleSystem(labelfile, outfile)
	else:
		print "System output file format error: " + outputfile

	return

# Evaluate all system outputs 
#
# if __name__ == "__main__":
# 
# 	mypath = "./systemoutputs/"
# 	testlabelfile = "./data/test.label"
# 	onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
# 	for filename in onlyfiles:
# 		if re.match(r".*\.output$", filename):
# 			outputfile = mypath + "/" + filename
# 			print PITEval(testlabelfile, outputfile)

# Evaluate a single system output
if __name__ == "__main__":

	testlabelfile = sys.argv[1]
	outputfile    = sys.argv[2]
	
	print PITEval(testlabelfile, outputfile)