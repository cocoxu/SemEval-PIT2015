# ==============  SemEval-2015 Task 1  ==============
#  Paraphrase and Semantic Similarity in Twitter
# ===================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# a baseline system that completely use random outputs
#

import random

if __name__ == "__main__":

	testfilename = "../data/test.unlabelled.data"
	outfilename  = "./PIT2015_BASELINE_01_random.output"
	
	ntline = 0
	with open(testfilename) as tf:
		for tline in tf:
			tline = tline.strip()
			if len(tline.split('\t')) == 6:
				ntline += 1

    # output the results into a file
	outf = open(outfilename,'w') 
    	
	for x in range(ntline):
		score = random.random()		
		if score >= 0.5:
			outf.write("true\t" + "{0:.4f}".format(score) + "\n")
			#outf.write("true\t" + "0.0000" + "\n")
		else:
			outf.write("false\t" + "{0:.4f}".format(score) + "\n")
			#outf.write("false\t" + "0.0000" + "\n")
    
	outf.close()



