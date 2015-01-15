# ==============  SemEval-2015 Task 1  ==============
#  Paraphrase and Semantic Similarity in Twitter
# ===================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# use this script to check output format before submitting
#
# To run:
# > python pit2015_checkformat.py sampletestdata samplesystemoutput

import sys
import re

# return True if pass the check
def CheckFileFormat(testlabelfile, outputfile):
	anyerror = False 
	rfilename = re.compile(r'PIT2015_[A-Z\-]{2,8}_0[1-2]_[a-z0-9]{2,10}.output')

	if not rfilename.match(outputfile):
		anyerror = True
		return anyerror

	### check the content of the output file ###

	rdecimal = re.compile("^-?[0-9]\.[0-9][0-9][0-9][0-9]$")

	ntline = 0
	with open(testlabelfile) as tf:
		for tline in tf:
			tline = tline.strip()
			if len(tline.split('\t')) == 2:
				ntline += 1

	noline = 0	
	oline_count = 0		
	with open(outputfile) as of:
		for oline in of:
			oline = oline.strip()
			oline_count += 1
			ocolumns = oline.split('\t')
			if len(ocolumns) == 2:
				noline += 1
				if ocolumns[0] != "true" and ocolumns[0] != "false":
					anyerror = True
				if not rdecimal.match(ocolumns[1]):
					anyerror = True
				else:
					score = float(ocolumns[1])
					if score < 0 or score > 1:
						anyerror = True
			else:
				anyerror = True			
			
	if ntline != noline:
		anyerror = True
	
	return not anyerror	


# print messages of format checking
def CheckFormat(testdatafile, outputfile):
	anyerror = False 

	###  check the name of the output file  ###
	# it has to be named as PIT2015_XXXXX_dd_xxxxxx.output 
	# XXXXX is the team name,  2-8 characters of English letters in uppercase or '-'
	# dd is the index of the runs, can only be 01 or 02 or 03
	# xxxxx is the name of run, 2-10 characters of English letters in lowercase or digits

	rfilename = re.compile(r'PIT2015_[A-Z\-]{2,8}_0[1-3]_[a-z0-9]{2,10}.output')

	if not rfilename.match(outputfile):
		anyerror = True
		print "The name of the output file doesn't meet the requirements:"
		print
		print "    it must be named as PIT2015_XXXXX_dd_xxxxxx.output"
		print "    XXXXX is your team name, must be 2-8 characters of English letters in uppercase or '-'"
		print "    dd is the index of the runs, can only be 01 or 02"
		print "    xxxxx is the name of this run, must be 2-10 characters of English letters in lowercase or digits"
		print 
		print "Sorry! The output file didn't pass the format check."
		print
		return anyerror

	### check the content of the output file ###

	rdecimal = re.compile("^[0-9]\.[0-9][0-9][0-9][0-9]$")

	ntline = 0
	with open(testdatafile) as tf:
		for tline in tf:
			tline = tline.strip()
			if len(tline.split('\t')) == 6 or len(tline.split('\t')) == 2:
				ntline += 1

	noline = 0	
	oline_count = 0		
	with open(outputfile) as of:
		for oline in of:
			oline = oline.strip()
			oline_count += 1
			ocolumns = oline.split('\t')
			if len(ocolumns) == 2:
				noline += 1
				if ocolumns[0] != "true" and ocolumns[0] != "false":
					anyerror = True
					print "Error in line " + str(oline_count) + " (1st column must be \"true\" or \"false\")"
				if not rdecimal.match(ocolumns[1]):
					anyerror = True
					print "Error in line " + str(oline_count) + " (2nd column must be a decimal in X.XXXX format)"
				else:
					score = float(ocolumns[1])
					if score < 0 or score > 1:
						anyerror = True
						print "Error in line " + str(oline_count) + " (2nd column must be a decimal between 0 and 1)"

			else:
				anyerror = True			
				print "Error in line " + str(oline_count) + " (must have 2 columns seperated by tab)"
			
	if ntline != noline:
		anyerror = True
		print "Error (the total number of lines in the output file does not match with test data)"
	
	print
	if anyerror == False:
		print "Congratulations! The output file has passed the format check."
	else:
		print "Sorry! The output file didn't pass the format check."
	print

	return anyerror	



if __name__ == "__main__":

	CheckFormat(sys.argv[1], sys.argv[2])