

# SemEval-2015 Task 1: Paraphrase and Semantic Similarity in Twitter (PIT)
  
Updated: Nov 10, 2020 

**The full pack of data and code for the official evaluation is at: .data/SemEval-PIT2015-github.zip**

## ORGANIZERS 

  * Wei Xu, University of Pennsylvania
  * Chris Callison-Burch, University of Pennsylvania
  * Bill Dolan, Microsoft Research


## RELEVANT PAPERS 

  paper about the dataset, baselines, and the MultiP model (multiple-instance learning paraphrase):
   
	@article{Xu-EtAl-2014:TACL,
	  author = {Xu, Wei and Ritter, Alan and Callison-Burch, Chris and Dolan, William B. and Ji, Yangfeng"}
	  title = {Extracting Lexically Divergent Paraphrases from {Twitter}},
	  journal = {Transactions of the Association for Computational Linguistics},
	  volume =  {2},
	  year =    {2014},
	  pages = {435--448},
	  publisher = {Association for Computational Linguistics},
	  url = {https://www.aclweb.org/anthology/Q14-1034}, 
	  doi = {10.1162/tacl_a_00194}
	}		    

   overview paper of the shared task:

	@inproceedings{xu2015semeval,
	  author    = {Xu, Wei and Callison-Burch, Chris and Dolan, Bill},
	  title     = {{SemEval-2015 Task} 1: Paraphrase and Semantic Similarity in {Twitter} ({PIT})},
	  booktitle = {Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)},
	  year      = {2015},
	  address.  = {Denver, Colorado},
          publisher = {Association for Computational Linguistics},
          url = {https://www.aclweb.org/anthology/S15-2001},
          doi = {10.18653/v1/S15-2001},
          pages = {1--11},
	}


## TRAIN/DEV/TEST DATA 
  

  The dataset contains the following files:
  
    ./data/train.data (13063 sentence pairs)
    ./data/dev.data   (4727 sentence pairs)
	./data/test.data  (972 sentences pairs)
	./data/test.label (a separate file of labels only, used by evaluation scripts)

  Both data files come in the tab-separated format. Each line contains 7 columns:
    
     Topic_Id | Topic_Name | Sent_1 | Sent_2 | Label | Sent_1_tag | Sent_2_tag |
 
  The "Topic_Name" are the names of trends provided by Twitter, which are not hashtags.
  
  The "Sent_1" and "Sent_2" are the two sentences, which are not necessarily full 
  tweets. Tweets were tokenized by Brendan O'Connor et al.'s toolkit (ICWSM 2010) 
  and split into sentences. 

  The "Sent_1_tag" and "Sent_2_tag" are the two sentences with part-of-speech 
  and named entity tags by Alan Ritter et al.'s toolkit (RANLP 2013, EMNLP 2011). 

  The "Label" column for *dev/train data * is in a format like "(1, 4)", which means 
  among 5 votes from Amazon Mechanical turkers only 1 is positive and 4 are negative.
  We would suggest map them to binary labels as follows:
    
    paraphrases: (3, 2) (4, 1) (5, 0)
    non-paraphrases: (1, 4) (0, 5)
    debatable: (2, 3)  which you may discard if training binary classifier

  The "Label" column for *test data* is in a format of a single digit between 
  between 0 (no relation) and 5 (semantic equivalence), annotated by expert.  
  We would suggest map them to binary labels as follows:
    
    paraphrases: 4 or 5
    non-paraphrases: 0 or 1 or 2  
    debatable: 3   which we discarded in Paraphrase Identification evaluation

  We discarded the debatable cases in the evaluation of Paraphrase Identification task,
  but kept them in the evaluation of Semantic Similarity task.  

## EVALUATION  

  There are two scripts for the official evaluation:
  
      ./scripts/pit2015_checkformat.py (checks the format or the system output file)
      ./scripts/pit2015_eval_single.py (evaluation metrics)


  The participants are required to produce a binary label (paraphrase) for each sentence 
  pair, and optionally a real number between 0 (no relation) and 1 (semantic equivalence)  
  for measuring semantic similarity.

  The system output file should match the lines of the test data. Each line has 2 columns 
  and separated by a tab in between, like this:
     | Binary Label (true/false) | Degreed Score (between 0 and 1, in the 4 decimal format) |
  if your system only gives binary labels, put "0.0000" in all second columns.  
  
  The output file names look like this:
      PIT2015_TEAMNAME_01_nameofthisrun.output 
      PIT2015_TEAMNAME_02_nameofthisrun.output      
    

          
## BASELINES & STATE-OF-THE-ART SYSTEMS 
  
  There are scripts for two baselines: 
  
    ./scripts/baseline_random.py
    ./scripts/baseline_logisticregression.py
  
  and their outputs on the test data, plus outputs from two state-of-the-art systems:
  
    ./systemoutputs/PIT2015_BASELINE_01_random.output
    ./systemoutputs/PIT2015_BASELINE_02_LG.output
    ./systemoutputs/PIT2015_BASELINE_03_WTMF.output
    ./systemoutputs/PIT2015_BASELINE_04_MultiP.output
  
  
  (1) The logistic regression (LG) model using simple lexical overlap features:
    
  It is our reimplementation in Python. This is a baseline originally 
  used by Dipanjan Das and Noah A. Smith (ACL 2009):
  "Paraphrase Identification as Probabilistic Quasi-Synchronous Recognition".

  To run the script, you will need to install NLTK and Megam packages:
    http://www.nltk.org/_modules/nltk/classify/megam.html
    http://www.umiacs.umd.edu/~hal/megam/index.html
    
  If you have troubles with Megam, you may need to rebuild it from source code:
    http://stackoverflow.com/questions/11071901/stuck-in-using-megam-in-python-nltk-classify-maxentclassifier

  Example output, if training on train.data and test on dev.data will look like:
    
    Read in 11513 training data ...  (after discarding the data with debatable cases)
    Read in 4139 test data ...       (see details in TRAIN/DEV DATA section)
    PRECISION: 0.704069050555
    RECALL:    0.389229720518
    F1:        0.501316944688
    ACCURACY:  0.725537569461 

  The script will provide the numbers for plotting precision/recall curves, or a 
  single precision/recall/F1 score with 0.5 cutoff of predicated probability. 


  (2) The Weighted Matrix Factorization (WTMF) model is a unsupervised approach
    developed by Weiwei Guo and Mona Diab (ACL 2012):
    "Modeling Sentences in the Latent Space"
    Its code is available at: http://www.cs.columbia.edu/~weiwei/code.html
  
  
  (3) The Multiple-instance Learning Paraphrase model (MultiP) is a supervised approach
    developed by Wei Xu et al. (TACL 2014):
	"Extracting Lexically Divergent Paraphrases from Twitter"
	Its code is available at: http://www.cis.upenn.edu/~xwe/multip/
