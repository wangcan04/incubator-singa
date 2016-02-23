#!/usr/bin/python

# data pre-processing
# 1. read from the directory list
# 2. walk all the files
import os
import os.path as osp
import MySQLdb
import pickle
import numpy as np

DIRS=['../20151101', '../20151102', '../20151103'] #raw data directories
DATES=['2015-11-01', '2015-11-02', '2015-11-03']
DB=['localhost','root','','UniversalAvDb2'] #database login
DATE_START='2015-11-01'
DATE_END='2015-11-03'
PREFIX='.'  # prefix to processed sample directories
CLASS_0_DIR=PREFIX+'/0'
CLASS_1_DIR=PREFIX+'/1'
TRUNCATE_SIZE=50000  # size to truncate file
CACHED_DICT_FILE='.files_cache' # cached dictionary of filenames
CACHED_TRAIN_LIST='.train_files_cache' # cached of training files (name: label)
CACHED_TEST_LIST='.test_files_cache' # cached of testing files (name: label)
SPLIT_RATIO=0.9

#truncate file and move it to dst
def truncate_and_move_file(src, dst):
  fs = open(src, 'rb')
  fd = open(dst, 'wb')
  fd.write(fs.read(TRUNCATE_SIZE))
  fs.close()
  fd.close()

def move_files(fileList):
  # truncate and move to new directory
  percent = len(fileList)/10; 
  progress= 0 
  for k in fileList:
    truncate_and_move_file(k[3], k[4])
    progress=progress+1
    if progress % percent == 0:
      print (progress/percent)


# preprocess data
# 1. Move data samples from raw directory to PREFIX+'/0' and PREFIX+'/1' directories
# 2. Truncate each sample to 750000 (the original sizes are larger)
# 3. Return a dictionary: k = filename, v = '(class, date)'
def pre_process_data():
  if osp.exists(CACHED_DICT_FILE): #load directly from file
    f = open(CACHED_DICT_FILE,'rb')
    fileList = pickle.load(f)  
    f.close()
    return fileList 

  # connect to the database
  con = MySQLdb.connect(host=DB[0], user=DB[1], passwd=DB[2], db=DB[3])
  cur = con.cursor()

  # get a set of filenames for each class
  selection = 'select BinaryFilePath from Application where IsVirus=%d AND FileSize >=%d AND SubmissionDate >=\'%s\' AND SubmissionDate <=\'%s\''
  cur.execute(selection % (0, TRUNCATE_SIZE,
                              DATE_START,DATE_END)) 
  l1 = [osp.basename(x[0]) for x in list(cur) if x[0] != None] # extract basenames of class 0 
  cur.execute(selection % (1, TRUNCATE_SIZE,
                              DATE_START,DATE_END)) 
  l2 = [osp.basename(x[0]) for x in list(cur) if x[0] != None] # extract basenames of class 1 

  class0FileSet = set()
  class1FileSet = set()
  for x in l1:
    class0FileSet.add(x)
  for x in l2:
    class1FileSet.add(x)

  print ('From database: class 0 size = %d, class 1 size = %d' % (len(class0FileSet), len(class1FileSet)))

  # walk the directories of raw data, and move to new directory
  # create the dirs if not exists
  if (not osp.exists(CLASS_0_DIR)) or (not osp.exists(CLASS_1_DIR)):
    os.mkdir(CLASS_0_DIR)
    os.mkdir(CLASS_1_DIR)

  # create a list of (filename, class, submission_date)
  fileList=[]

  for idx, di in enumerate(DIRS):
    for (x,y,files) in os.walk(di):
      for f in files:
        if (f in class0FileSet):
          assert (f not in class1FileSet)
          fileList.append((f,0,DATES[idx], osp.join(x,f), osp.join(CLASS_0_DIR,f)))
        elif (f in class1FileSet):
          assert (f not in class0FileSet)
          fileList.append((f,1,DATES[idx], osp.join(x,f), osp.join(CLASS_1_DIR,f)))

  print ('From raw file directory, total size = %d ' % (len(fileList)))

  move_files(fileList)

  #fileList= [(x[0], x[1], x[2], x[4]) for x in fileList]

  f = open(CACHED_DICT_FILE, 'wb')
  pickle.dump(fileList, f)
  f.close()
  return fileList 


# split the sample set into training and test set
# each split will permute the original set
# stored in CACHED_TRAIN_LIST and CACHED_TEST_LIST 
def split_samples():
  if osp.exists(CACHED_TRAIN_LIST) and osp.exists(CACHED_TEST_LIST): #load directly from cache
    f1 = open(CACHED_TRAIN_LIST)
    f2 = open(CACHED_TEST_LIST)
    trainList = pickle.load(f1)
    testList = pickle.load(f2)
    f1.close()
    f2.close()
    print ('train list size = %d, test list size = %d' % (len(trainList), len(testList)))
    print(trainList[0], testList[0])

    return (trainList, testList)

  fileList = pre_process_data()
  size = len(fileList)
  ntrain = size*SPLIT_RATIO

  trainList=[] # (file, label, date) tuples
  testList=[] # (file, label, date) tuples

  idx = np.random.permutation(size)
  for i in range(size):
    if (ntrain>=0):
      trainList.append(fileList[idx[i]])
      ntrain = ntrain-1
    else:
      testList.append(fileList[idx[i]])

  f1 = open(CACHED_TRAIN_LIST, 'wb')
  f2 = open(CACHED_TEST_LIST, 'wb')
  pickle.dump(trainList, f1)
  pickle.dump(testList, f2)
  f1.close()
  f2.close()
  print ('train list size = %d, test list size = %d' % (len(trainList), len(testList)))
  print(trainList[0], testList[0])
  return (trainList, testList)

split_samples()
