import kvstore 
import sys
import pickle
import os
import pefile
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'../../pb2'))
from common_pb2 import RecordProto

PREFIX='/home/dinhtta/antivirus'
TRAIN_FILE='train_data.bin'
TEST_FILE='test_data.bin'
TRAIN_FILE_CODE='train_data_code.bin' # extract .text segment from binary
TEST_FILE_CODE='test_data_code.bin'
TRUNCATE_SIZE=50000
SPLIT_RATIO=0.9
CACHE_FILE='.cache_list'

# load cached list of test and train file
def load_cache():
  f = open(os.path.join(PREFIX, '.train_files_cache'), 'rb')
  trainList = pickle.load(f)
  f.close()
  f = open(os.path.join(PREFIX, '.test_files_cache'), 'rb')
  testList = pickle.load(f)
  f.close()

  return (trainList, testList)


def get_content(path): #read completely from file
  f = open(path, 'rb')
  content = f.read()
  f.close()
  return content

def get_text_segment(path):
  try:
    pe = pefile.PE(path)
    for section in pe.sections:
      if (section.Name.startswith('.text') and section.SizeOfRawData>=TRUNCATE_SIZE):
        content = section.get_data()[:TRUNCATE_SIZE]
        return content
  except:
    print ('Error at initializing PE file for %s' % (path))
    pass
  return None; 

# create training and test file
def create_train_and_test():
  (trainList, testList) = load_cache()

  trainStore = kvstore.FileStore()
  trainStore.open(TRAIN_FILE, "create")
  testStore = kvstore.FileStore()
  testStore.open(TEST_FILE, "create")

 
  tenpc=len(trainList)/10
  count=0
  for idx,sample in enumerate(trainList):
    trainRecord = RecordProto()
    trainRecord.label = sample[1]
    trainRecord.pixel = get_content(os.path.join(PREFIX, sample[4]))
    value = trainRecord.SerializeToString()
    key = "%d" %idx
    trainStore.write(key,value)
    if (idx % tenpc==0):      
      print (count)
      count = count + 1

  trainStore.close()

  tenpc = len(testList)/10
  count = 0
  for idx,sample in enumerate(testList):
    testRecord = RecordProto()
    testRecord.label = sample[1]
    testRecord.pixel = get_content(os.path.join(PREFIX, sample[4]))
    value = testRecord.SerializeToString()
    key = "%d" %idx
    testStore.write(key,value)
    if (idx % tenpc == 0):
      print (count)
      count = count + 1

  testStore.close()

# create training and test file from .text segment
# go through all files in .files_cache
def create_code_sets():

  if (os.path.exists(CACHE_FILE)):
    code_samples = pickle.load(open(CACHE_FILE, 'rb')) 
  else:
    f=open(CACHE_FILE,'wb')
    # first, extract a list of all files with .text segment >= 50K
    samples = pickle.load(open(os.path.join(PREFIX, '.files_cache'), 'rb'))
    code_samples=[] # list of (path,label)
    for sample in samples:
      try:
        pe = pefile.PE(os.path.join(PREFIX,sample[4]))
        for section in pe.sections:
          if (section.Name.startswith('.text') and section.SizeOfRawData>=TRUNCATE_SIZE):
           code_samples.append((sample[3], sample[1]))
           break
      except:
        pass
    pickle.dump(code_samples,f)
    f.close()

    
  print('exit... len = %d' %(len(code_samples)))
  permu = np.random.permutation(len(code_samples))
  split_idx = len(code_samples)*SPLIT_RATIO

  trainStore = kvstore.FileStore()
  trainStore.open(TRAIN_FILE_CODE, "create")
  testStore = kvstore.FileStore()
  testStore.open(TEST_FILE_CODE, "create")

  tenpc=len(code_samples)/10
  count=0
  print('split idx = %d' % (split_idx))
  for idx,sample in enumerate(code_samples):
    if (idx<split_idx):
      trainRecord = RecordProto()
      trainRecord.label = code_samples[permu[idx]][1]
      content =  get_text_segment(os.path.join(PREFIX, code_samples[permu[idx]][0]))
      if (content != None):
        trainRecord.pixel = content
        value = trainRecord.SerializeToString()
        key = "%d" %idx
        trainStore.write(key,value)
    else:
      testRecord = RecordProto()
      testRecord.label = code_samples[permu[idx]][1]
      content = get_text_segment(os.path.join(PREFIX, code_samples[permu[idx]][0]))
      if (testRecord.pixel != None):
        testRecord.pixel = content
        value = testRecord.SerializeToString()
        key = "%d" %idx
        testStore.write(key,value)
    if (idx % tenpc==0):      
      print (count)
      count = count + 1

  trainStore.close()
  testStore.close()

#create_train_and_test()
print('creating dataset of .text segments')
create_code_sets()
