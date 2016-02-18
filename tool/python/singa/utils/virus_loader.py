import kvstore 
import sys
import pickle
import os

sys.path.append(os.path.join(os.path.dirname(__file__),'../../pb2'))
from common_pb2 import RecordProto

PREFIX='/home/dinhtta/antivirus'
TRAIN_FILE='train_data.bin'
TEST_FILE='test_data.bin'

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


# create training file
def create_train():
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

create_train()

