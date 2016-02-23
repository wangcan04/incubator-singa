#!/usr/bin/python

# read through .files_cache and produce the list of value and percent, to be rendered by CDF
# each class has serveral lists: total size, size of .text, sizeof .data

import os, os.path
import pickle
import sys
import pefile

def size_list(fileList):
  sizes=[]
  textsizes=[]
  datasizes=[]
  count=len(fileList)/100
  notpe_count = 0
  for (idx,f) in enumerate(fileList):
    sizes.append(os.stat(f[3]).st_size)
    try:
      pe = pefile.PE(f[3])
      for section in pe.sections:
        if section.Name.startswith('.text'):
          textsizes.append(section.SizeOfRawData)
        if section.Name.startswith('.data'):
          datasizes.append(section.SizeOfRawData)
    except:
      #print ('not PE file %s' % (f[3]))
      notpe_count = notpe_count + 1
    if (idx%count==0):
      print ('progress ... %d' % (idx))

  print ('total non-PE files = %d' % (notpe_count))

  sizes.sort()
  textsizes.sort()
  datasizes.sort()
  return (sizes, textsizes, datasizes)

# read the file
if (os.path.exists('.sizes_cache')):
  fs = open('.sizes_cache','rb') 
  (total_sizes_c0, text_sizes_c0, data_sizes_c0, total_size_c1, text_sizes_c1, data_sizes_c1) = pickle.load(fs)
else:
  f = open('.files_cache', 'rb')
  fileList = pickle.load(f)
  list_class0 = [x for x in fileList if x[1]==0]
  list_class1 = [x for x in fileList if x[1]==1]
  (total_sizes_c0, text_sizes_c0, data_sizes_c0) = size_list(list_class0)
  (total_sizes_c1, text_sizes_c1, data_sizes_c1) = size_list(list_class1)
  fs = open('.sizes_cache','wb')
  pickle.dump((total_sizes_c0, text_sizes_c0, data_sizes_c0, total_sizes_c1, text_sizes_c1, data_sizes_c1), fs)
  print (len(total_sizes_c0), len(text_sizes_c0), len(data_sizes_c0))
  print (len(total_sizes_c1), len(text_sizes_c1), len(data_sizes_c1))
  fs.close()
  f.close()
