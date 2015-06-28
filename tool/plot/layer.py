from common_pb2 import LabelFeatureProto
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib
import os
import sys
matplotlib.rcParams.update({'font.size': 17})
import numpy as np

def make_N_colors(cmap_name, N):
  cmap = cm.get_cmap(cmap_name, N)
  return cmap(np.arange(N))


def plot2d(x, y, label, fname):
  plt.clf()
  color=make_N_colors('gist_rainbow', 10)
  marker=["o","x","d","s","+","v",">","p",'*','.']
  count=[0]*10
  for i,lb in enumerate(label):
    plt.scatter(x[i],y[i], marker=marker[lb], s=40, c='w', edgecolor=color[lb])
    count[lb]+=1
  print count
  #plt.show()
  plt.savefig(fname)


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print 'Usage: python plot.py <path.dat>'
    print 'the program will generate a picture at <path.jpg>'
    sys.exit()
  inputfile = sys.argv[1]
  print inputfile
  proto=LabelFeatureProto();
  fd=open(inputfile, 'rb')
  proto.ParseFromString(fd.read())
  label=proto.label
  x=proto.x
  y=proto.y
  paths=os.path.splitext(inputfile)
  plot2d(x,y,label,paths[0]+'.jpg')
