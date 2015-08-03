from PIL import Image
import glob, os, sys, random, shutil

size = 32,32

def imToBin(im):
  red = []
  green = []
  blue = []
  pix = im.load()
  for x in range(0,size[0]):
    for y in range(0,size[1]):
      pixel = pix[x,y]
      red.append(pixel[0])
      green.append(pixel[1])
      blue.append(pixel[2])     
  byteArray = bytearray(red+green+blue)
  return byteArray


def centerFileToBin(file):
  im = Image.open(file)
  byteArray1 = []
  byteArray2 = []
  oldSize = im.size
  #bigest center cube
  data=(0,0,0,0)
  if oldSize[0] < oldSize[1]:
    data= (0,(oldSize[1]-oldSize[0])/2,oldSize[0],(oldSize[1]+oldSize[0])/2)
  else :
    data= ((oldSize[0]-oldSize[1])/2,0,(oldSize[0]+oldSize[1])/2,oldSize[1])
  newIm = im.transform(size,Image.EXTENT,data)
  byteArray=imToBin(newIm) 
  return byteArray 
  
def extendFileToBin(file):
  im = Image.open(file)
  byteArray1 = []
  byteArray2 = []
  oldSize = im.size  
  data=(0,0,oldSize[0],oldSize[1])
  newIm = im.transform(size,Image.EXTENT,data)
  byteArray=imToBin(newIm) 
  return byteArray

if __name__ == "__main__":
   main()
