import pandas as pd
import numpy as np
import cv2

f1 = pd.read_csv('train.csv').values
f2 = pd.read_csv('test.csv').values

for i in f1:
	img=cv2.imread('data/Training/'+str(int(i[0])+1)+'.jpg')
	cv2.imshow('image',img)
	cv2.waitKey(1)
cv2.destroyAllWindows()
	


