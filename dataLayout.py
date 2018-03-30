import os
import sys
import numpy as np
import matplotlib.pyplot as plt

MAIN_DATA_FILE = "./BirdTerrainData/"


collection = []
dataStages= os.listdir(MAIN_DATA_FILE)
numDataStages = len(dataStages)
numClasses =0;
label_classes =0
for stage in dataStages:
	label_classes = os.listdir(MAIN_DATA_FILE+stage)
	numClasses    = len(label_classes)
	for lc in label_classes:
		collection.append(len(os.listdir(MAIN_DATA_FILE+stage+'/'+lc)))
	


print(collection)	
PALLETE=['y','g', 'blue', 'k', 'maroon']

for i in range(0,numDataStages):
	plt.clf()
	plt.bar(range(0,numClasses), collection[i*numClasses:(i*numClasses)+numClasses], color=PALLETE)
	print(label_classes)
	plt.xticks(np.arange(numClasses), label_classes)
	plt.ylabel('Num Images')
	plt.title(dataStages[i])
	for lc in range(0, numClasses):
		plt.annotate(collection[(i*numClasses+lc)], xy=(lc,10), horizontalalignment='center', verticalalignment='center')
	plt.show()
