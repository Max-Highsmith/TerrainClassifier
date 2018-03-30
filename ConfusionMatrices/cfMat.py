import numpy as np
import matplotlib.pyplot as plt
import copy

allModels = ['alexNet', 'densenet_201', 'resnet18', 'vgg19', 'vgg19_bn']
for model in allModels:
	confusionMatrix = np.loadtxt(model+'.confuse')
	colSumsTotals = [0,0,0]
	HeatMatrix = copy.deepcopy(confusionMatrix)
	for i in range(0,3):
		for j in range(0,3):
			colSumsTotals[j] += confusionMatrix[i,j]
	for i in range(0,3):
		for j in range(0,3):
			HeatMatrix[i,j] = confusionMatrix[i,j]/colSumsTotals[j] 
#		a = 0
#		tmp_arr = []
#		a = sum(i, 1)
#		for j in i:
#			tmp_arr.append(float(j)/float(a))
#		norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(HeatMatrix, cmap=plt.cm.Reds, 
			interpolation='nearest')

	width, height = confusionMatrix.shape

	for x in range(width):
		for y in range(height):
			ax.annotate(str(confusionMatrix[x][y]), xy=(y, x), 
				    horizontalalignment='center',
				    verticalalignment='center')

	cb = fig.colorbar(res)
	plt.xlabel("Ground Truth")
	plt.ylabel("Prediction")
	plt.title(model+" Confusion Matrix Validation")
	plt.xticks(range(width), ['land', 'both', 'water'])
	plt.yticks(range(height),['land', 'both', 'water'] )
	plt.savefig(model+'_confusion_matrix.png', format='png')
	plt.show()
