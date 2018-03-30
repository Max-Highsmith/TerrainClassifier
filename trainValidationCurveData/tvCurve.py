import matplotlib.pyplot as plt
import numpy as np

modelNames = ['alexNet', 'densenet','resnet18', 'vgg19_bn', 'vgg19', 'inception']
evalTypes  = ['Acc', 'Loss']
dataStep   = ['train', 'val']

for mn in modelNames:
	for et in evalTypes:
		for ds in dataStep:
			xAxis = np.loadtxt(mn+ds+et+".data")
			plt.plot(xAxis)
		plt.ylabel(et)
		plt.xlabel('epoch')
		plt.title(mn+" Training and Validation")
		plt.show()
		plt.savefig(mn+et+"_Train_Val_Curve.png")
			
