import os
from PIL import Image
trainImages = open("train.txt", "r")
valImages   = open("val.txt", "r")

def sendToCorrectFolder(folder, line ):
	imageName = line.split(" ")[0]
	birdImage     = Image.open(imageName)
	label     = line.split(" ")[1][0]
	if(int(label)==0):
		birdImage.save("BirdTerrainData/"+folder+"/land/"+imageName.split("/")[1])
	if(int(label)==1):
		birdImage.save("BirdTerrainData/"+folder+"/mixed/"+imageName.split("/")[1])
	if(int(label)==2):
		birdImage.save("BirdTerrainData/"+folder+"/water/"+imageName.split("/")[1])

for line in trainImages:
	sendToCorrectFolder("train",line)

for line in valImages:
	sendToCorrectFolder('val', line)
