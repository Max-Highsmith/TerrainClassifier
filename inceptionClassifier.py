

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()


data_transforms = {
        'train': transforms.Compose([
                transforms.RandomResizedCrop(299),  
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.224])
        ]),
        'val': transforms.Compose([
                transforms.Resize(299),                
                transforms.CenterCrop(299),             
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, -.224, 0.225])
        ]),
}

data_dir = "BirdTerrainData"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
					data_transforms[x])
		for x in ['train', 'val']}

dataloaders     = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
						shuffle = True, num_workers=2)
		for x in ['train', 'val']}

testLoader     = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=2)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names= image_datasets['train'].classes
use_gpu = torch.cuda.is_available()
print(class_names)
print(dataset_sizes)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, model_name="name"):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    trainCrossEntropyCurve = []
    trainAccuracyCurve     = []
    valCrossEntropyCurve   = []
    valAccuracyCurve       = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                print(outputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
                

               #store curve
            if phase =='train':
               trainCrossEntropyCurve.append(epoch_loss) 
               trainAccuracyCurve.append(epoch_acc)
            if phase =='val':
               valCrossEntropyCurve.append(epoch_loss)
               valAccuracyCurve.append(epoch_acc)    
                   
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    #save train/validation curve and Best model
    T_V_F= "trainValidationCurveData/"
    np.savetxt(T_V_F+ model_name+"valLoss.data", valCrossEntropyCurve) 
    np.savetxt(T_V_F+ model_name+"valAcc.data", valAccuracyCurve) 
    np.savetxt(T_V_F+ model_name+"trainLoss.data", trainCrossEntropyCurve) 
    np.savetxt(T_V_F+ model_name+"trainAcc.data", trainAccuracyCurve) 
    torch.save(model.state_dict(), "BestModels/"+model_name)
    return model



#finetuning the convnet


#Models
#AlexNet
'''
alexnet       = models.alexnet(pretrained=True)
alexFeatures  = alexnet._modules['features']
alexClassifier    = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(9216, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, 3))

alexClassifier[1].load_state_dict(alexnet._modules['classifier'][1].state_dict())
alexClassifier[4].load_state_dict(alexnet._modules['classifier'][4].state_dict())
alexnet._modules['classifier'] = alexClassifier

#Vgg-19
vgg19 = models.vgg19(pretrained=True)
vgg19.classifier._modules['6']=nn.Linear(4096, 3)

#Vgg-19 batch norm
vgg19 = models.vgg19_bn(pretrained=True)
vgg19.classifier._modules['6']=nn.Linear(4096,3)

#Resnet18
resnet18    = models.resnet18(pretrained=True)
num_ftrs    = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs,3)

#Resnet-152

#SqueezeNet 1.1

#Densenet-201
densenet_201 = models.densenet201(pretrained=True)
densenet_201._modules['classifier'] = nn.Linear(1920, 3)
'''
#Inception v3
inception = models.inception_v3(pretrained=True)
inception._modules['fc'] = nn.Linear(2048, 3)


if use_gpu:
#	resnet18       = resnet18.cuda()
#	alexnet        = alexnet.cuda()	
#	vgg19          = vgg19.cuda()
#	vgg19_bn       = vgg19.cuda()
#	densenet_201   = densenet_201.cuda()
	inception      = inception.cuda()

#########################333
##Loss functions, optimizers and lr schedulers
#############################
'''
alexcrit              = nn.CrossEntropyLoss()
alexopt               = optim.SGD(alexnet._modules['classifier'][6].parameters(), lr=0.001, momentum=0.9)
alexexp_lr_scheduler  = lr_scheduler.StepLR(alexopt, step_size=7, gamma=0.1)

res18crit             = nn.CrossEntropyLoss()
res18opt              = optim.SGD(resnet18._modules['fc'].parameters(), lr=0.001, momentum=0.9)
res18exp_lr_scheduler = lr_scheduler.StepLR(res18opt, step_size=7, gamma=0.1)

vgg19crit             = nn.CrossEntropyLoss()
vgg19opt              = optim.SGD(vgg19._modules['classifier'][6].parameters(), lr= 0.001, momentum=0.9)
vgg19exp_lr_scheduler = lr_scheduler.StepLR(vgg19opt, step_size=6, gamma=0.1)

vgg19_bn_crit             = nn.CrossEntropyLoss()
vgg19_bn_opt              = optim.SGD(vgg19_bn._modules['classifier'][6].parameters(), lr= 0.001, momentum=0.9)
vgg19_bn_exp_lr_scheduler = lr_scheduler.StepLR(vgg19_bn_opt, step_size=6, gamma=0.1)

densenet_201_crit             = nn.CrossEntropyLoss()
densenet_201_opt              = optim.SGD(densenet_201._modules['classifier'].parameters(), lr=0.001, momentum=0.9)
densenet_201_exp_lr_scheduler = lr_scheduler.StepLR(densenet_201_opt, step_size=6, gamma=0.1)
'''
inception_crit            = nn.CrossEntropyLoss()
inception_opt             = optim.SGD(inception._modules['fc'].parameters(), lr=0.001, momentum=0.9)
inception_exp_lr_scheduler= lr_scheduler.StepLR(inception_opt, step_size=6, gamma=0.1)





#Train Models
#alex_Best         = train_model(alexnet, alexcrit, alexopt, alexexp_lr_scheduler, num_epochs=1, model_name="alexNet")
#resnet18_Best     = train_model(resnet18, res18crit, res18opt, res18exp_lr_scheduler, num_epochs=1, model_name="resnet18")
#vgg19_Best        = train_model(vgg19, vgg19crit, vgg19opt, vgg19exp_lr_scheduler, num_epochs=1, model_name="vgg19")
#vgg19_bn_Best     = train_model(vgg19_bn, vgg19_bn_crit, vgg19_bn_opt, vgg19_bn_exp_lr_scheduler, num_epochs=1, model_name="vgg19_bn")
#densenet_201_Best = train_model(densenet_201, densenet_201_crit, densenet_201_opt, densenet_201_exp_lr_scheduler, num_epochs=1, model_name="densenet")
inception_Best    =  train_model(inception, inception_crit, inception_opt, inception_exp_lr_scheduler, num_epochs=1, model_name="inception") 
       
#resnet18_model= train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)


def confusionMatrix(model):
    confusion = torch.IntTensor([[0,0,0],[0,0,0],[0,0,0]])
    for data in testLoader:
        inputs, labels  = data
        if use_gpu:
             inputs = Variable(inputs.cuda())
             labels = labels.cuda()
        else:
             inputs, labels =  Variable(inputs), Variable(labels)
 
        outputs  = model(inputs)
        _, preds = torch.max(outputs.data, 1)	

        confusion[int(preds.cpu().numpy()),int(labels.cpu().numpy())] += 1
    return confusion 


#ModelsToConfuse       = [alex_Best, resnet18_Best, vgg19_Best, vgg19_bn_Best, densenet_201_Best ]
ModelsToConfuse       = [inception_Best]
#ModelsToConfuseNames  = ['alexNet', 'resnet18', "vgg19", "vgg19_bn", "densenet_201"]
ModelsToConfuseNames  = ['inception_v3']
for i, model in enumerate(ModelsToConfuse):
    confused = confusionMatrix(model)
    name     = ModelsToConfuseNames[i]
    print(confused)
    np.savetxt("ConfusionMatrices/" + name + ".confuse", confused.numpy())



