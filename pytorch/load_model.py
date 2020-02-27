import numpy as np
import torch
from torch import nn

from image_loader import image_loader
from preprocess import preprocess_image_1


gpu_name = 'cuda:1'
gpu_flag = True

path_model = '/home/adarsh/project/CDAN/pytorch/snapshot/san/best_model.pth.tar'

model = torch.load(path_model)
# model = nn.Sequential(model)
# torch.save(model.statedict(),'/home/adarsh/project/disentanglement/resnet_50_da.pt')
model.cuda(gpu_name)
model.eval()
print(model)
path_quick_draw = '/home/adarsh/project/CDAN/pytorch/dataset/QuickDraw_sketches_final/'

path_class_list = '/home/adarsh/project/CDAN/pytorch/common_class_list.txt'

class_list = np.loadtxt(path_class_list,dtype='str')

quick_draw = image_loader(path_quick_draw,folder_list=class_list, split=[0,0.5,0])

criterion = nn.CrossEntropyLoss()

def entropy(arr):
    ent = []
    for i in arr:
        exp = np.exp(i)
        exp = exp/np.sum(exp)
        ent.append( np.sum( -1 * np.log(exp+0.000001)*exp)/np.log(2))
    return np.array(ent)

# print(entropy(np.array([[1,2,-1],[2,2,2]])))

feature_array = []
loss_array = []
correct_classified = []
label_array = []

classifier_weights = model.state_dict()
fc_w = classifier_weights['0.fc.weight']
fc_b = classifier_weights['0.fc.bias']
bn_w = classifier_weights['0.bottleneck.weight']
bn_b = classifier_weights['0.bottleneck.bias']



for (images, labels) in quick_draw.image_gen(split_type='val'):

    images = preprocess_image_1( array = images,
                                split_type = 'val',
                                use_gpu = True, gpu_name= gpu_name  )

    labels = torch.tensor(labels,dtype=torch.long)

    if(gpu_flag == True):
        labels = labels.cuda(gpu_name)

    features, preds = model(images)
    
    # loss = criterion(preds, labels).item()
    _, predicted = torch.max(preds.data,1)
    loss = entropy(preds.cpu().detach().numpy())
    correct = (predicted == labels).cpu().detach().numpy()
    features = features.cpu().detach().numpy()

    # print(loss)
    # print(correct)

    for f, l, c, lb in zip( features, loss, correct, labels):
        feature_array.append(f)
        loss_array.append(l)
        correct_classified.append(c)
        label_array.append(lb)
    
    # total += labels.size(0)
    # correct += (predicted == labels).sum().item()

np.savetxt('loss_array.txt',np.array(loss_array))
np.savetxt('correct_classified.txt',np.array(correct_classified))
np.savetxt('label_array.txt',np.array(label_array))
np.savetxt('feature_array.txt',np.array(feature_array))
classifier_weights = {'fc_w':fc_w,'fc_b':fc_b,'bn_w':bn_w,'bn_b':bn_b}

for i in classifier_weights:
    classifier_weights[i] = classifier_weights[i].cpu().detach().numpy()

import pickle
with open('classifier_weights','wb') as f:
    pickle.dump(classifier_weights,f)
