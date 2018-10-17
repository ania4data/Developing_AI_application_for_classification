# Imports here

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,utils, models
from torch.utils.data import Dataset, DataLoader
#import helper

import matplotlib.pyplot as plt
from collections import OrderedDict

import json
from PIL import Image
import time
import argparse
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#from workspace_utils import active_session

start_time=time.time()





parser = argparse.ArgumentParser(description='Deep Learning App')

#parser.add_argument('data_dir', action="store", help="Data directory")
#parser.add_argument('--save_dir', action="store", dest="save_dir",help="Saving directory")

parser.add_argument('image_dir', action="store", default="flowers/test/35/image_06984.jpg", help="path to image to get prediction") #
parser.add_argument('checkpoint_dir', action="store", default="checkpoint_adam_resnet152_10epoch_opti_name.pth",help="path to the checkpoint pth file") #
parser.add_argument('--category_names',     action="store", dest="json_file", default="cat_to_name.json",   help="category to name file path")
#parser.add_argument('--arch',     action="store", dest="archh", default="resnet152",   help="Model Architecture") #
#parser.add_argument('--learning_rate', action="store", dest="learn_rate", type=float, default = 0.001, help="learning rate for optimizer") #
#parser.add_argument('--hidden_units', action="store", dest="node_hidden_", type=int, default = 512, help="# Hidden units/nodes")  #
#parser.add_argument('--epochs', action="store", dest="num_epoch_",type=int, default = 1, help="Epochs/iteration")   #
parser.add_argument('--topk', action="store", dest="top_num",type=int, default = 5, help="top K propability ")  #
parser.add_argument('--gpu', action="store_true", default=False, help="Default cpu, unless --gpu")              #
args = parser.parse_args()


if args.gpu == True:
    device__ = "cuda"
else:
    device__ = "cpu"

#print ("Data directory :   " + str(args.data_dir) )
#print ("saving directory : " + str(args.save_dir) )
#print ("architecture :     " + str(args.archh) )
print ("category_names :   " + str(args.json_file) )
#print ("learning rate :    " + str(args.learn_rate) )
#print ("hidden units :     " + str(args.node_hidden_) )
#print ("epochs :           " + str(args.num_epoch_) )
print ("topk   :           " + str(args.top_num) )
print ("checkpoint_dir :   " + str(args.checkpoint_dir) )
print ("image_dir :        " + str(args.image_dir) )
print ("device :           " + device__)
    
checkpoint_path=str(args.checkpoint_dir)     #checkpoint_dir      #'checkpoint_adam_resnet152_10epoch_opti_name.pth'
image_path=str(args.image_dir)               #image_dir      #'flowers/test/35/image_06984.jpg'
#top_num=10
topk=args.top_num
#json file needed 
#cpu vs. gpu neede
#device__='cpu'
input_device=device__


with open(str(args.json_file), 'r') as f:
    cat_to_name = json.load(f)
    


def label_to_name_function(label_,cat_to_name,label_to_class):
    
    class_=str(label_to_class[label_.item()])
    name_=cat_to_name[class_]
    return name_





def load_checkpoint(filepath,input_device):
    
    checkpoint = torch.load(filepath)
    print('')
    print('===============')
    print('checkpoint:')
    print(checkpoint['model_name'])
    print(checkpoint['model_classifier'])
    print('')

    if (checkpoint['model_name']=='resnet152'):

            model = models.resnet152(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            #model=checkpoint['model_base']
            model.class_to_idx=checkpoint['class_to_label']
            model.fc = checkpoint['model_classifier']
            model.load_state_dict(checkpoint['model_state_dict'])


    elif (checkpoint['model_name']=='alexnet'):

            model = models.alexnet(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
                
            model.class_to_idx=checkpoint['class_to_label']    
            model.classifier = checkpoint['model_classifier']
            model.load_state_dict(checkpoint['model_state_dict'])




    elif (checkpoint['model_name']=='densenet161'):

            model = models.densenet161(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
                
            model.class_to_idx=checkpoint['class_to_label']   
            model.classifier = checkpoint['model_classifier']
            model.load_state_dict(checkpoint['model_state_dict'])


    else:

        print('!!!!Prediction module supports only resnet152, densenet161, and alexnet checkpoint!!!!')
        print('')
        print('')


    if(input_device=='cuda'):
        model=model.cuda()
    if(input_device=='cpu'):
        model=model.cpu()


    # model=model.cuda()
    model=model.eval()
    return model,checkpoint,input_device
    

def process_image(filepath):
    im = Image.open(filepath)

    ratio=im.width/im.height

    if(ratio<1.0):
        new_height=int(256/ratio)
        im_resize=im.resize((256,new_height))
    else:
        new_width=int(256*ratio)
        im_resize=im.resize((new_width,256))

 

    center_y=int(im_resize.width/2.)
    center_x=int(im_resize.height/2.)
    #print(center_x,center_y)


    upper=center_y-112   # PIL  real x,is not actually along width, width->y   height->x

    left=center_x-112

    lower=center_y+112

    right=center_x+112



    im_resize_crop=im_resize.crop((upper,left,lower,right))

    np_image = np.array(im_resize_crop)

    np_image_norm=np_image/255.


    mean_=[0.485, 0.456, 0.406]
    std_=[0.229, 0.224, 0.225]

    np_image_norm_transform=np.zeros((224, 224,3))

    for i in range(3):

        np_image_norm_transform[:,:,i]=(np_image_norm[:,:,i]-mean_[i])/std_[i]



    np_image_norm_transform_=np_image_norm_transform.transpose((2, 0, 1))
    return np_image_norm_transform_



def predict(image_path, checkpoint_path, topk,input_device):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model_,checkpoint_,input_device= load_checkpoint(checkpoint_path,input_device)
#    device_=checkpoint_['device']
    device_=input_device
    
    np_image_norm_transform_=process_image(image_path)
    torch_img=torch.from_numpy(np_image_norm_transform_)
    title_= '?'   #str(cat_to_name[image_path.split('/')[2]])    #for image title
    class__= '?'  #int(image_path.split('/')[2])
    inputs=torch_img
    inputs=inputs.type_as(torch.FloatTensor())
    inputs=inputs.to(device_)
    print(device_)


    model_.eval()
    with torch.no_grad():
        img=inputs
        img.unsqueeze_(0)  #   with single image batch size missing, so need to add another dimension  
                           #   https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/4
        output = model_.forward(img)
        out_exp = torch.exp(output)
        pred=out_exp.max(1)[1]
        pred_list=out_exp



    pred_numpy=pred_list.cpu().numpy()   #need to make to cpu 

    sort_list=np.argsort(pred_numpy[0])
    sort_list_rev=sort_list[::-1]

    top_k_list=sort_list[-topk:]
    top_k_list_rev=sort_list_rev[0:topk]
    
    top_k_list_rev=np.array(top_k_list_rev)

    class_to_label=checkpoint_['class_to_label']

    label_to_class = {v: k for k, v in class_to_label.items()}


    dict_= checkpoint_['label_to_name']
    dict__=label_to_class
    step=0

    flower_name_list_topk=[]
    flower_class_list_topk=[]
    probability_list_topk=[]

    for i in top_k_list_rev:
        step +=1

        flower_name_list_topk.append(dict_[i])
        flower_class_list_topk.append(dict__[i])
        probability_list_topk.append(pred_numpy[0][i])

    return title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk  


def plot_flower_probability(image_path,checkpoint_path,topk):
    
    title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk=predict(image_path, checkpoint_path, topk)

    flower_name_list_topk=flower_name_list_topk[::-1]
    probability_list_topk=probability_list_topk[::-1]
    flower_tuple_name=tuple(flower_name_list_topk)
    
    np_image_norm_transform_=process_image(image_path)
    torch_img=torch.from_numpy(np_image_norm_transform_)
    title_="?"   #str(cat_to_name[image_path.split('/')[2]])    #for image title

    ax=imshow(torch_img,title=title_)
    ax.set_title(title_)
    
    fig2, ax2 = plt.subplots()
    ind = np.arange(1, topk+1)
    plt.barh(ind, probability_list_topk)
    plt.yticks(ind,flower_tuple_name)
    plt.show()
    

    
title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk_=predict(image_path, checkpoint_path, topk,input_device)
#print(title_,class__,flower_name_list_topk,flower_class_list_topk,probability_list_topk,topk_)
print('')
print('====== Real flower =====')
print('')
print('Flower actual name:',title_) 
print('Flower actual class:',class__) 
print('')
print('====== Most likely flower =====')
print('')
print('Flower name:',flower_name_list_topk[0]) 
print('Flower class:',flower_class_list_topk[0]) 
print('Flower probability:',probability_list_topk[0]) 
print('')
print('======    Top %d   ======' %(topk_))
print('')
print('List of flower prediction: name: %s' %(flower_name_list_topk)) 
print('List of flower prediction: class: %s' %(flower_class_list_topk)) 
print('List of flower prediction: probability: %s' %(probability_list_topk)) 

#plot_flower_probability(image_path,checkpoint_path,5)    

end_time=time.time()

print('')
print('===== time =====')
print(end_time-start_time,' sec')

    