#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import json
import logging
import os
import sys
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import smdebug.pytorch as smd
from smdebug import modes
# from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader,criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    hook.set_mode(smd.modes.EVAL)
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, average test loss: {total_loss}")

def train(model, train_loader,
         validation_loader, criterion, 
         optimizer, args, 
         device, hook):

    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=args.epochs
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    # =================================================#
    # 2. Set the SMDebug hook for the training phase. #
    # =================================================#
    hook.set_mode(smd.modes.TRAIN)
    hook.register_loss(criterion)  

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                hook.set_mode(modes.TRAIN)
                model.train()
            else:
                hook.set_mode(modes.EVAL)
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%) Time: {}".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                            time.asctime() # for measuring time for testing, remove for students and in the formatting
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                # if running_samples>(0.2*len(image_dataset[phase].dataset)):
                #     break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    #freeze the convolutional layer 
    for param in model.parameters():
        param.requires_grad = False   
    # then add fully connected layer
    num_features=model.fc.in_features ##check how many features present in output of the model
    model.fc = nn.Sequential(
                nn.Linear(num_features, 133)) 
    return model


def create_data_loaders(args):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    traindir = os.path.join(args.data_dir , 'train')
    testdir = os.path.join(args.data_dir, 'test')
    valdir = os.path.join(args.data_dir , 'valid')
    
    training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224,244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.ImageFolder(
        root=traindir, 
        transform=training_transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,shuffle=True, batch_size=args.batch_size)

    validationset = torchvision.datasets.ImageFolder(
        root=valdir, 
        transform=testing_transform)

    validationloader = torch.utils.data.DataLoader(
        validationset,shuffle=False, batch_size=args.batch_size)

    testset = torchvision.datasets.ImageFolder(
        root=testdir, 
        transform=testing_transform)

    testloader = torch.utils.data.DataLoader(
        testset,shuffle=False, batch_size=args.batch_size)
    

    return trainloader, validationloader, testloader
 
def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model.to(device)

    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors. #
    # ======================================================#
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    hook.register_loss(criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info('create data loaders')
    trainloader ,validationloader, testloader = create_data_loaders(args)
    logger.info('training model')
    model = train(model, trainloader,validationloader, criterion, optimizer, args, device, hook)

    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info('final model accuracy')
    test(model, testloader, criterion, device, hook)

    '''
    TODO: Save the trained model
    '''
    logger.info('saving model')  
    with open(os.path.join(args.model_dir, 'dogmodel_profdebug.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])


    args=parser.parse_args()

    main(args)