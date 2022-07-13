import numpy as np
from tifffile import imread
from os import listdir
from pymatreader import read_mat
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def sequence(start, end):
    res = []
    diff = 1
    x = start
    while x <= end:
        res.append(x)
        x += diff
    return res

class NeuralDataset(Dataset) :
#tiff_path is path to .tiff files containing widefield images
#spec_path is path to spectrograms
#timeStamps is .mat file that is stims x nReps, containing frame info
  def __init__(self, tiff_path, spec_path, timeStamps, transforms = None):
    self.tiffs = listdir(tiff_path)
    self.specs = listdir(spec_path)
    self.timeStamps = read_mat(timeStamps)
    self.transform = transforms
  
  def __len__(self):
    return len(self.specs)

  def __getitem__(self, idx):
    #load 30 image stack, tiffs in 9798 n stacks
    reps = []
    for i in range(2) :
        start = self.timeStamps[idx, i]
        end = self.timeStamps[idx, i] + 30

        #basically figures out what stack the start and end frames are in
        if start > 9798 or end > 9798 :
            tempStart = start
            counterS = 0
            tempEnd = end
            counterE = 0
            while tempStart > 9798 :
                counterS += 1
                tempStart -= 9798
            while tempEnd > 9797 :
                counterE += 1
                tempEnd -= 9798

        #if start and end of sequence are in same stack
        if counterE == counterS :
            seq = sequence(tempStart, tempEnd)
            neurons = imread(self.tiffs[counterS], key = seq)
            neurons = ToTensor()(neurons)
            reps.append(neurons)
        #if they are in different stacks
        #to test
        else :
            seq = sequence(tempStart, 9798)
            neurons_stack1 = imread(self.tiffs[counterS], key = seq)
            seq = sequence(1, tempEnd)
            neurons_stack2 = imread(self.tiffs[counterE], key = seq)
            neurons = ToTensor()(neurons_stack1 + neurons_stack2)
            reps.append(neurons)
        
    spec = torchvision.io.read_image(self.specs[idx])
    neurons = torch.stack(reps[0], reps[1], 0)
    return neurons, spec

