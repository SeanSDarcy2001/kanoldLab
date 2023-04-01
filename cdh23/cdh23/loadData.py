import numpy as np
from pymatreader import read_mat
from torchvision.transforms import ToTensor
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from matplotlib import animation 
from IPython.display import HTML
import os

class loadTheData :
      
     
    def __init__(self, mice: list):
        self.mice = mice
        self.dFs = []
        self.stimHistories = []
        self.miceSessions = []



        for mouse in self.mice :
            print("------------------")
            print("Mouse:", mouse)
            print("------------------")
            print("Loading data...")
            dates = os.listdir(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23"))
            for filename in dates:
                if filename.startswith('2'):
                    print("Date:", filename)
                    self.miceSessions.append(mouse)
                    Fl = np.load(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23", filename, "suite2p/plane0/F.npy"), allow_pickle=True)
                    Fneu = np.load(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23", filename, "suite2p/plane0/Fneu.npy"), allow_pickle=True)
                    isCell = np.load(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23", filename, "suite2p/plane0/iscell.npy"), allow_pickle=True)
                    print("Calculating dF...")
                    dF = Fl - (0.7*Fneu)
                    dF = np.take(dF, np.where(isCell == 1)[0], axis = 0)
                    print("Neurons x Time:", dF.shape)
                    self.dFs.append(dF)
                    print("Loading stim history...")
                    contents = os.listdir(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23", filename))
                    for files in contents:
                        if 'sound_file' in files:
                            stimID = read_mat(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23", filename, files))['stimPlayedID']
                            atten = read_mat(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23", filename, files))['stimPlayedAtten']
                            onFrame = read_mat(Path("/Volumes/Data2/Travis/Cdh23/Cdh23 Data", mouse, "2P/L23", filename, files))['stimOnFrame']
                    stimHistory = np.array([stimID, atten, onFrame], dtype=int)
                    stimHistory = stimHistory.T
                    self.stimHistories.append(stimHistory)
                #else :
                    #dates.remove(filename)
        self.stimHistories = np.array(self.stimHistories)
        for i, k in enumerate(self.stimHistories) :
            self.stimHistories[i] =   self.stimHistories[i][np.lexsort((self.stimHistories[i][:,2], self.stimHistories[i][:,1],self.stimHistories[i][:,0]))]
         
        
    def processAndSort(self, mouse):
        orderedData = []
        stimsToReturn = []
        indeces = np.where(np.array(self.miceSessions) == mouse)[0]
        if len(indeces) > 1 :
            print("Mouse has", len(indeces), "sessions.")
        for i in indeces :
            k = 0
            session = []
            sessionStim = []
            for stim in range(len(self.stimHistories[i])) :
                start = self.stimHistories[i][stim][2] - 15 #1 seconds before (151 gets you back to index 150
                end = self.stimHistories[i][stim][2] + 53 #3 secs after
                Fo = np.mean(self.dFs[i][:, start :start + 15], 1) #have to shift mouse data
                trial = (((self.dFs[i][:, start:end].T) - Fo) / Fo).T #have to shift mouse data
                session.append(trial)
                sessionStim.append(self.stimHistories[i])
            orderedData.append(session)
            stimsToReturn.append(sessionStim)
        return orderedData, stimsToReturn
    
    def stackMice(self):
        orderedData = []
        stimsToReturn = []
        for stim in range(200) :
            for i in range(len(self.mice)) :
                start = self.stimHistories[i][stim][2] - 15 #1 seconds before (151 gets you back to index 150
                end = self.stimHistories[i][stim][2] + 53 #3 secs after
                Fo = np.mean(self.dFs[i][:, start :start + 15], 1) #have to shift mouse data
                if i == 0 :
                    trial = (((self.dFs[i][:, start:end].T) - Fo) / Fo).T #have to shift mouse data
                else :
                    trial = np.concatenate((trial, (((self.dFs[i][:, start:end].T) - Fo) / Fo).T), axis=0)
            orderedData.append(trial)
            stimsToReturn.append(self.stimHistories[i])
        return orderedData, stimsToReturn
