# The Kanold Laboratory: Sensory Brain-Computer Interfacing

Code written by Sean Sebastian Darcy, Johns Hopkins Biomedical Engineering major focused in Neuroengineering, minoring in Computer Integrated Surgery and Psychology.

CNNforTones.ipynb is a python notebook containing a convolutional neural network (CNN) classifier trained on widefield imaging frames capturing auditory cortex responding to 16 distinct pure tone stimuli at 2 intensity levels each. 

NeuralWAV_VAE.ipynb is a python notebook containing a variational autoencoder network architecture. The encoder component takes sequences of neural frames as input and maps to a latent representation z. The decoder takes the latent representation z and maps to points in 2-channel audio space.

soundVis.ipynb is a python notebook that leverages the NeuralDataset() custom pytorch dataset/dataloader to visualize the neural activity.

alignScope.py leverages cv2 registration methods to compute a transformation matrix Freg_i between the current image frame/plane of the 2P scope and some template image (for example, a key ROI imaged previously). The registration error is computed between Freg_i and the identity matrix I. The scope can be adjusted prior until this error falls below some threshold prior to imaging.
