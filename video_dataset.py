import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data.dataset import Dataset

class Video_dataset(Dataset):
    """ PyTorch Data implementation to load and process videos """
    
    def __init__(self,video_names,labels,sequence_length = 60,transform = None):
        """
      video_names : list ------- list of videos' path
      labels : DataFrame -------  having two columns file name and labels associated
      sequence_length : int ------- nb of frames to extract for each videos
      transform : transforms.Compose ------- transformations to apply to each frame (normalization, resize
        """
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        """
        -> nb of videos in the dataset, used by PyTorch Dataloader
        """
        return len(self.video_names)
    
    def __getitem__(self,idx):
        """
        idx : int ------- index of the video to process
        -> exctracts nb of frames, transforms and returns a tensor of frames
        """
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        temp_video = video_path.split('/')[-1]
        #print(temp_video)
        label = self.labels.iloc[(labels.loc[labels["file"] == temp_video].index.values[0]),1]
        if(label == 'FAKE'):
          label = 0
        if(label == 'REAL'):
          label = 1
        for i,frame in enumerate(self.frame_extract(video_path)):
          frames.append(self.transform(frame))
          if(len(frames) == self.count):
            break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        #print("length:" , len(frames), "label",label)
        return frames,label
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image