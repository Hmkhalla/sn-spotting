from torch.utils.data import Dataset

import numpy as np
import random
import os
import time


from tqdm import tqdm

import torch

import logging
import json

#from SoccerNet.Downloader import getListGames
import SoccerNet.Downloader as scntdownload
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1
EVENT_DICTIONARY_V3 = {"comments":0}
INVERSE_EVENT_DICTIONARY_V3 = {0: "comments"}

MISSING_GAMES = {
"france_ligue-1/2015-2016/2015-09-26 - 18-30 Nantes 1 - 4 Paris SG",
"france_ligue-1/2015-2016/2015-11-07 - 19-00 Paris SG 5 - 0 Toulouse",
"france_ligue-1/2015-2016/2015-09-19 - 18-30 Reims 1 - 1 Paris SG",
"france_ligue-1/2016-2017/2016-08-12 - 21-00 Bastia 0 - 1 Paris SG",
"france_ligue-1/2016-2017/2016-09-16 - 21-45 Caen 0 - 6 Paris SG",
"france_ligue-1/2016-2017/2016-09-23 - 21-45 Toulouse 2 - 0 Paris SG",
"france_ligue-1/2016-2017/2016-10-01 - 18-00 Paris SG 2 - 0 Bordeaux",
"france_ligue-1/2016-2017/2016-10-15 - 18-00 Nancy 1 - 2 Paris SG",
"france_ligue-1/2016-2017/2016-10-28 - 21-45 Lille 0 - 1 Paris SG",
"france_ligue-1/2016-2017/2016-11-06 - 22-45 Paris SG 4 - 0 Rennes",
"france_ligue-1/2016-2017/2016-11-19 - 19-00 Paris SG 2 - 0 Nantes",
"france_ligue-1/2016-2017/2016-11-27 - 22-45 Lyon 1 - 2 Paris SG",
"france_ligue-1/2016-2017/2016-12-11 - 22-45 Paris SG 2 - 2 Nice",
"france_ligue-1/2016-2017/2016-12-17 - 19-00 Guingamp 2 - 1 Paris SG",
"france_ligue-1/2016-2017/2016-12-21 - 22-50 Paris SG 5 - 0 Lorient",
"france_ligue-1/2016-2017/2017-02-19 - 23-00 Paris SG 0 - 0 Toulouse",
"france_ligue-1/2016-2017/2017-04-18 - 19-30 Metz 2 - 3 Paris SG",
"france_ligue-1/2016-2017/2017-04-22 - 18-00 Paris SG 2 - 0 Montpellier",
"france_ligue-1/2016-2017/2017-05-06 - 18-00 Paris SG 5 - 0 Bastia",
"france_ligue-1/2016-2017/2017-05-20 - 22-00 Paris SG 1 - 1 Caen",
"france_ligue-1/2016-2017/2016-08-21 - 21-45 Paris SG 3 - 0 Metz",
"france_ligue-1/2016-2017/2016-09-09 - 21-45 Paris SG 1 - 1 St Etienne",
"france_ligue-1/2016-2017/2016-09-20 - 22-00 Paris SG 3 - 0 Dijon",
"france_ligue-1/2016-2017/2016-10-23 - 21-45 Paris SG 0 - 0 Marseille",
"france_ligue-1/2016-2017/2016-12-03 - 19-00 Montpellier 3 - 0 Paris SG",
"france_ligue-1/2016-2017/2017-03-04 - 19-00 Paris SG 1 - 0 Nancy",
"france_ligue-1/2016-2017/2017-04-09 - 22-00 Paris SG 4 - 0 Guingamp",
"france_ligue-1/2016-2017/2016-08-28 - 21-45 Monaco 3 - 1 Paris SG",
"france_ligue-1/2016-2017/2016-11-30 - 23-00 Paris SG 2 - 0 Angers",

}

def getListGames(*args, **kwargs):
    return [game for game in scntdownload.getListGames(*args, **kwargs) if game not in MISSING_GAMES]



def feats2clip(feats, stride, clip_length, padding = "replicate_last", off=0):
    if padding =="zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0]/stride)*stride
        print("pad need to be", clip_length-pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length-pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feats.shape[0]-1)
    # print(idx)
    return feats[idx,...]


class SoccerNetClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["train"], version=1, 
                framerate=2, window_size=15):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size*framerate
        self.version = version
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"
        elif version == 3:
            self.dict_event = EVENT_DICTIONARY_V3
            self.num_classes = 1
            self.labels="Labels-v3.json"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=True)


        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()

        # game_counter = 0
        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame, clip_length=self.window_size_frame)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame, clip_length=self.window_size_frame)

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes+1))
            label_half1[:,0]=1 # those are BG classes
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes+1))
            label_half2[:,0]=1 # those are BG classes


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes ) 

                
                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame//self.window_size_frame>=label_half1.shape[0]:
                    continue
                if half == 2 and frame//self.window_size_frame>=label_half2.shape[0]:
                    continue

                if half == 1:
                    label_half1[frame//self.window_size_frame][0] = 0 # not BG anymore
                    label_half1[frame//self.window_size_frame][label+1] = 1 # that's my class

                if half == 2:
                    label_half2[frame//self.window_size_frame][0] = 0 # not BG anymore
                    label_half2[frame//self.window_size_frame][label+1] = 1 # that's my class
            
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats[index,:,:], self.game_labels[index,:]

    def __len__(self):
        return len(self.game_feats)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"], version=1, 
                framerate=2, window_size=15):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size*framerate
        self.framerate = framerate
        self.version = version
        self.split=split
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels="Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels="Labels-v2.json"
        elif version == 3:
            self.dict_event = EVENT_DICTIONARY_V3
            self.num_classes = 1
            self.labels="Labels-v3.json"

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        for s in split:
            if s == "challenge":
                downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)
            else:
                downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))
        feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

        # Load labels
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        # check if annoation exists
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                
                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

        
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=1, off=int(self.window_size_frame/2), 
                        clip_length=self.window_size_frame)

        
        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)

