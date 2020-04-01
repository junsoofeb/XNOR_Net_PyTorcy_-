#-*- coding: utf-8 -*-
import os
import torch
import pickle as cPickle
#import cPickle as pickle
import numpy
import torchvision.transforms as transforms

class dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()  # transforms.ToTensor() --> PIL 이미지 또는 넘파이배열을 pytorch의 텐서로 변형
        
        # train 일때
        if self.train:
            train_data_path = os.path.join(root, 'train_data') # os.path.join(root, 'train_data') --> 경로 병합 "{root}/train_data"
            train_labels_path = os.path.join(root, 'train_labels')
            
            # open(train_data_path, 'rb') --> 경로에서 파일 읽기, cf) 파일경로 : train_data_path, 모드 : rb (read binary)
            # numpy.load() --> 저장된 넘파이배열 불러오기
            self.train_data = numpy.load(open(train_data_path, 'rb')) 
            
            # torch.from_numpy() --> 넘파이배열을 텐서로 변형
            # 원래의 ndarray 객체를 참조하므로 원래 ndarray 객체의 값을 바꾸면 텐서 자료형의 값도 바뀌고 반대도 마찬가지
            self.train_data = torch.from_numpy(self.train_data.astype('float32')) 
            
            self.train_labels = numpy.load(open(train_labels_path, 'rb')).astype('int')
        
        # inference 일때
        else:
            test_data_path = os.path.join(root, 'test_data')
            test_labels_path = os.path.join(root, 'test_labels')
            self.test_data = numpy.load(open(test_data_path, 'rb'))
            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            self.test_labels = numpy.load(open(test_labels_path, 'rb')).astype('int')

    # 매직 메소드
    def __len__(self): # 데이터의 개수 반환
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index): # img와 label 반환
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        return img, target
