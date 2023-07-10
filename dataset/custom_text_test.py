#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import os
import numpy as np
from dataset.dataload import TextDataset, TextInstance
from dataset.augmentation import *
import cv2
from PIL import ImageEnhance, Image
import random
from segment_anything.utils.transforms import ResizeLongestSide

class CustomText_test(TextDataset):
    def __init__(self, data_root, label_root,cfg=None, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory
        self.cfg = cfg
        self.image_root = os.path.join(data_root)
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
        # self.augmentation = Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        self.SAM_transform = ResizeLongestSide(1024)
        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img(item))
        txt = open(label_root, 'r')
        self.polygonlist = {}
        for line in txt.readlines():
            s = line.split('/')
            f_name = s[0]
            s_1 = prefix(s[1])
            s_2 = prefix(s[2])
            s_3 = prefix(s[3])
            s_4 = prefix(s[4])
            polygon = [s_1, s_2, s_3, s_4]
            self.polygonlist[f_name] = polygon
                    # self.txt.write('{}/{},{}/{},{}/{},{}/{},{}\n'.format(image_id, polygons[0][0], polygons[0][1], \
        #                                                     polygons[1][0], polygons[1][1], \
        #                                                     polygons[2][0], polygons[2][1], \
        #                                                     polygons[3][0], polygons[3][1], ))
        
    def load_img(self, img_root, image_id):
        image_path = os.path.join(img_root, image_id)

        # Read image data
        image = Image.open(image_path)
        image = image.convert('RGBA')
        data = dict()
        data["image"] = image
        # data["polygons"] = polygons
        data["image_id"] = image_id.split("/")[-1]
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):

        image_id = self.image_list[item]
        data = self.load_img(self.image_root, image_id)
        
        polygons = self.polygonlist[image_id]
        image = np.array(data["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        h, w, c = image.shape
        center_point = np.array([[int(w//2), int(h//2)]])
                                
        polygons = self.polygon_extender(polygons, num_poly=16)
        
        return self.get_test_data(image, polygons, center_point, image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.image_list)
    
    def poly(self, s, e, num_poly):
        nps = []
        poly_num = num_poly 
        nps.append([s[0], s[1]])
        min_x, min_y = min(s[0], e[0]), min(s[1], e[1])
        if min_x == s[0]:
            min_y = s[1]
            max_x = e[0]
            max_y = e[1]
        else :
            min_y = e[1]
            min_x = e[0]
            max_x = s[0]
            max_y = s[1]
            
        s_x = (max_x - min_x)  / poly_num
        s_y = (max_y - min_y) / poly_num
        for i in range(1, poly_num):
            n_p = [int(min_x + (s_x * i)), int(min_y +int(s_y * i))]
            
            nps.append(n_p)
        return nps
    # read input
    def polygon_extender(self, polygon, num_poly=2):
        polygons = []
        n1 = self.poly(polygon[0], polygon[1], num_poly)
        n2 = self.poly(polygon[1], polygon[2], num_poly)
        n3 = self.poly(polygon[2], polygon[3], num_poly)
        n4 = self.poly(polygon[3], polygon[0], num_poly)
        new = n1 + n2 + n3 + n4
        polygons.append(TextInstance(new, 'c', "**"))

        return polygons
    
def prefix(s):
    s_1 = s.split(',')
    return (int(s_1[0]), int(s_1[1]))