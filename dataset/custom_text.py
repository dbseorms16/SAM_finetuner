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

class CustomText(TextDataset):
    def __init__(self, data_root, cfg=None, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory
        self.cfg = cfg
        self.image_root = os.path.join(data_root, 'train' if is_training else 'test')
        self.back_root =  os.path.join(data_root, 'train_back' if is_training else 'test_back')
        # self.annotation_root = os.path.join(data_root, 'train' if is_training else 'test', "text_label_circum")
        self.image_list = os.listdir(self.image_root)
        self.back_image_list = os.listdir(self.back_root)
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
        # self.augmentation = Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        self.SAM_transform = ResizeLongestSide(1024)
        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img(item))
        # self.txt = open('./polygon.txt', 'w')

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

        if self.load_memory:
            data = self.datas[item]
        else:
            image_id = self.image_list[item]
            data = self.load_img(self.image_root, image_id)
            if self.is_training:
                random_id = random.randint(0, len(self.back_image_list)) - 1
                image_id = self.back_image_list[random_id]
            else:
                ##Todo auto
                image_id = '0'+ image_id[1:-4] + '.jpg'
                # image_id = image_id[1:-4] + '.jpg'
            back_data = self.load_img(self.back_root, image_id)
            
        image, polygons, center_point = perform_operation(data['image'], back_data['image'], magnitude=0.1, is_training=self.is_training )
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # self.txt.write('{}/{},{}/{},{}/{},{}/{},{}\n'.format(image_id, polygons[0][0], polygons[0][1], \
        #                                                     polygons[1][0], polygons[1][1], \
        #                                                     polygons[2][0], polygons[2][1], \
        #                                                     polygons[3][0], polygons[3][1], ))
        # cv2.imwrite('./data/validation/'+image_id, image)

        image_shape = image.shape[:2]
        # SAM preprocessing
        # image = self.SAM_transform.apply_image(image)
        polygons, extended_poly = self.polygon_extender(polygons, num_poly=16)
        
        if self.is_training:
            return self.get_training_data(image, polygons, center_point,
                                          image_id=data["image_id"], image_path=data["image_path"], extended_poly=extended_poly)
            
            # return self.get_training_data(data["image"], data["polygons"],
            #                               image_id=data["image_id"], image_path=data["image_path"])
        else:
            return self.get_test_data(image, polygons, center_point, image_id=data["image_id"], image_path=data["image_path"], 
                                      extended_poly=extended_poly)

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

        return polygons, new