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

class CustomText_test_realdata(TextDataset):
    def __init__(self, data_root, label_root,cfg=None, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory
        self.cfg = cfg
        self.image_root = os.path.join(data_root, 'sample')
        self.gt_root = os.path.join(data_root, 'gt')
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
        txt.close()
        polygon = self.polygonlist['0937.jpg']

        self.dict = {
        '1' : {
            'input_point': np.array([[126, 207]]),
            'label' : np.array([1])
            },
        '1_1' : {
            'input_point': np.array([[524, 243]]),
            'label' : np.array([1])
            },
        '2' : {
            'input_point': np.array([[309, 228]]),
            'label' : np.array([1])
            },
        '2_1' : {
            'input_point': np.array([[110, 264]]),
            'label' : np.array([1])
            },
        '2_2' : {
            'input_point': np.array([[487, 224]]),
            'label' : np.array([1])
            },
        '3' : {
            'input_point': np.array([[110, 64]]),
            'label' : np.array([1])
            },
        '4' : {
            'input_point': np.array([[29, 61]]),
            'label' : np.array([1]),
            },
        '4_1' : {
            'input_point': np.array([[239, 125]]),
            'label' : np.array([1]),
            },
        '5' : {
            'input_point': np.array([[186, 53]]),
            'label' : np.array([1])},
        '5_1' : {
            'input_point': np.array([[186, 78]]),
            'label' : np.array([1])},
        '5_2' : {
            'input_point': np.array([[186, 106]]),
            'label' : np.array([1])},
        '6' : {
            'input_point': np.array([[177, 24]]),
            'label' : np.array([1])},
        '6_1' : {
            'input_point': np.array([[177, 58]]),
            'label' : np.array([1])},
        '7' : {
            'input_point': np.array([[27, 114]]),
            'label' : np.array([1])},
        '8' : {
            'input_point': np.array([[419, 304]]),
            'label' : np.array([1])},
        '8_1' : {
            'input_point': np.array([[59, 308]]),
            'label' : np.array([1])},
        '9' : {
            'input_point': np.array([[30, 68]]),
            'label' : np.array([1])},
        '9_1' : {
            'input_point': np.array([[38, 135]]),
            'label' : np.array([1])},
                
        '10' : {
            'input_point': np.array([[70, 73]]),
            'label' : np.array([1])},
                        
        '10_1' : {
            'input_point': np.array([[65, 91]]),
            'label' : np.array([1])},
        
        '10_2' : {
            'input_point': np.array([[65, 110]]),
            'label' : np.array([1])},
        
        'f_1' : {
            'input_point': np.array([[65, 110]]),
            'label' : np.array([1])},
        'f_2' : {
            'input_point': np.array([[65, 110]]),
            'label' : np.array([1])},
        'f_3' : {
            'input_point': np.array([[65, 110]]),
            'label' : np.array([1])},
        'f_4' : {
            'input_point': np.array([[65, 110]]),
            'label' : np.array([1])},
        'f_5' : {
            'input_point': np.array([[65, 110]]),
            'label' : np.array([1])},
        }
        # self.dict = {
        # '1' : {
        #     # 'input_point': np.array([[352, 139]]),
        #     'input_point': np.array([[354, 392]]),
        #     # 'input_point': np.array([[352, 139], [354, 392]]),
        #     # 'input_point': np.array([[352, 139], [354, 392]]),
        #     # 'label' : np.array([1, 1])
        #     'label' : np.array([1])
        #     },
        # '1_1' : {
        #     # 원포인트
        #     # 'input_point': np.array([[352, 139]]),
        #     'input_point': np.array([[342, 135]]),
        #     'label' : np.array([1])
        #     },
        # '1_2' : {
        #     # 원포인트
        #     # 'input_point': np.array([[354, 392]]),
        #     'input_point': np.array([[360, 392]]),
        #     'label' : np.array([1])
        # },
        # '2' : {
        #     'input_point': np.array([[172, 165]]),
        #     # 'input_point': np.array([[172, 165], [170, 200], [168, 240], [168, 281], [165, 321],
        #     #                          [485, 137], [484, 175], [485, 219], [483, 266], [485, 318]]),
        #     'label' : np.array([1]),
        #     },
        # '2_1' : {
        #     'input_point': np.array([[185, 165]]),
        #     'label' : np.array([1])},
        # '2_2' : {
        #     'input_point': np.array([[170, 200]]),
        #     'label' : np.array([1])},
        # '2_3' : {
        #     'input_point': np.array([[168, 240]]),
        #     'label' : np.array([1])},
        # '2_4' : {
        #     'input_point': np.array([[163, 275]]),
        #     'label' : np.array([1])},
        # '2_5' : {
        #     'input_point': np.array([[170, 325]]),
        #     'label' : np.array([1])},
        # '2_6' : {
        #     'input_point': np.array([[495, 137]]),
        #     'label' : np.array([1])},
        # '2_7' : {
        #     'input_point': np.array([[485, 175]]),
        #     'label' : np.array([1])},
        # '2_8' : {
        #     'input_point': np.array([[485, 219]]),
        #     'label' : np.array([1])},
        # '2_9' : {
        #     'input_point': np.array([[478, 268]]),
        #     'label' : np.array([1])},
        # '2_10' : {
        #     'input_point': np.array([[417, 315]]),
        #     'label' : np.array([1])},
                                                        
        
        # '3' : {
        #     # 'input_point': np.array([[681, 308], [684, 580]]),
        #     'input_point': np.array([[681, 308]]),
        #     # 'label' : np.array([1, 1])
        #     'label' : np.array([1])
        #     },
        # '3_1' : {
        #     'input_point': np.array([[681, 308]]),
        #     'label' : np.array([1])
        #     },
        # '3_2' : {
        #     'input_point': np.array([[689, 580]]),
        #     'label' : np.array([1])
        #     },
        # # '3_1' : {
        # #     'input_point': np.array([[684, 580]]),
        # #     },
        # '4' : {
        #     'input_point': np.array([[750, 540]]),
        #     'label' : np.array([1])},
        # '5' : {
        #     'input_point': np.array([[766, 530]]),
        #     'label' : np.array([1])}}
        
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
        filename = image_id.split('.')[0]
        center_point = self.dict[filename]['input_point']
        label_point = self.dict[filename]['label']
        data = self.load_img(self.image_root, image_id)
        # gt_mask = np.load(self.gt_root + f'/gt_{filename}.npy').astype('float32')
        
        polygons = self.polygonlist['0937.jpg']
        
        image = np.array(data["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        h, w, c = image.shape
                                
        polygons, extended_poly = self.polygon_extender(polygons, num_poly=16)
        
        return self.get_test_data(image, polygons, center_point, image_id=data["image_id"], image_path=data["image_path"], extended_poly=extended_poly)

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
    
def prefix(s):
    s_1 = s.split(',')
    return (int(s_1[0]), int(s_1[1]))