

import numpy as np
#from model_self import DQN_model,grasp_model
import time
import utils

import cv2

from keras.models import load_model


add_obj_num=10

def main():
    data_addr_base='/home/zhsy/work/DataSet/grasp3/'

    count=1;
    model = load_model('/home/zhsy/work/Pycharm/Projects/Visual-pushing-grasping/push-grasp-CNN/data/local_cnn4.h5')
    starttime=time.time()
    while True:
        depth_heightmap = cv2.imread(data_addr_base+'depth_image/depth_img_%d.jpg'%count,cv2.IMREAD_GRAYSCALE)
        depth_heightmap = depth_heightmap[:, :] * 0.12 / 255
        lable=utils.generateGlobalData(depth_heightmap, model,np.mean(depth_heightmap))

        color_heightmap=cv2.imread(data_addr_base+"color_image/color_img_%d.jpg"%count)

        img=utils.color_img_addlable(color_heightmap, lable)
        cv2.imwrite(data_addr_base + 'color_lable1/lable_img_%d.jpg' % count,img)
        np.save(data_addr_base + 'global_lable1/lable_%d.npy'%count,lable )

        count+=1
        currenttime=time.time()
        speed=count*3600/(currenttime-starttime)
        print("computing lable of img %d"%count +"- compute speed = %d p/h------"%speed)

if __name__ == '__main__':
    main()