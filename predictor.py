# coding : utf-8


import numpy as np
import time
import utils
from keras.models import load_model
import random

class Predictor(object):
    def __init__(self):
        self.LCNN = load_model(
            '/home/arm/Desktop/RobotGrasp_zhsy/data/local_cnn4.h5')
        self.GCNN = load_model(
            '/home/arm/Desktop/RobotGrasp_zhsy/data/global_cnn1.h5')

    def GCNN_LCNN(self,depth_heightmap):
        inputImage = np.zeros([240 + 72, 240 + 72],dtype=np.float)
        inputImage[37:37 + 240, 37:37 + 240] = depth_heightmap
        inputImage = (inputImage[:, :]).reshape(-1, 312, 312, 1)

        StartTime=time.time()

        lable = self.GCNN.predict(inputImage).reshape((31, 31))

        location = np.where(lable == np.max(lable))
        depth = depth_heightmap[int(location[0][0] * 7.73), int(location[1][0] * 7.73)]
        location = [location[0][0] * 8, location[1][0] * 8]

        angles = list(range(0, 180, 20))
        rectSlides = utils.GetRoatePatch(location, angles, depth_heightmap)
        score = self.LCNN.predict(rectSlides.reshape(-1, 36, 72, 1))
        angleind = int(np.where(score == np.max(score))[0][0])
        angle = angles[angleind]

        EndTime = time.time()

        location = [location[0] , location[1] ]
        #print(location, angle)
        PlanningTime=EndTime-StartTime


        patch=rectSlides.reshape(-1, 36, 72, 1)[int(np.where(score == np.max(score))[0][0])]

        return location,angle,PlanningTime,lable,depth,patch


