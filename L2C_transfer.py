# coding: utf-8
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config)) # 此处不同
'''

import numpy as np
#from model_self import DQN_model,grasp_model
import time
import utils
from keras.models import Sequential
from keras.layers import Activation,Convolution2D,MaxPooling2D
from keras.models import load_model

#from PIL import Image
import cv2


def main():
    #rbt=Robot()
    #ret, color_heightmap, depth_heightmap = rbt.get_height_img()
    data_path_base='/home/zhsy/work/DataSet/grasp3/'

    # build model
    model = Sequential()

    model.add(Convolution2D(nb_filter=32,
                            nb_row=5,
                            nb_col=5,
                            border_mode='valid',
                            input_shape=(240+72, 240+72, 1)
                            ))
    model.add(Activation('relu')) #308*308

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same'
    ))  # 1 #154*154
    model.add(Convolution2D(nb_filter=64,
                            nb_row=5,
                            nb_col=5,
                            border_mode='valid'
                            ))
    model.add(Activation('relu')) #150*150

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same'
    ))  # 75*75
    model.add(Convolution2D(nb_filter=128,
                            nb_row=5,
                            nb_col=5,
                            border_mode='valid'
                            ))#71*71
    model.add(Activation('relu'))


    model.add(Convolution2D(nb_filter=1024,
                            nb_row=11,
                            nb_col=11,

                            border_mode='valid'
                            ))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=1,
                            nb_row=1,
                            nb_col=1,
                            border_mode='valid'
                            ))
    model.add(Activation('sigmoid'))

    model.summary()
    #time.sleep(10000)
    #移植
    model1 = load_model(
        '/home/zhsy/work/Pycharm/Projects/Visual-pushing-grasping/push-grasp-CNN/data/local_global_cnn3.h5')

    for i in range(1,6):
        mWeight=model1.get_layer("conv2d_%d"%i).get_weights()
        print(mWeight)
        model.get_layer("conv2d_%d"%i).set_weights(mWeight)
        mWeight = model.get_layer("conv2d_%d" % i).get_weights()
        print(mWeight[0][0][0][0][0])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
    model.save('/home/zhsy/work/Pycharm/Projects/Visual-pushing-grasping/push-grasp-CNN/data/global_succ_cnn1.h5')



    #test
    num = 1
    while True:
        num += 10
        im_gray = cv2.imread(data_path_base + 'depth_image/depth_img_%d.jpg' % num, cv2.IMREAD_GRAYSCALE)
        #inputImage=im_gray

        im_gray=im_gray[:, :] * 0.12 / 255.
        #print(im_gray[45,45])
        inputImage=np.zeros([240+72,240+72])
        inputImage[37:37+240,37:37+240] = im_gray
        im_gray=inputImage
        #print(im_gray[37+45,37+45])
        #im_gray = cv2.flip(im_gray, 1)

        im_gray=(im_gray[:,:]).reshape(-1,312,312,1)

        print(time.time())
        lable=model.predict(im_gray)
        print(time.time())
        lable=lable.reshape(61, 61)[:,:]*255
        #lable.dtype="uint8"
        #print(lable)



        color_heightmap = cv2.imread(data_path_base + "color_image/color_img_%d.jpg" % num)
        #color_heightmap = cv2.flip(color_heightmap, 1)
        img = utils.color_img_addlable61(color_heightmap, lable)


        cv2.imshow("image",img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()