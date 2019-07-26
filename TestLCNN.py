'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config)) # 此处不同

'''

from robot import Robot
#from model_self import DQN_model,grasp_model
import time
import utils
from keras.models import load_model

#from PIL import Image


add_obj_num=10

def main():
    rbt = Robot()
    model = load_model('/home/zhsy/work/Pycharm/Projects/Visual-pushing-grasping/push-grasp-CNN/data/local_cnn4.h5')

    rbt.add_objects(add_obj_num)
    time.sleep(3)
    ret, color_heightmap, depth_heightmap = rbt.get_height_img()
    model=load_model('/home/zhsy/work/Pycharm/Projects/Visual-pushing-grasping/push-grasp-CNN/data/local_cnn4.h5')
    while True:





        grasparg, rectSlide, colorrectslide = utils.GraspbyCNN(depth_heightmap,color_heightmap,model)
        grapsPoint = [int(grasparg[0] * 34. / 240.), int(grasparg[1] * 34. / 240.)]
        graspangle = grasparg[2]
        succ = rbt.grasp(grapsPoint,graspangle)





if __name__ == '__main__':
    main()