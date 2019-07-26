'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config)) # 此处不同
'''

#data_process package
import numpy as np

#system package


#image process package

import cv2


#keras package
from keras.models import load_model



#self package
#from model_self import DQN_model,grasp_model
import utils




def main():
    data_path_base = '/home/zhsy/work/DataSet/grasp3/'
    model = load_model('/home/zhsy/work/Pycharm/Projects/Visual-pushing-grasping/push-grasp-CNN/data/local_global_cnn3.h5')

    num=1
    while True:
        num+=10
        im_gray = cv2.imread(data_path_base + 'depth_image/depth_img_%d.jpg' % num, cv2.IMREAD_GRAYSCALE)
        im_gray=cv2.flip(im_gray,1)[:,:]*0.12/255.
        print(im_gray[129,201])

        lable=np.zeros((30,30))
        #print(lable)
        for r in range(4,240,8):
            for c in range(4,240,8):
                rect_gray = utils.SquareRecttailor(im_gray, [r,c])
                Y=model.predict(rect_gray.reshape(1,72,72,1)).reshape(1)
                lable[int(r/8),int(c/8)]=Y[0]


                #cv2.imshow("image", img)
                #cv2.waitKey(0)
        #print(lable)
        color_heightmap = cv2.imread(data_path_base + "color_image/color_img_%d.jpg" % num)
        color_heightmap=cv2.flip(color_heightmap,1)
        img = utils.color_img_addlable(color_heightmap, lable)
        cv2.imshow("image",img)
        cv2.waitKey(0)



    '''
    for P_rect in P_list:
        if P_rect[0]>=5:
            rect_gray = utils.SquareRecttailor(im_gray, P_rect[1])
            lable_list = lable_list + 'depth_square_image_%d.jpg' % rect_num + "   1" + "\n"
            count+=1
        else:
            rect_gray = utils.SquareRecttailor(im_gray, P_rect[1])
            lable_list = lable_list + 'depth_square_image_%d.jpg' % rect_num + "   0" + "\n"
            count-=1

        cv2.imwrite(data_path_base + 'depth_square_image/depth_square_img_%d.jpg' % rect_num, rect_gray)

        rect_num+=1;
        if count<0:
            break

    f = open(data_path_base + 'lable_square.txt', 'a')
    f.write(lable_list)
    f.close()
    lable_list = ""
    print("computing ---- ",num)
    '''





if __name__ == '__main__':
    main()