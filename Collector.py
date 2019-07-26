
from robot import Robot
import numpy as np
import time
import utils
import cv2
import random
import pickle


First_run=True


def main():

    if First_run:
        num = 1;
        storelist = [0]
    else:
        storelist = pickle.load(open('/home/zhsy/work/DataSet/grasp3/data/ind.txt', 'rb'))
        num=storelist[0]

    rbt = Robot()
    lable_list=""

    data_addr_base="/home/zhsy/work/DataSet/grasp3/"
    add_obj_num = 10
    rbt.add_objects(add_obj_num)
    time.sleep(3)
    ret,color_heightmap, depth_heightmap = rbt.get_height_img()

    graspcount=0

    while True:


        if np.sum(depth_heightmap) < 150 or rbt.checkState() or graspcount>(2*add_obj_num):#没有物品或者出现异常
            rbt.simrestart()
            add_obj_num=random.randint(5, 15)
            rbt.add_objects(add_obj_num)
            graspcount = 0
            time.sleep(3)


        ret, color_heightmap, depth_heightmap = rbt.get_height_img()
        if ret == 8:
            time.sleep(10)
            continue

        if(random.randint(0,9)>=5):
            print('feature grasp')
            grasparg, rectSlide, colorrectslide = utils.GraspBaseFeature2(depth_heightmap,color_heightmap,1000)
        else:
            print('random grasp')
            grasparg, rectSlide, colorrectslide = utils.GraspBaseFeature2(depth_heightmap, color_heightmap, 200)
        grapsPoint = [int(grasparg[0] * 34. / 240.), int(grasparg[1] * 34. / 240.)]
        graspangle = grasparg[2]
        succ = rbt.grasp(grapsPoint,graspangle)
        cv2.imwrite(data_addr_base + 'color_rect_image/color_rect_img_%d.jpg' % num, colorrectslide)
        cv2.imwrite(data_addr_base + 'depth_rect_image/depth_rect_img_%d.jpg' % num, rectSlide[:, :] / 0.12 * 255)
        cv2.imwrite(data_addr_base + 'color_image/color_img_%d.jpg' % num, color_heightmap)
        cv2.imwrite(data_addr_base + 'depth_image/depth_img_%d.jpg' % num, depth_heightmap[:, :] / 0.12 * 255)


        if succ == True:
            lable_list=lable_list+'depth_rect_img_%d.jpg' % num+"   1"+"\n"
            print('grasp success !')

        else:
            lable_list=lable_list+'depth_rect_img_%d.jpg' % num + "   0"+"\n"
            print('grasp failed !')
       

        graspcount+=1
        num+=1;
        storelist[0]=num
        pickle.dump(storelist, open('/home/zhsy/work/DataSet/grasp2/data/ind.txt', 'wb'))
        print(time.strftime("-------->>>>>>>%Y-%m-%d %H:%M:%S", time.localtime()))
        print("num=%d"%num)
        if num%3==0:
            f = open(data_addr_base+'lable.txt', 'a')
            f.write(lable_list)
            f.close()
            lable_list=""
            pass



if __name__ == '__main__':
    main()