# coding : utf-8

# -----------------------------------
# Self-supervision learning
# part 1 ： Collecting data in V-REP
# Steps :
#     1. Latch V-REP
#     2. Add Objects
#     3. Grasping trials
#     4. Record results
#     -> data  : local patch
#     -> lable : success ? true : false
# output：
#     color_rect_image
#     lable.txt
# -----------------------------------

from robot import Robot
import numpy as np
import time
import utils
import cv2
import random
import pickle

def main():


    data_dir = "/home/zhsy/work/GraspDataSet/"

    First_run = True

    if First_run:
        num = 1;
        storelist = [0]
    else:
        storelist = pickle.load(open(data_dir+'ind.txt', 'rb'))
        num=storelist[0]

    rbt = Robot()

    lable_list=""

    add_obj_num = 10

    rbt.add_objects(add_obj_num)

    # 物品添加完后，等待物品静止后进行抓取
    time.sleep(3)

    # get RGB image and depth image
    ret,color_heightmap, depth_heightmap = rbt.get_height_img()

    graspcount=0

    while True:

        # np.sum(depth_heightmap) < 150 所有物品抓取完成
        # rbt.checkState() 出现异常
        # graspcount > (2*add_obj_num)  End of this epoch
        if np.sum(depth_heightmap) < 150 or rbt.checkState() or graspcount > (2*add_obj_num):

            rbt.simrestart()

            add_obj_num=random.randint(5, 15)
            rbt.add_objects(add_obj_num)
            time.sleep(3)

            graspcount = 0


        ret, color_heightmap, depth_heightmap = rbt.get_height_img()

        # 异常
        if ret == 8:
            time.sleep(10)
            continue

        #
        # feature grasp ： u = 1000  Success Rate > 0.8
        # random grasp  ： u = 200   Success Rate < 0.2
        #
        # 为了保证收集的正负样本数量大致相同
        # 因此抓取尝试一半采用 feature grasp， 一半采用random grasp
        # 这里并未使用动态抓取策略
        #
        if(random.randint(0,9)>=5):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'Feature grasp -----')
            grasparg, rectSlide, colorrectslide = utils.GraspBaseFeature2(depth_heightmap,color_heightmap,1000)
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'Random grasp -----')
            grasparg, rectSlide, colorrectslide = utils.GraspBaseFeature2(depth_heightmap, color_heightmap, 200)

        # 计算抓取参数
        grapsPoint = [int(grasparg[0] * 34. / 240.), int(grasparg[1] * 34. / 240.)]
        graspangle = grasparg[2]

        # 执行抓取试验
        succ = rbt.grasp(grapsPoint,graspangle)

        # 记录数据样本
        #
        # color_rect_image ： 局部数据，一个local patch，与抓取时爪子位置对齐
        # depth_rect_image ： 局部数据，深度图
        # color_image      ： 抓取时整个作业空间对应的图像
        # depth_image      ： 作业空间深度图
        cv2.imwrite(data_dir + 'color_rect_image/color_rect_img_%d.jpg' % num, colorrectslide)
        cv2.imwrite(data_dir + 'depth_rect_image/depth_rect_img_%d.jpg' % num, rectSlide[:, :] / 0.12 * 255)
        cv2.imwrite(data_dir + 'color_image/color_img_%d.jpg' % num, color_heightmap)
        cv2.imwrite(data_dir + 'depth_image/depth_img_%d.jpg' % num, depth_heightmap[:, :] / 0.12 * 255)

        # 记录 lable
        if succ == True:
            lable_list=lable_list+'depth_rect_img_%d.jpg' % num+"   1"+"\n"
            print('Result：grasp success !')

        else:
            lable_list=lable_list+'depth_rect_img_%d.jpg' % num + "   0"+"\n"
            print('Result：grasp failed !')

        graspcount+=1
        num+=1;
        storelist[0]=num
        pickle.dump(storelist, open('/home/zhsy/work/DataSet/grasp2/data/ind.txt', 'wb'))
        print("Grasp count = %d"%num)

        # 写入文件
        if num%30==0:
            f = open(data_dir+'lable.txt', 'a')
            f.write(lable_list)
            f.close()
            lable_list=""


if __name__ == '__main__':
    main()