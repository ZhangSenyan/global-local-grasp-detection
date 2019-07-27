

import numpy as np
import os
import random
import cv2
import utils


def main():
    data_path_base = '/home/zhsy/work/DataSet/grasp3/'
    rect_num=0
    dirnum = 0
    for num in range(1,19000):
        lable = np.load(data_path_base + "global_lable1/lable_%d.npy" % num).reshape(24, 24, 1)
        im_gray = cv2.imread(data_path_base + "depth_image/depth_img_%d.jpg" % num, cv2.IMREAD_GRAYSCALE)
        P_list = []
        N_list=[]
        lable_list=""

        for r in range(24):
            for c in range(24):
                mPoint = [int(lable[r, c]), [int(r*10+5), int(c*10+5)]]
                if lable[r, c]>=0.9:
                    P_list.append(mPoint)
                if lable[r, c]<=0.1:
                    N_list.append(mPoint)

        random.shuffle(N_list)

        l=min(len(P_list),len(N_list))
        P_list=P_list[:l]
        N_list=N_list[:l]


        for i in range(l):
            if rect_num%10000 == 0:
                print(rect_num)
                dirnum+=1
                os.mkdir(data_path_base + 'depth_square_image/1/%3d'%dirnum)
                os.mkdir(data_path_base + 'depth_square_image/0/%3d' % dirnum)

            rect_gray = utils.SquareRecttailor(im_gray, P_list[i][1])
            lable_list = lable_list + 'depth_square_image_%d.jpg' % rect_num + "   1" + "\n"
            cv2.imwrite(data_path_base + 'depth_square_image/source/depth_square_img_%d.jpg' % rect_num, rect_gray)
            cv2.imwrite(data_path_base + 'depth_square_image/1/%3d/depth_square_img_%d.jpg' % (dirnum,rect_num), rect_gray)


            rect_num+=1
            rect_gray = utils.SquareRecttailor(im_gray, N_list[i][1])
            lable_list = lable_list + 'depth_square_image_%d.jpg' % rect_num + "   0" + "\n"
            cv2.imwrite(data_path_base + 'depth_square_image/source/depth_square_img_%d.jpg' % rect_num, rect_gray)
            cv2.imwrite(data_path_base + 'depth_square_image/0/%3d/depth_square_img_%d.jpg' % (dirnum,rect_num), rect_gray)
            rect_num += 1



        f = open(data_path_base + 'depth_square_image/lable_square.txt', 'a')
        f.write(lable_list)
        f.close()

if __name__ == '__main__':
    main()
