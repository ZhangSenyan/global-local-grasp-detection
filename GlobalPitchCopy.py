# coding : utf-8
import os
import random
import cv2


def main():
    path="/home/zhsy/work/DataSet/grasp2/lable_square.txt"
    file=open(path,'r')
    lines=file.readlines()
    random.shuffle(lines)
    for line in lines:
        print(line)
        line.split()
        #print("/home/zhsy/work/DataSet/grasp2/depth_square_image/"+line.split()[0])
        image = cv2.imread("/home/zhsy/work/DataSet/grasp2/depth_square_image/depth_square_img_"+line.split()[0][19:],cv2.IMREAD_GRAYSCALE)
        #print(image)
        cv2.imshow("image",image)
        cv2.waitKey(0)

    pass


if __name__ == '__main__':
    main()