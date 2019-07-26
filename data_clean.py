

import utils
import cv2

First_run=True

def main():

    f=open("/home/zhsy/work/DataSet/grasp3/lable.txt",'r')
    f_clean = open('/home/zhsy/work/DataSet/grasp3/data_clean/lable_clean.txt', 'a')

    lines=f.readlines()
    for line in lines:
        print (line)
        ind=line.split()[0][10:]

        img = cv2.imread("/home/zhsy/work/DataSet/grasp3/depth_rect_image/depth_rect_img_"+ind ,
                         cv2.IMREAD_GRAYSCALE)

        score = utils.getGraspRectScore(img)
        print(score)
        label=line.split()[1]
        print(label)
        if score>800 and label=='1':
            f_clean.write(line)
            cv2.imwrite("/home/zhsy/work/DataSet/grasp3/data_clean/1/"+line.split()[0],img)
        if score<600 and label=='0':
            f_clean.write(line)
            cv2.imwrite("/home/zhsy/work/DataSet/grasp3/data_clean/0/" + line.split()[0],img)
        print(line)

    f_clean.close()




if __name__ == '__main__':
    main()