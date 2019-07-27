# coding : utf-8

# -----------------------------------
# 相机校准
# output：
#     T_W2C1.npy  :  1号相机内参矩阵
#     T_W2C2.npy  :  2号相机内参矩阵
# -----------------------------------

import pyrealsense2 as rs
import numpy as np
import cv2
import utils

#
# 2D图像坐标 -> 3D相机坐标
#
def pixel2XYZ(depthimage,x,y,transin):
    depth=depthimage[y,x]
    p=np.array([x,y,1]).reshape((3,1))*depth
    xyz=np.dot(np.linalg.inv(transin),p)
    return xyz

#
# 求向量余弦 cos(v1,v2)
#
def CosV(vector1,vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


def arrayinv(arr):
    return arr[np.arange(arr.shape[0],0,-1)-1]


device1_num='831612073809' # Camera 1
device2_num='831612073761' # Camera 2

# configuration of camera 1
pipeline1 = rs.pipeline()
config1 = rs.config()
config1.enable_device(device1_num)
config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# configuration of camera 2
pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device(device2_num)
config2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile1 = pipeline1.start(config1)
profile2 = pipeline2.start(config2)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # camera 1 获取深度图和彩色图
        frames1 = pipeline1.wait_for_frames()
        aligned_frames1 = align.process(frames1)
        aligned_depth_frame1 = aligned_frames1.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame1 = aligned_frames1.get_color_frame()
        if not aligned_depth_frame1 or not color_frame1:
            continue
        depth_image1 = np.asanyarray(aligned_depth_frame1.get_data())
        color_image1 = np.asanyarray(color_frame1.get_data())

        (w, h)=(9,7)
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点

        gray = cv2.cvtColor(color_image1, cv2.COLOR_BGR2GRAY)

        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        # 如果找到足够点对，将其存储起来
        if ret==True:

            # 阈值
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

            # 将角点在图像上显示
            cv2.drawChessboardCorners(color_image1, (w, h), corners, ret)

            depth_intrinsics = rs.video_stream_profile(
                aligned_depth_frame1.profile).get_intrinsics()

            # 内参矩阵
            transin = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                [0, 0, 1]])

            corners_mat=np.zeros((h,w,3))
            for i in range(h):
                for j in range(w):
                    xyz = pixel2XYZ(depth_image1, int(corners[i*w+j,0,0]), int(corners[i*w+j,0,1]), transin)
                    corners_mat[i,j]=xyz.reshape(3)
            Y_axis=np.array([0.,0.,0.])

            for i in range(h):
                Y_axis+=(corners_mat[i,0]-corners_mat[i,6])

            Y_axis=Y_axis/h
            Y_axis=Y_axis/np.linalg.norm(Y_axis)

            print("Y_axis1:\n",Y_axis)

            X_axis = np.array([0., 0., 0.])
            for i in range(w):
                X_axis += (corners_mat[0, i] - corners_mat[h-1, i])

            X_axis = X_axis / w
            X_axis = X_axis / np.linalg.norm(X_axis)
            print("X_axis1:\n",X_axis)

            Z_axis = np.cross(X_axis, Y_axis)
            print("Z_axis1:\n",Z_axis)

            center_coor_3D=corners_mat[3,4]/1000
            print("center_coor_3D1:\n",center_coor_3D)

            X = np.array([1, 0, 0])
            Y = np.array([0, 1, 0])
            Z = np.array([0, 0, 1])

            #
            # 计算转换矩阵：相机坐标系 -> 世界坐标系
            #
            T_C2W = np.array([
                [CosV(X_axis, X), CosV(Y_axis, X), CosV(Z_axis, X), center_coor_3D[0]],
                [CosV(X_axis, Y), CosV(Y_axis, Y), CosV(Z_axis, Y), center_coor_3D[1]],
                [CosV(X_axis, Z), CosV(Y_axis, Z), CosV(Z_axis, Z), center_coor_3D[2]],
                [0, 0, 0, 1]
            ])

            # 转换矩阵求逆： 世界坐标系 -> 相机坐标系
            T_W2C = np.linalg.inv(T_C2W)

            print("T_W2C1:\n",T_W2C)

            np.save("./model/T_W2C1.npy", T_W2C)

        cv2.imshow('findCorners1', color_image1)

        cv2.waitKey(1)





        # camera 2 获取深度图和彩色图
        frames2 = pipeline2.wait_for_frames()
        aligned_frames2 = align.process(frames2)
        aligned_depth_frame2 = aligned_frames2.get_depth_frame()
        color_frame2 = aligned_frames2.get_color_frame()

        if not aligned_depth_frame2 or not color_frame2:
            continue

        depth_image2 = np.asanyarray(aligned_depth_frame2.get_data())
        color_image2 = np.asanyarray(color_frame2.get_data())

        (w, h) = (9, 7)
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点

        gray = cv2.cvtColor(color_image2, cv2.COLOR_BGR2GRAY)

        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        # 如果找到足够点对，将其存储起来
        if ret == True:

            # 阈值
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

            # 将角点在图像上显示
            corners=arrayinv(corners)
            cv2.drawChessboardCorners(color_image2, (w, h), corners, ret)

            depth_intrinsics = rs.video_stream_profile(
                aligned_depth_frame2.profile).get_intrinsics()

            transin = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                [0, 0, 1]])

            corners_mat = np.zeros((h, w, 3))
            for i in range(h):
                for j in range(w):
                    xyz = pixel2XYZ(depth_image2, int(corners[i * w + j, 0, 0]), int(corners[i * w + j, 0, 1]), transin)
                    corners_mat[i, j] = xyz.reshape(3)

            Y_axis = np.array([0., 0., 0.])
            for i in range(h):
                Y_axis += (corners_mat[i, 0] - corners_mat[i, 6])

            Y_axis = Y_axis / h
            Y_axis = Y_axis / np.linalg.norm(Y_axis)
            print("Y_axis2:\n", Y_axis)

            X_axis = np.array([0., 0., 0.])
            for i in range(w):
                X_axis += (corners_mat[0, i] - corners_mat[h - 1, i])

            X_axis = X_axis / w
            X_axis = X_axis / np.linalg.norm(X_axis)
            print("X_axis2:\n", X_axis)

            Z_axis = np.cross(X_axis, Y_axis)
            print("Z_axis2:\n", Z_axis)

            center_coor_3D = corners_mat[3, 4] / 1000
            print("center_coor_3D2:\n", center_coor_3D)

            X = np.array([1, 0, 0])
            Y = np.array([0, 1, 0])
            Z = np.array([0, 0, 1])

            #
            # 计算转换矩阵：相机坐标系 -> 世界坐标系
            #
            T_C2W = np.array([
                [CosV(X_axis, X), CosV(Y_axis, X), CosV(Z_axis, X), center_coor_3D[0]],
                [CosV(X_axis, Y), CosV(Y_axis, Y), CosV(Z_axis, Y), center_coor_3D[1]],
                [CosV(X_axis, Z), CosV(Y_axis, Z), CosV(Z_axis, Z), center_coor_3D[2]],
                [0, 0, 0, 1]
            ])

            # 转换矩阵求逆： 世界坐标系 -> 相机坐标系
            T_W2C = np.linalg.inv(T_C2W)

            print("T_W2C2:\n", T_W2C)

            np.save("./model/T_W2C2.npy", T_W2C)

        cv2.imshow('findCorners2', color_image2)
        cv2.waitKey(1)

finally:
    pipeline1.stop()
    pipeline2.stop()