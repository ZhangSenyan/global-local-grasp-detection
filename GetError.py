# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer
This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.
Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.
Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs


from queue import Queue
from threading import Thread
from predictor import Predictor

device1_num='831612073809' #zhsy
device2_num='831612073761' #liu

device=2

depth_heightmap = np.zeros((240, 240), dtype=np.float)
out = np.zeros((240, 240, 3), dtype=np.uint8)


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
if device==1:
    config.enable_device(device1_num)
    T_W2C = np.load("T_W2C1.npy")
    errsavepath="err1.npy"
elif device==2:
    config.enable_device(device2_num)
    T_W2C = np.load("T_W2C2.npy")
    errsavepath="err2.npy"
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()



R_W2C = T_W2C[:-1, :-1]
P_W2C = T_W2C[:-1, -1]

def view(v):
    # print(v.shape)
    VT = np.dot(R_W2C, v.T).T + P_W2C
    return VT



while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())

    color_source = color_image

    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)

    # Pointcloud data to arrays
    v, t = points.get_vertices(), points.get_texture_coordinates()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    v = view(verts)

    s = v[:, 2].argsort()[::-1]

    proj = v[s][:, :-1] * 705 + np.array([120, 120])

    j, i = proj.astype(np.uint32).T
    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < 240)
    jm = (j >= 0) & (j < 240)
    m = im & jm

    cw, ch = color_source.shape[:2][::-1]
    c, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    np.clip(u, 0, ch - 1, out=u)
    np.clip(c, 0, cw - 1, out=c)
    out[i[m], j[m]] = color_source[u[m], c[m]]

    depth_heightmap[i[m], j[m]] = v[:, 2][s][m]
    #q_p2s.put((depth_heightmap,out))



    # Camera 1 calibration
    (w, h) = (9, 7)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点

    imgpoints = []  # 在图像平面的二维点

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    out_show=out.copy()
    if ret == True:
        # 阈值
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示

        cv2.drawChessboardCorners(out_show, (w, h), corners, ret)
        print(corners[3*9+4])
        print(depth_heightmap[int(corners[3*9+4][0,0]),int(corners[3*9+4][0,1])])
        err=np.array([(corners[3*9+4][0,0]-120)*0.34/240,(corners[3*9+4][0,1]-120)*0.34/240,\
                      depth_heightmap[int(corners[3*9+4][0,0]),int(corners[3*9+4][0,1])]])
        np.save(errsavepath, err)

    cv2.imshow('findCorners2', out_show)
    cv2.waitKey(1)
# Configure depth and color streams




