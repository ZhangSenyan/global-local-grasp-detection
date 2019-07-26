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
import socket

from queue import Queue
from threading import Thread
from predictor import Predictor
#from robotiq_close import Robotiq85GripperTestClose  as robotiq_close

device1_num='831612073809' #zhsy
device2_num='831612073761' #liu
#os.system("xfce4-terminal -e './init.sh'")
time.sleep(5)
# close
#robotiq_close(False, True)
# open
#robotiq_close(True, False)


s=False
depth_heightmap = np.zeros((240, 240), dtype=np.float)
out1 = np.zeros((240, 240, 3), dtype=np.uint8)
out2 = np.zeros((240, 240, 3), dtype=np.uint8)
color_show = np.zeros((240, 240, 3), dtype=np.uint8)
out1_flag=np.zeros((240, 240), dtype=np.uint8)
#pre = Predictor()
gconf=[120,120,0.3,0.3,0]
#import matplotlib.pyplot as plt
from UR5 import Robot



def left_vision_capturer(devicenum,depth_heightmap,out,flag,T_W2C,err,color,id=0):

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(devicenum)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Processing blocks
    pc = rs.pointcloud()
    colorizer = rs.colorizer()


    R_W2C = T_W2C[:-1, :-1]
    P_W2C = T_W2C[:-1, -1]-err
    #print(T_W2C)
    #print(P_W2C)
    #print(err)
    def view(v):
        # print(v.shape)
        VT = np.dot(R_W2C, v.T).T + P_W2C
        return VT



    #depth_heightmap = np.empty((240, 240), dtype=np.float)
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

        s = (-v[:, 2]).argsort()[::-1]

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
        if color:
            flag[i[m], j[m]]=10
            out[i[m], j[m]] = color_source[u[m], c[m]]
        flag[flag>0]-=1

        depth_heightmap[i[m], j[m]] = v[:, 2][s][m]

        #q_p2s.put((depth_heightmap,out))


def right_vision_capturer(devicenum, depth_heightmap, out, T_W2C, err, color, id=0):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(devicenum)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Processing blocks
    pc = rs.pointcloud()
    colorizer = rs.colorizer()

    R_W2C = T_W2C[:-1, :-1]
    P_W2C = T_W2C[:-1, -1] - err

    # print(T_W2C)
    # print(P_W2C)
    # print(err)
    def view(v):
        # print(v.shape)
        VT = np.dot(R_W2C, v.T).T + P_W2C
        return VT

    # depth_heightmap = np.empty((240, 240), dtype=np.float)
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

        s = (-v[:, 2]).argsort()[::-1]

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
        if color:
            out[i[m], j[m]] = color_source[u[m], c[m]]

        depth_heightmap[i[m], j[m]] = v[:, 2][s][m]

        # q_p2s.put((depth_heightmap,out))

def get_grasppoint(patch):
    patch = np.sum(patch.reshape(36, 72)[17:20, :], axis=0) / 3.
    # print(patch)
    err = np.max(patch) - np.min(patch)
    for i in range(30):
        if patch[36 - i] - patch[36 - i - 5] > err / 3.:
            leftpoint = 36 - i
            break
    for i in range(30):
        if patch[36 + i] - patch[36 + i + 5] > err / 3.:
            rightpoint = 36 + i
            break
    #print(rightpoint - leftpoint)

def view_show(depth_heightmap,out1,out2,flag,gconf):
    pre = Predictor()
    while True:

        np.clip(depth_heightmap, 0, 0.2,out=depth_heightmap)
        depth_heightmap_used=depth_heightmap[range(239,-1,-1),:]

        location, angle, PlanningTime, lable, depth,patch=pre.GCNN_LCNN(depth_heightmap_used)
        gconf[0]=location[0]
        gconf[1]=location[1]
        gconf[2]=angle
        gconf[3]=depth

        get_grasppoint(patch)
        #print(gconf)

        depth_out = ((depth_heightmap_used+0.04)*1000).astype(np.uint8)

        color_1=out1.copy()
        index=flag==0
        color_1[index]=out2[index]
        color_1=color_1[range(239,-1,-1),:]



        cv2.line(color_1, (int(location[1]+36*math.cos(angle*math.pi/180.)), int(location[0]+36*math.sin(angle*math.pi/180.))),
                 (int(location[1] - 36 * math.cos(angle*math.pi/180.)), int(location[0] - 36 * math.sin(angle*math.pi/180.))), (0, 0, 255), 3)


        cv2.imshow("color1",color_1)
        cv2.imshow("image",depth_out)


        graspability=(cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(lable)))*255).astype(np.uint8)
        graspability=cv2.applyColorMap(graspability, cv2.COLORMAP_JET)

        cv2.imshow("depth",graspability)


        key = cv2.waitKey(1)

gripper_length=0.187
pale_length=0.155
margin=0.015

#remove_to_no([0,0,0], -180)
p0=np.array([0.456,-0.054,0.154])
def action_exe(gconf):
    ur5 = Robot()
    angle = 0
    print("init start")
    ur5.m_move_to([0.161, -0.109, 0.478], -90, False)
    depth_heightmap[:, :] = 0
    print("init end")
    ur5.open_gripper()
    time.sleep(3)
    while True:

        location = [gconf[0], gconf[1]]
        #location=[120,120]
        angle = -90 - gconf[2]

        # 坐标转换
        to_position0 = -location[0] * 0.34 / 240. + 0.17
        to_position1 = -location[1] * 0.34 / 240. + 0.17
        to_depth = gconf[3] -0.05
        if to_depth<0:
            to_depth=0
        print(to_depth)
        ur5.m_move_to([to_position0, to_position1, 0.2], angle, True)
        ur5.m_move_to([to_position0, to_position1, to_depth], angle, True)
        ur5.close_gripper()
        time.sleep(3)
        ur5.m_move_to([to_position0, to_position1, 0.2], angle, True)
        #time.sleep(3)
        ur5.m_move_to([0.036, -0.533, 0.35], -90, False)

        depth_heightmap[:, :] = 0
        ur5.open_gripper()
        time.sleep(1)


T_W2C1=np.load("T_W2C1.npy")
T_W2C2=np.load("T_W2C2.npy")
err1=np.load("err1.npy")
err2=np.load("err2.npy")
#err2=np.load("err2.npy")+np.array([0.00708333,0,0])
print(err2)

t1 = Thread(target=left_vision_capturer, args=(device1_num,depth_heightmap,out1,out1_flag,T_W2C1,err1,True,1))
t2 = Thread(target=right_vision_capturer, args=(device2_num,depth_heightmap,out2,T_W2C2,err2,True,0))
t3 = Thread(target=view_show,args=(depth_heightmap,out1,out2,out1_flag,gconf))
print(type(depth_heightmap))
t4 = Thread(target=action_exe, args=(gconf,))

t1.start()
t2.start()
t3.start()
time.sleep(5)
t4.start()

# Configure depth and color streams




