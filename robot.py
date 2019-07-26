import time
import os
import numpy as np
from simulation import vrep
import math
#from PIL import Image
import utils
import random
import operator
from tkinter import messagebox


class Robot(object):

    def __init__(self):

        vrep.simxFinish(-1)  # just in case, close all opened connections
        #self.clientID = vrep.simxStart('10.20.5.229', 19997, True, True, -500000,5)  # Connect to V-REP, set a very large time-out for blocking commands
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, -500000, 5)  # Connect to V-REP, set a very large time-out for blocking commands
        if self.clientID != -1:
            print('Connected to remote API server')
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
            time.sleep(1)
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
            time.sleep(1)
            print('has Start')
            self.env_prepare()
        else:
            print('Failed connecting to remote API server')

    def simrestart(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        time.sleep(1)
        print("simulation restart")
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        time.sleep(1)



    def env_prepare(self):
        sim_ret, self.UR3_target_handle = vrep.simxGetObjectHandle(self.clientID, 'UR3_Target',
                                                                   vrep.simx_opmode_blocking)
        sim_ret, self.WorkSpace_handle = vrep.simxGetObjectHandle(self.clientID, 'WorkSpace', vrep.simx_opmode_blocking)
        sim_ret, self.WorkSpace_position = vrep.simxGetObjectPosition(self.clientID, self.WorkSpace_handle, -1,
                                                                      vrep.simx_opmode_blocking)
        print("WorkSpace_position",self.WorkSpace_position)
        self.WorkSpace_size=0.34   # 0.36*0.36
        self.workspace_limits=np.array([[self.WorkSpace_position[0] - self.WorkSpace_size/2,
                                         self.WorkSpace_position[0] + self.WorkSpace_size / 2],
                                        [self.WorkSpace_position[1] - self.WorkSpace_size / 2,
                                         self.WorkSpace_position[1] + self.WorkSpace_size / 2]])
        self.move_abs_ownframe([self.WorkSpace_size/2,self.WorkSpace_size/2,0.3])
        self.setangle_360(0)
        self.setVisionSensor()

        self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                                       [89.0, 161.0, 79.0],  # green
                                       [156, 117, 95],  # brown
                                       [242, 142, 43],  # orange
                                       [237.0, 201.0, 72.0],  # yellow
                                       [186, 176, 172],  # gray
                                       [255.0, 87.0, 89.0],  # red
                                       [176, 122, 161],  # purple
                                       [118, 183, 178],  # cyan
                                       [255, 157, 167]])/255   # pink




    def getangle_360(self):
        sim_ret, UR3_target_orientation = vrep.simxGetObjectOrientation(self.clientID,
                                                                        self.UR3_target_handle, -1,
                                                                        vrep.simx_opmode_blocking)
        if (UR3_target_orientation[0] < 0):
            UR3_target_angle_360 = -(UR3_target_orientation[1] + math.pi / 2)
        else:
            UR3_target_angle_360 = UR3_target_orientation[1] + math.pi / 2
        return UR3_target_angle_360/math.pi*180

    def setangle_360(self,angle):
        angle=angle/180*math.pi
        if (angle >= 0):
            vrep.simxSetObjectOrientation(self.clientID, self.UR3_target_handle,
                                          -1, (math.pi / 2, angle - math.pi / 2, math.pi / 2), vrep.simx_opmode_blocking)
        else:
            vrep.simxSetObjectOrientation(self.clientID, self.UR3_target_handle,
                                          -1, (-math.pi / 2, -angle - math.pi / 2, -math.pi / 2), vrep.simx_opmode_blocking)



    def rotate_gripper(self,angle,da_step=10):

        angle=-angle%180
        #print("angle",angle)


        #UR3_target_orientation -> 360

        UR3_target_angle_360=self.getangle_360()
        to_angle_360=UR3_target_angle_360;

        UR3_target_angle_180=UR3_target_angle_360%180

        #print("UR3_target_angle:",UR3_target_angle_360)


        if (abs(angle-UR3_target_angle_180)>90):
            if angle>UR3_target_angle_180:
                da=-da_step
            else:
                da=da_step
        else:
            if angle>UR3_target_angle_180:
                da=da_step
            else:
                da=-da_step
        #print("da:",da)


        while True:

            to_angle_360=self.getangle_360()
            if ( abs(to_angle_360%180- angle) <= da_step):
                 break

            to_angle_360=to_angle_360+da
            if (to_angle_360>180):
                to_angle_360=to_angle_360-180
            if(to_angle_360<-180):
                to_angle_360=to_angle_360+180
            self.setangle_360( to_angle_360)





    def move_relative(self, dp,step=0.02):
        '''
        :param dp: [dx,dy,dz]
        :param step: step lenght
        :return:
        '''

        move_direction = np.asarray(dp)
        move_magnitude = np.linalg.norm(move_direction)
        move_step = step * move_direction / move_magnitude
        num_move_steps = int(np.floor(move_magnitude / step))

        for step_iter in range(num_move_steps):

            sim_ret, UR3_target_position = vrep.simxGetObjectPosition(self.clientID,
                                                                      self.UR3_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)

            vrep.simxSetObjectPosition(self.clientID, self.UR3_target_handle, -1, (
            UR3_target_position[0] + move_step[0], UR3_target_position[1] + move_step[1],
            UR3_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)



    def move_abs_world(self, target_position):

        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR3_target_position = vrep.simxGetObjectPosition(self.clientID, self.UR3_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray([target_position[0] - UR3_target_position[0], target_position[1] - UR3_target_position[1],
                                     target_position[2] - UR3_target_position[2]])
        self.move_relative(move_direction)


    def move_abs_ownframe(self,target_position_image):


        target_position=[target_position_image[1],self.WorkSpace_size-target_position_image[0],target_position_image[2]]
        #print(target_position)


        if(target_position[0]>self.WorkSpace_size):
            target_position[0] = self.WorkSpace_size
        if (target_position[0] <0):
            target_position[0] = 0
        if (target_position[1]> self.WorkSpace_size):
            target_position[1] = self.WorkSpace_size
        if (target_position[0] < 0):
            target_position[0] = 0

        target_position_world=[self.WorkSpace_position[0]-self.WorkSpace_size/2+target_position[0],
                               self.WorkSpace_position[1] - self.WorkSpace_size / 2 + target_position[1],
                               target_position[2]]

        self.move_abs_world(target_position_world)

    def open_gripper(self):

        motorVelocity = 0.5   # m / s
        motorForce = 20   # N
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.clientID, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.clientID, RG2_gripper_handle, motorForce, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.clientID, RG2_gripper_handle, motorVelocity, vrep.simx_opmode_blocking)
        time.sleep(0.2)


    def close_gripper(self):

        motorVelocity = -0.5   # m / s
        motorForce = 200   # N
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.clientID, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.clientID, RG2_gripper_handle, motorForce, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.clientID, RG2_gripper_handle, motorVelocity, vrep.simx_opmode_blocking)
        time.sleep(0.2)



    def setVisionSensor(self):
        sim_ret, self.VS_left_handle = vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor_left',
                                                                vrep.simx_opmode_blocking)
        ret,self.VS_left_position= vrep.simxGetObjectPosition(self.clientID, self.VS_left_handle, -1,
                                                              vrep.simx_opmode_blocking)
        ret, self.VS_left_orientation = vrep.simxGetObjectOrientation(self.clientID, self.VS_left_handle, -1,
                                                                      vrep.simx_opmode_blocking)
        self.left_cam_intrinsics = np.asarray([[561.82, 0, 320], [0, 561.82, 240], [0, 0, 1]])  # 内部参数
        self.leftmat=utils.Creat_posemat(self.VS_left_orientation,self.VS_left_position)

        sim_ret, self.VS_right_handle = vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor_right',
                                                                 vrep.simx_opmode_blocking)
        ret, self.VS_right_position = vrep.simxGetObjectPosition(self.clientID, self.VS_right_handle, -1,
                                                                 vrep.simx_opmode_blocking)
        ret, self.VS_right_orientation = vrep.simxGetObjectOrientation(self.clientID, self.VS_right_handle, -1,
                                                                       vrep.simx_opmode_blocking)
        self.right_cam_intrinsics = np.asarray([[561.82, 0, 320], [0, 561.82, 240], [0, 0, 1]])  # 内部参数
        self.rightmat = utils.Creat_posemat(self.VS_right_orientation, self.VS_right_position)

    def get_VS_image(self,VS_handle):
        # Get color image from simulation

        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.clientID, VS_handle, 0,
                                                                       vrep.simx_opmode_blocking)
        if sim_ret==0:
            color_img = np.array(raw_image).reshape((resolution[1], resolution[0], 3)).astype(np.uint8)
        else:
            return 8,0,0

        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, VS_handle,
                                                                                vrep.simx_opmode_blocking)
        if sim_ret==0:
            depth_img = np.asarray(depth_buffer).reshape((resolution[1], resolution[0])).astype(np.double)
        else:
            return 8,0,0

        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear
        return 0,color_img,depth_img

    def get_height_img(self,image_size=(240,240)):
        ret,color_img_left, depth_img_left = self.get_VS_image(self.VS_left_handle)
        if ret==0:
            surface_pts_left, rgb_pts_left=utils.get_pointcloud(color_img_left, depth_img_left, self.left_cam_intrinsics, self.leftmat)
        else:
            return 8,0,0

        ret,color_img_right, depth_img_right = self.get_VS_image(self.VS_right_handle)
        if ret==0:
            surface_pts_right, rgb_pts_right = utils.get_pointcloud(color_img_right, depth_img_right,self.right_cam_intrinsics, self.rightmat)
        else:
            return 8,0,0

        surface_pts=np.concatenate((surface_pts_left, surface_pts_right), axis=0)
        color_pts = np.concatenate((rgb_pts_left, rgb_pts_right), axis=0)
        #surface_pts=surface_pts_left
        #color_pts=rgb_pts_left


        #surface_pts = surface_pts_right
        #color_pts = rgb_pts_right



        heightmap_size = np.round(image_size).astype(int)

        # Filter out surface points outside heightmap boundaries
        heightmap_valid_ind = np.logical_and(np.logical_and(
            np.logical_and(surface_pts[:, 0] >= self.workspace_limits[0][0], surface_pts[:, 0] < self.workspace_limits[0][1]),
            surface_pts[:, 1] >= self.workspace_limits[1][0]), surface_pts[:, 1] < self.workspace_limits[1][1])
        surface_pts = surface_pts[heightmap_valid_ind]
        color_pts = color_pts[heightmap_valid_ind]

        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        depth_heightmap = np.zeros(heightmap_size)

        heightmap_pix_x = np.floor((surface_pts[:, 0] - self.workspace_limits[0][0]) / (self.workspace_limits[0][1]-self.workspace_limits[0][0])*heightmap_size[0]).astype(int)
        heightmap_pix_y = np.floor((surface_pts[:, 1] - self.workspace_limits[1][0]) / (self.workspace_limits[1][1]-self.workspace_limits[1][0])*heightmap_size[1]).astype(int)
        color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
        color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
        color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]

        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
        depth_heightmap[depth_heightmap < 0] = 0
        return 0,color_heightmap, depth_heightmap


    def add_objects(self,num):

        basePath="/home/zhsy/work/V_rep/Model_House/"
        shapes= os.listdir(basePath)
        #print(shapes)
        self.object_handles=[]
        for i in range(num):

            curr_mesh_file = os.path.join(basePath, shapes[random.randint(0,len(shapes)-1)])
            print(curr_mesh_file)

            object_position = [random.uniform(self.workspace_limits[0][0], self.workspace_limits[0][1]),
                               random.uniform(self.workspace_limits[1][0], self.workspace_limits[1][1]), 0.15]
            object_orientation = [random.uniform(-math.pi,math.pi),
                                  random.uniform(-math.pi, math.pi),
                                  random.uniform(-math.pi, math.pi)]

            #print(object_color)
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.clientID,
                            'RemoteServer', vrep.sim_scripttype_childscript, 'importTTM', [0, 0, 255, 0], object_position
                                                                                                  + object_orientation, [curr_mesh_file], bytearray(),
                                                                                                  vrep.simx_opmode_blocking)
            #ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.clientID,
            #                 'RemoteServer',vrep.sim_scripttype_childscript,'importTTM',[0, 0, 255, 0],[0,0,0.15]
            #                 + [0,0,0],[curr_mesh_file],bytearray(),vrep.simx_opmode_blocking)

            if ret_resp == 0:
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
            else:
                print('Failed to add new objects to simulation')
                #print msg
                print("ret_resp:",ret_resp)
                print("ret_ints:", ret_ints)
                print("ret_floats:", ret_floats)
                print("ret_strings:", ret_strings)
                print("ret_buffer:", ret_buffer)
                print("object_position:",object_position)
                print("object_orientation:",object_orientation)
                messagebox.showerror("showerror", "错误信息框！")
                while True:
                    str = input("请输入：");
                    if operator.eq(str, "zhsy4631"):
                        break
                time.sleep(10)
                self.simrestart()

            time.sleep(1)

    def grasp(self,postion,angle):
        postion = [float(postion[0]) * 0.01, float(postion[1]) * 0.01]
        print("grsping start")
        if postion[0]<0:
            postion[0]=0
            print("out manpulate range")
        if postion[0]>0.34:
            postion[0]=0.34
            print("out manpulate range")
        if postion[1]<0:
            postion[1]=0
            print("out manpulate range")
        if postion[1]>0.34:
            postion[1]=0.34
            print("out manpulate range")

        #sim_ret, UR3_target_position = vrep.simxGetObjectPosition(self.clientID, self.UR3_target_handle, -1,vrep.simx_opmode_blocking)
        #print(UR3_target_position)
        #self.move_abs_ownframe([0.15, 0.15, 0.2])
        #self.move_relative([0,0,0.2-UR3_target_position[2]],step=0.02)
        self.move_abs_ownframe([0.17, 0.17, 0.15])
        self.move_abs_ownframe([postion[0], postion[1], 0.15])
        self.rotate_gripper(angle)
        self.open_gripper()

        self.move_abs_ownframe([postion[0], postion[1], 0.01])
        self.close_gripper()
        time.sleep(0.5)
        self.move_abs_ownframe([postion[0], postion[1], 0.15])
        succ=self.checkgrasp()
        self.move_abs_ownframe([0.17, 0.17, 0.25])
        print("grsping end")

        return succ

    def checkgrasp(self):
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.clientID, 'RG2_openCloseJoint',
                                                               vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.clientID, RG2_gripper_handle,
                                                                    vrep.simx_opmode_blocking)

        if gripper_joint_position > -0.047:
            self.remove_height_object()
            return True
        else:
            return False

    def remove_height_object(self):
        maxhandle = self.object_handles[0]
        sim_ret, obj_position = vrep.simxGetObjectPosition(self.clientID, self.object_handles[0], -1,
                                                           vrep.simx_opmode_blocking)
        maxposition = obj_position[2]
        for i in range(1, len(self.object_handles)):
            sim_ret, obj_position = vrep.simxGetObjectPosition(self.clientID, self.object_handles[i], -1,
                                                               vrep.simx_opmode_blocking)
            if obj_position[2] > maxposition:
                maxposition = obj_position[2]
                maxhandle = self.object_handles[i]

        #print(obj_position)
        vrep.simxSetObjectPosition(self.clientID, maxhandle, -1, (0.5, -0.2, 0.3), vrep.simx_opmode_blocking)


    def getimage_var(self,depth_heightmap):


        slideRectSize=[int(depth_heightmap.shape[0]/2),int(depth_heightmap.shape[1]/2)]

        denseSet=[]
        for i in range(int((depth_heightmap.shape[0]-slideRectSize[0])/4)):
            for j in range(int((depth_heightmap.shape[1] - slideRectSize[1]) / 4)):
                slideimage = depth_heightmap[i*4:i*4 + slideRectSize[0],j*4:j*4 + slideRectSize[1]]
                dense = np.sum(np.sum(slideimage, axis=0), axis=0)
                denseSet.append(dense)
        image_var=np.var(np.array(denseSet))


        return image_var

    def push(self,postion,angle,dp=0.1):
        sim_ret, UR3_target_position = vrep.simxGetObjectPosition(self.clientID, self.UR3_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        #self.move_abs_ownframe([0.15, 0.15, 0.2])
        self.move_relative([0,0,0.2-UR3_target_position[2]],step=0.02)
        self.move_abs_ownframe([postion[0], postion[1], 0.2])
        self.rotate_gripper(angle+90)
        self.move_abs_ownframe([postion[0], postion[1], 0.002])
        rad=angle/180*math.pi
        to_x=postion[0]+dp*math.cos(rad)
        to_y=postion[1]+dp*math.sin(rad)
        if(to_x > self.WorkSpace_size):
            to_x = self.WorkSpace_size
        if (to_x < 0):
            to_x = 0
        if (to_y > self.WorkSpace_size):
            to_y = self.WorkSpace_size
        if (to_y < 0):
            to_y = 0
        #print("to_x",to_x,"to_y",to_y)
        self.move_abs_ownframe([to_x,to_y , 0.002])
        self.move_abs_ownframe([to_x,to_y , 0.2])

    def checkState(self):
        sim_ret, UR3_target_position = vrep.simxGetObjectPosition(self.clientID,
                                                                  self.UR3_target_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.clientID, 'RG2_openCloseJoint',
                                                               vrep.simx_opmode_blocking)
        sim_ret, RG2_gripper_position = vrep.simxGetObjectPosition(self.clientID,
                                                                   RG2_gripper_handle, -1, vrep.simx_opmode_blocking)

        #print("===================================================================")
        #print(UR3_target_position)
        #print(RG2_gripper_position)
        dist=pow(pow(UR3_target_position[0]-RG2_gripper_position[0],2)+\
             pow(UR3_target_position[1] - RG2_gripper_position[1], 2) +\
             pow(UR3_target_position[2] - RG2_gripper_position[2], 2),0.5)

        if dist>0.1:
            return True #异常状态
        else:
            return False








