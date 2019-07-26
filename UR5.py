import socket
import struct
import time
import numpy as np
import math

from multiprocessing.connection import Listener
from multiprocessing.connection import Client
import os



class multi_connection(object):
    def __init__(self, conn_type='client', host='127.0.0.200', port_num=8086, bufsize=4096):
        self.host = host
        self.port_num = port_num
        self.BUFSIZE = bufsize

        self.connect_flag =         [110]
        self.end_data_flag =        [111]

        if conn_type == 'server':
            self.conn = self.create_connect_server()
            recv_data = self.conn.recv()
            if self.recv_flag(recv_data):
                print('Server is connecting with the client.')
        elif conn_type == 'client':
            self.conn = self.create_connect_client()
            self.send_flag(self.connect_flag)

        #os.system("xfce4-terminal -e './init.sh'")
        #time.sleep(10)
        #rospy.init_node('calibrate')

    def create_connect_server(self):
        address = ('127.0.0.200', self.port_num)
        listener = Listener(address, authkey=b'secret password A')
        conn = listener.accept()
        return conn

    def create_connect_client(self):
        address = (self.host, self.port_num)
        conn = Client(address, authkey=b'secret password A')
        return conn

    def recv_flag(self, input_flag):
        if input_flag[0] == self.connect_flag[0]:
            return 1
        elif input_flag[0] == self.end_data_flag[0]:
            print('Received end_data_flag, so the connection will be closed.')
            self.close()
        else:
            return 0

    def send_flag(self, input_flag):
        self.conn.send(input_flag)

    def recv(self):
        recv_data = self.conn.recv()
        self.recv_flag(recv_data)
        return recv_data

    def send(self, data_list):
        self.conn.send(data_list)

    def close(self):
        self.conn.close()




class Robot(object):
    def __init__(self):
        self.tcp_host_ip="192.168.88.15"
        self.tcp_port=30002
        self.tool_acc=1.5
        self.tool_vel=0.2
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]
        self.p0 = np.array([0.466, -0.037, 0.279])
        os.system("xfce4-terminal -e 'python /home/arm/visual-pushing-grasping/rtde_client_3.5/examples/state_record.py --verbose'")
        time.sleep(2)
        global connn
        connn = multi_connection()
        self.gripper_addr=('127.0.0.1', 31501)


    def move_to(self, tool_position, tool_orientation):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],tool_position[1],tool_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches target tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
            # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)] + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) for j in range(3,6)])
            tcp_state_data = self.tcp_socket.recv(2048)
            prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            time.sleep(0.01)
        self.tcp_socket.close()

    def m_move_to(self, tool_position, angle, coor):
        if coor:
            tool_position = self.p0 + tool_position
        angle = angle * math.pi / 180
        tool_orientation=(math.cos(angle/2)*math.pi,math.sin(angle/2)*math.pi,0)
        self.move_to(tool_position,tool_orientation)
    def close_gripper(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(b'close', self.gripper_addr)
        s.close()
        # close

    def open_gripper(self):
        # open
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(b'open', self.gripper_addr)
        s.close()
    def parse_tcp_state_data(self, state_data, subpackage):

        # # Read package header
        # data_bytes = bytearray()
        # data_bytes.extend(state_data)
        # data_length = struct.unpack("!i", data_bytes[0:4])[0]
        # robot_message_type = data_bytes[4]
        # assert(robot_message_type == 16)
        # byte_idx = 5
        #
        # # Parse sub-packages
        # subpackage_types = {'joint_data' : 1, 'cartesian_info' : 4, 'force_mode_data' : 7, 'tool_data' : 2}
        # while byte_idx < data_length:
        #     # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
        #     package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx+4)])[0]
        #     byte_idx += 4
        #     package_idx = data_bytes[byte_idx]
        #     if package_idx == subpackage_types[subpackage]:
        #         byte_idx += 1
        #         break
        #     byte_idx += package_length - 4
        if state_data:
            request = [999]
            connn.send(request)
            state_data = connn.recv()
            state_data = state_data[0]


        def parse_joint_data():
            # actual_joint_positions = [0,0,0,0,0,0]
            # target_joint_positions = [0,0,0,0,0,0]
            # for joint_idx in range(6):
            #     actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            #     target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+8):(byte_idx+16)])[0]
            #     byte_idx += 41
            actual_joint_positions = state_data[:6]
            return actual_joint_positions

        def parse_cartesian_info():
            # actual_tool_pose = [0,0,0,0,0,0]
            # for pose_value_idx in range(6):
            #     actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            #     byte_idx += 8
            actual_tool_pose = state_data[6:13]
            return actual_tool_pose

        def parse_tool_data():
            # byte_idx += 2
            # tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            #tool_analog_input2 = check_state()
            return 0

        parse_functions = {'joint_data' : parse_joint_data(), 'cartesian_info' : parse_cartesian_info(), 'tool_data' : parse_tool_data()}
        return parse_functions[subpackage]

    def parse_rtc_state_data(self, state_data):

        # Read package header
        # data_bytes = bytearray()
        # data_bytes.extend(state_data)
        # data_length = struct.unpack("!i", data_bytes[0:4])[0];
        # assert(data_length == 812)
        # byte_idx = 4 + 8 + 8*48 + 24 + 120
        # TCP_forces = [0,0,0,0,0,0]
        # for joint_idx in range(6):
        #     TCP_forces[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
        #     byte_idx += 8
        if state_data:
            request = [999]
            connn.send(request)
            state_data = connn.recv()
            state_data = state_data[0]
            TCP_forces = state_data[13:]
        return TCP_forces

    def get_state(self):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        self.tcp_socket.close()
        return state_data


 # init


