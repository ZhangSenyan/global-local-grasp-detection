#!/usr/bin/env python
# license removed for brevity
import os
import time
import rospy
from gripper.robotiq_close import Robotiq85GripperTestClose  as robotiq_close
import socket

os.system("xfce4-terminal -e './init.sh'")
time.sleep(10)
rospy.init_node('calibrate')
# close

address = ('127.0.0.1', 31501)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(address)
while True:
    data, addr = s.recvfrom(2048)
    if not data:
        print "client has exist"
        break
    print data
    if data == 'open':
        print 'have open'
        robotiq_close(True, False)
    if data == 'close':
        robotiq_close(False, True)
    pass
