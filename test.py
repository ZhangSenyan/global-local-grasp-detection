import os
import time
import rospy
from gripper.robotiq_close import Robotiq85GripperTestClose  as robotiq_close


os.system("xfce4-terminal -e './init.sh'")
time.sleep(12)
rospy.init_node('calibrate')
# close
robotiq_close(False, True)
# open
# robotiq_close(True, False)