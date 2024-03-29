"""--------------------------------------------------------------------
COPYRIGHT 2015 Stanley Innovation Inc.


Software License Agreement:

The software supplied herewith by Stanley Innovation Inc. (the "Company")
for its licensed Segway RMP Robotic Platforms is intended and supplied to you,
the Company's customer, for use solely and exclusively with Stanley Innovation
products. The software is owned by the Company and/or its supplier, and is
protected under applicable copyright laws.  All rights are reserved. Any use in
violation of the foregoing restrictions may subject the user to criminal
sanctions under applicable laws, as well as to civil liability for the
breach of the terms and conditions of this license. The Company may
immediately terminate this Agreement upon your use of the software with
any products that are not Stanley Innovation products.

The software was written using Python programming language.  Your use
of the software is therefore subject to the terms and conditions of the
OSI- approved open source license viewable at http://www.python.org/.
You are solely responsible for ensuring your compliance with the Python
open source license.

You shall indemnify, defend and hold the Company harmless from any claims,
demands, liabilities or expenses, including reasonable attorneys fees, incurred
by the Company as a result of any claim or proceeding against the Company
arising out of or based upon:

(i) The combination, operation or use of the software by you with any hardware,
    products, programs or data not supplied or approved in writing by the Company,
    if such claim or proceeding would have been avoided but for such combination,
    operation or use.

(ii) The modification of the software by or on behalf of you

(iii) Your use of the software.

 THIS SOFTWARE IS PROVIDED IN AN "AS IS" CONDITION. NO WARRANTIES,
 WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED
 TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. THE COMPANY SHALL NOT,
 IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL OR
 CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER.

 \file   robotiq_85_test.py

 \brief  Node for testing Robotiq 85 communication

 \Platform: Linux/ROS Indigo
--------------------------------------------------------------------"""
import sys
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")
import rospy
from robotiq_85_msgs.msg import GripperCmd, GripperStat


class Robotiq85GripperTestClose:

    def __init__(self,open1,close1,zero=True):
    
        '''
        self._num_grippers = rospy.get_param('~num_grippers',1)
        self.open = rospy.get_param('~open',True)
        self.close = rospy.get_param('~close', False)
        self._prefix = rospy.get_param('~prefix', None)
        '''
        self._num_grippers = 1
        #self.open = False
        #self.close = True
        self.open = open1
        self.close = close1
        self.zero = zero
        self._prefix = "iqr_"
        #self._prefix = None

        if (self._num_grippers == 1 and self._prefix is None):
            rospy.Subscriber("/gripper/stat", GripperStat, self._update_gripper_stat, queue_size=10)
            self._gripper_pub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)
        elif(self._num_grippers == 1 and self._prefix is not None):
            # rospy.logwarn('gripper prefix = {}'.format(self._prefix))
            rospy.Subscriber("/"+self._prefix+"gripper/stat", GripperStat, self._update_gripper_stat, queue_size=10)
            self._gripper_pub = rospy.Publisher('/'+self._prefix+'gripper/cmd', GripperCmd, queue_size=10)
        elif (self._num_grippers == 2):
            rospy.Subscriber("/left_gripper/stat", GripperStat, self._update_gripper_stat, queue_size=10)
            self._left_gripper_pub = rospy.Publisher('/left_gripper/stat', GripperCmd, queue_size=10)
            rospy.Subscriber("/right_gripper/stat", GripperStat, self._update_right_gripper_stat, queue_size=10)
            self._right_gripper_pub = rospy.Publisher('/right_gripper/cmd', GripperCmd, queue_size=10)
        else:
            rospy.logerr("Number of grippers not supported (needs to be 1 or 2)")
            return
        
        
        self._gripper_stat = [GripperStat()] * self._num_grippers
        self._gripper_cmd = [GripperCmd()]  * self._num_grippers

        self._run_test()

   
    def _update_gripper_stat(self, stat):
        self._gripper_stat[0] = stat
    def _update_right_gripper_stat(self, stat):
        self._gripper_stat[1] = stat
    '''
    def _run_test(self):
        rospy.logwarn('Running Gripper Test')
        test_state = 0
        r = rospy.Rate(1)
      
      
        ready = False
        while not (ready):
            ready = True
            for i in range(self._num_grippers):
                ready &= self._gripper_stat[i].is_ready 
                
        if (0 == test_state and (self.open or self.close)):
            rospy.logwarn('Opening/Closing Gripper')
            for i in range(self._num_grippers):
                if self.open:
                    self._gripper_cmd[i].position = 0.85
                else:
                    self._gripper_cmd[i].position = 0.0
                self._gripper_cmd[i].speed = 0.02
                self._gripper_cmd[i].force = 1.0
            test_state = 1
        elif (1 == test_state):
            success = True
            for i in range(self._num_grippers):
                if (self._gripper_stat[i].is_moving):
                    success = False
            if success:
                test_state = 2

        if (self._num_grippers == 1):
            self._gripper_pub.publish(self._gripper_cmd[0])
        elif (self._num_grippers == 2):
            self._left_gripper_pub.publish(self._gripper_cmd[0])
            self._right_gripper_pub.publish(self._gripper_cmd[1])

        r.sleep()
        print("robotiq yes")
    
    '''


    def _run_test(self):
        # rospy.logwarn('Running Gripper Test')
        test_state = 0
        r = rospy.Rate(1)
        ii=2
        while ii != 0:
            ii -=1
            ready = False
            while not (ready):
                ready = True
                for i in range(self._num_grippers):
                    ready &= self._gripper_stat[i].is_ready 
                
            if (0 == test_state and (self.open or self.close)):
                # rospy.logwarn('Opening/Closing Gripper')
                for i in range(self._num_grippers):
                    if self.open:
                        self._gripper_cmd[i].position = 0.85
                    else:
                        if self.zero:
                            self._gripper_cmd[i].position = 0.0
                        else:
                            self._gripper_cmd[i].position = 0.000001
                    self._gripper_cmd[i].speed = 1
                    self._gripper_cmd[i].force =30
                test_state = 1
            elif (1 == test_state):
                success = True
                for i in range(self._num_grippers):
                    if (self._gripper_stat[i].is_moving):
                        success = False
                if success:
                    test_state = 2

            if (self._num_grippers == 1):
                self._gripper_pub.publish(self._gripper_cmd[0])
            elif (self._num_grippers == 2):
                self._left_gripper_pub.publish(self._gripper_cmd[0])
                self._right_gripper_pub.publish(self._gripper_cmd[1])

            r.sleep()
     
