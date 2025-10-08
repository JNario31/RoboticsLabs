import os
import sys
import rclpy
import cv2
import datetime
import numpy as np
import pandas as pd
import math
import threading

from rclpy.node import Node
from ultralytics import YOLO
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from ultralytics.engine.results import Results, Keypoints
from ament_index_python.packages import get_package_share_directory

from kinova_gen3_interfaces.srv import Status, SetGripper, GetGripper, SetJoints, GetJoints, GetTool, SetTool

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

from cpmr_ch12.utilities import parseConnectionArguments, DeviceConnection


# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check


def example_angular_action_movement(base, angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):

    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    # Place arm straight up
    print(actuator_count.count)
    if actuator_count.count != len(angles):
        print(f"bad lengths {actuator_count.count} {len(angles)}")
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished


def get_angular_state(base_cyclic):
    feedback = base_cyclic.RefreshFeedback()
    actuators = feedback.actuators
    v = []
    for j in actuators:
        v.append(j.position)
    return v

def example_cartesian_action_movement(base, x, y, z, theta_x, theta_y, theta_z):

    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = x
    cartesian_pose.y = y
    cartesian_pose.z = z
    cartesian_pose.theta_x = theta_x
    cartesian_pose.theta_y = theta_y
    cartesian_pose.theta_z = theta_z

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    return finished

def get_tool_state(base_cyclic):
    feedback = base_cyclic.RefreshFeedback()
    base = feedback.base

    return  base.tool_pose_x, base.tool_pose_y, base.tool_pose_z, base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z



class YOLO_Pose(Node):
    _BODY_PARTS = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                   "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
                   "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
    def __init__(self):
        super().__init__('pose_node')

        # params
        self._model_file = os.path.join(get_package_share_directory('cpmr_ch12'), 'yolov8n-pose.pt') 
        self.declare_parameter("model", self._model_file) 
        model = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("device", "cpu")
        self._device = self.get_parameter("device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.5)
        self._threshold = self.get_parameter("threshold").get_parameter_value().double_value


        self.declare_parameter("camera_topic", "/mycamera/image_raw")
        self._camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value

        
        self._move_flag = False
        self._bridge = CvBridge()
        self._model = YOLO(model)
        self._model.fuse()

        # subs
        self._sub = self.create_subscription(Image, self._camera_topic, self._camera_callback, 1) 

        # Create the Kinova Gen3 interface object
        self.create_service(Status, "home", self._handle_home)
        self.create_service(GetGripper, "get_gripper", self._handle_get_gripper)
        self.create_service(SetGripper, "set_gripper", self._handle_set_gripper)
        self.create_service(SetJoints, "set_joints", self._handle_set_joints)
        self.create_service(GetJoints, "get_joints", self._handle_get_joints)
        self.create_service(SetTool, "set_tool", self._handle_set_tool)
        self.create_service(GetTool, "get_tool", self._handle_get_tool)

        args = parseConnectionArguments()
        with DeviceConnection.createTcpConnection(args) as router:
            self._router = router
            self._base = BaseClient(self._router)
            self._base_cyclic = BaseCyclicClient(self._router)

        if example_move_to_home_position(self._base):
           self.get_logger().info('Robot initialized successfully')
        else:
           self.get_logger().error('Failed to initialize robot position')
        
    def parse_keypoints(self, results: Results):

        keypoints_list = []

        for points in results.keypoints:        
            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if conf >= self._threshold:
                    keypoints_list.append([kp_id, p[0], p[1], conf])

        return keypoints_list
    
    def _camera_callback(self, data):
        self.get_logger().info(f'{self.get_name()} camera callback')
        img = self._bridge.imgmsg_to_cv2(data)
        results = self._model.predict(
                source = img,
                verbose = False,
                stream = False,
                conf = self._threshold,
                device = self._device
        )

        if len(results) != 1:
            self.get_logger().info(f'{self.get_name()}  Nothing to see here or too much {len(results)}')
            return
            
        results = results[0].cpu()
        if len(results.boxes.data) == 0:
            self.get_logger().info(f'{self.get_name()}  boxes are too small')
            return

        if results.keypoints:
            keypoints = self.parse_keypoints(results)
            left_shoulder = None
            right_shoulder = None
            if len(keypoints) > 0:
                for i in range(len(keypoints)):
                    self.get_logger().info(f'{self.get_name()}  {YOLO_Pose._BODY_PARTS[keypoints[i][0]]} {keypoints[i]}')

                # Visualize results on frame        
                annotated_frame = results[0].plot()
                cv2.imshow('Results', annotated_frame)
                cv2.waitKey(1)

    # Kinova functions
    def _handle_home(self, request, response):
        """Move to home"""
        self.get_logger().info(f'{self.get_name()} moving to home')

        response.status = example_move_to_home_position(self._base)
        return response

    def _handle_set_joints(self, request, response):
        """Set joint values"""
        self.get_logger().info(f'{self.get_name()} Setting joint values')
        if len(request.joints) != 6:
            self.get_logger().info(f'{self.get_name()} Must specify exactly six joint angles')
            response.status = False
            return response

        response.status = example_angular_action_movement(self._base, angles=request.joints)
        return response
    
    def _handle_get_joints(self, request, response):
        """Get joint values"""
        self.get_logger().info(f'{self.get_name()} Getting joint values')
        response.joints = get_angular_state(self._base_cyclic)
        return response


    def _handle_set_tool(self, request, response):
        """Set tool values"""
        self.get_logger().info(f'{self.get_name()} Setting tool values')

        response.status = example_cartesian_action_movement(self._base, request.x, request.y, request.z, request.theta_x, request.theta_y, request.theta_z)
        return response

    def _handle_get_tool(self, request, response):
        """Get tool values"""
        self.get_logger().info(f'{self.get_name()} Getting tool values')
        x, y, z, theta_x, theta_y, theta_z = get_tool_state(self._base_cyclic)
        response.x = x
        response.y = y
        response.z = z
        response.theta_x = theta_x
        response.theta_y = theta_y
        response.theta_z = theta_z
        return response


def main(args=None):
    rclpy.init(args=args)
    node = YOLO_Pose()
    try:
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()