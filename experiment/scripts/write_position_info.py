import math
from autoware_msgs.msg import LaneArray, NDTStat, VehicleCmd
from geometry_msgs.msg import PoseStamped
import geometry_msgs
from rubis_msgs.msg import TwistStamped
from visualization_msgs.msg import MarkerArray
import rospy
import os
import csv
import time

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def dis(wp1, wp2):
    return math.sqrt((wp1[0] - wp2[0]) * (wp1[0] - wp2[0]) + (wp1[1] - wp2[1]) * (wp1[1] - wp2[1]))

def find_closest_point(map_wp_list, wp, yaw_deg):
    min_distance = 500
    min_wp = [0, 0]

    for map_wp in map_wp_list:
        if dis(map_wp, wp) < min_distance:
            min_distance = dis(map_wp, wp)
            min_wp = map_wp

    if 45 <= yaw_deg and yaw_deg < 135:
        if wp[0] < min_wp[0]:
            min_distance *= -1
    elif 135 <= yaw_deg and yaw_deg < 225:
        if wp[1] < min_wp[1]:
            min_distance *= -1
    elif 225 <= yaw_deg and yaw_deg < 315:
        if wp[0] > min_wp[0]:
            min_distance *= -1
    elif wp[1] > min_wp[1]:
        min_distance *= 1
    
    return min_wp, min_distance

def write_position_info():
    center_offset_file = 'center_offset.csv'
    if center_offset_file.split('.')[-1] != 'csv':
        print('Output file should be csv file!')
        exit(1)

    rospy.init_node('driving_progress_logger')

    map_wp_list = []

    gnss_pose_x = 0
    gnss_pose_y = 0
    current_pose_x = 0
    current_pose_y = 0

    lane_msg = rospy.wait_for_message('/lane_waypoints_array', LaneArray, timeout=None)

    for wp in lane_msg.lanes[0].waypoints:
        map_wp_list.append([wp.pose.pose.position.x, wp.pose.pose.position.y])

    center_line_file = 'center_line.csv'
    with open(center_line_file, "w") as f:
        center_line_wr = csv.writer(f)
        center_line_wr.writerow(['center_x', 'center_y'])
        for p in map_wp_list:
            center_line_wr.writerow([float(p[0]), float(p[1])])
    f.close()

    # Wait until car starts
    rospy.wait_for_message('/vehicle_cmd', VehicleCmd, timeout=None)
    with open(center_offset_file, "w") as f:
        center_offset_wr = csv.writer(f)
        center_offset_wr.writerow(['ts', 'state', 'center_offset', 'ndt_response_time(ms)', 'instance', 'gnss_pose_x', 'gnss_pose_y', 'ndt_score', 'current_pose_x', 'current_pose_y', 'current_velocity'])
        prev_dis = 0
        while not rospy.is_shutdown():
            gnss_pose_msg = rospy.wait_for_message('/gnss_pose', PoseStamped, timeout=None)
            current_pose_msg = rospy.wait_for_message('/current_pose', PoseStamped, timeout=None)
            ndt_stat_msg = rospy.wait_for_message('/ndt_stat', NDTStat, timeout=None)
            twist_msg = rospy.wait_for_message('/rubis_twist_cmd', TwistStamped, timeout=None)
            current_velocity_msg = rospy.wait_for_message('/current_velocity', geometry_msgs.msg.TwistStamped, time=None)

            instance=twist_msg.instance

            gnss_pose_x = round(gnss_pose_msg.pose.position.x, 3)
            gnss_pose_y = round(gnss_pose_msg.pose.position.y, 3)
            gnss_ori_x = gnss_pose_msg.pose.orientation.x
            gnss_ori_y = gnss_pose_msg.pose.orientation.y
            gnss_ori_z = gnss_pose_msg.pose.orientation.z
            gnss_ori_w = gnss_pose_msg.pose.orientation.w                                                         
            gnss_r, gnss_p, gnss_y = euler_from_quaternion(gnss_ori_x, gnss_ori_y, gnss_ori_z, gnss_ori_w)

            yaw_deg = (gnss_y * 180 / math.pi + 1800) % 360
            
            min_wp, min_dis = find_closest_point(map_wp_list, [gnss_pose_x, gnss_pose_y], yaw_deg)

            if abs(prev_dis) > 1.5 and min_dis * prev_dis < 0:
                min_dis *= 1
            prev_dis = min_dis

            state_text = 'None'

            current_pose_x = round(current_pose_msg.pose.position.x, 3)
            current_pose_y = round(current_pose_msg.pose.position.y, 3)
            ndt_ori_x = current_pose_msg.pose.orientation.x
            ndt_ori_y = current_pose_msg.pose.orientation.y
            ndt_ori_z = current_pose_msg.pose.orientation.z
            ndt_ori_w = current_pose_msg.pose.orientation.w                 
            current_velocity = current_velocity_msg.twist.linear.x

            center_offset_wr.writerow([time.clock_gettime(time.CLOCK_MONOTONIC), state_text, str(min_dis), str(ndt_stat_msg.exe_time), instance, gnss_pose_x, gnss_pose_y, str(ndt_stat_msg.score), current_pose_x, current_pose_y, current_velocity])
    f.close()

if __name__ == '__main__':
    write_position_info()