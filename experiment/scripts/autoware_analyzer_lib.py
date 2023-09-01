import csv
import yaml
# import rosbag
import math
import os
import signal
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp

class EXPERIMENT_STORAGE():
    def __init__(self, manager):
        self.is_collapsed_list = manager.dict()
        self.center_offset_list = manager.dict()
        self.is_matching_failed_list = manager.dict()
        self.E2E_response_time_list = manager.dict()
        # self.max_E2E_response_time_list = manager.list()
        self.miss_alignment_delay_list = manager.dict()
        self.velocity_list = manager.dict()
        self.max_miss_alignment_delay_list = manager.dict()
        self.avg_miss_alignment_delay_list = manager.dict()

    def sorting_dicts(self):
        self.is_collapsed_list = dict(sorted(self.is_collapsed_list.items()))
        self.is_matching_failed_list = dict(sorted(self.is_matching_failed_list.items()))
        self.center_offset_list = dict(sorted(self.center_offset_list.items()))

    def get_center_offset_for_experiment(self):
        max_center_offset = -1.0
        avg_center_offset = -1.0
        target_experiment_idx_list = get_idices_of_one_from_list(self.is_matching_failed_list.values(), reverse=True)

        target_center_offset = []
        # [value for key, value in self.center_offset_list.items() if key in target_experiment_idx_list]
        for idx in self.center_offset_list:
            if idx in target_experiment_idx_list:
                target_center_offset.extend(self.center_offset_list[idx])

        if len(target_center_offset) != 0:
            max_center_offset = max(target_center_offset)
            avg_center_offset = float(sum(target_center_offset)) / float(len(target_center_offset))

        return max_center_offset, avg_center_offset
    
    def get_miss_alignment_delay_for_experiment(self):
        max_miss_alignment_delay = -1.0
        avg_miss_alignment_delay = -1.0
        # target_experiment_idx_list = get_idices_of_one_from_list(self.is_matching_failed_list.values(), reverse=True)

        target_miss_alignment_delay = []
        # [value for key, value in self.center_offset_list.items() if key in target_experiment_idx_list]
        for idx in self.miss_alignment_delay_list:
            target_miss_alignment_delay.extend(self.miss_alignment_delay_list[idx])

        if len(target_miss_alignment_delay) != 0:
            max_miss_alignment_delay = float(max(target_miss_alignment_delay))
            avg_miss_alignment_delay = float(sum(target_miss_alignment_delay)) / float(len(target_miss_alignment_delay))

        return max_miss_alignment_delay, avg_miss_alignment_delay

class EXPERIMENT_INFO():
    def __init__(self, experiment_config):
        self.node_chain = experiment_config['node_chain']
        self.avoidance_x_range = experiment_config['avoidance_x_range']
        self.start_x = self.avoidance_x_range[0]
        self.end_x = self.avoidance_x_range[1]
        self.experiment_title = experiment_config['experiment_title']
        self.output_title = experiment_config['output_title']
        self.first_node = experiment_config['first_node']
        self.last_node = experiment_config['last_node']
        self.E2E_deadline = experiment_config['E2E_deadline']
        self.simulator = experiment_config['simulator']
        self.source_path = 'results/' + self.experiment_title
        self.iteration = self.get_number_of_files(self.source_path)
        # self.set_path_list()

    def get_number_of_files(self, path):
        output = str(os.popen('ls ' + path).read())
        output = output.split('\n')
        integer_cnt = 0
        for item in output:
            if item.isdigit():
                integer_cnt += 1
        # return len(output) - 1
        return integer_cnt
    
    # def set_path_list(self):
    #     experiment_info_path_list = []
    #     center_offset_path_list = []
    #     response_time_path_list = []
    #     for idx in range(self.iteration):
    #         experiment_info_path = self.source_path + '/' + str(idx) + '/experiment_info.yaml'
    #         experiment_info_path_list.append(experiment_info_path)
    #     for idx in range(self.iteration):
    #         center_offset_path = self.source_path + '/' + str(idx) + '/center_offset.csv'
    #         center_offset_path_list.append(center_offset_path)
    #     for idx in range(self.iteration):
    #         response_time_path = self.source_path + '/' + str(idx) + '/response_time'
    #         response_time_path_list.append(response_time_path)

    #     self.experiment_info_path_list = experiment_info_path_list
    #     self.center_offset_path_list = center_offset_path_list
    #     self.response_time_path_list = response_time_path_list

    def read_csv_to_dict(file_path):
        df = pd.read_csv(file_path)
        return df.to_dict(orient='list')

    def get_experiment_info(self, idx):
        experiment_info_path = self.source_path + '/' + str(idx) + '/experiment_info.yaml'
        experiment_info = {}
        with open(experiment_info_path, 'r') as f:
            experiment_info = yaml.safe_load(f)
        return experiment_info
    
    def get_driving_data(self, idx):
        driving_data_path = self.source_path + '/' + str(idx) + '/center_offset.csv'
        return pd.read_csv(driving_data_path, sep=',')
    
    def get_response_time(self, idx, node_name):
        response_time_path = self.source_path + '/' + str(idx) + '/response_time/' + node_name + '.csv'
        return pd.read_csv(response_time_path, sep=',')
    
    def get_response_time_path(self, idx, node_name):
        response_time_path = self.source_path + '/' + str(idx) + '/response_time/' + node_name + '.csv'
        return response_time_path

    def get_instance_pair(self, idx):
        output_start_instance = -1.0
        output_end_instance = -1.0
        
        lower_bound_x = min(self.start_x, self.end_x)
        upper_bound_x = max(self.start_x, self.end_x)
        x = None

        if self.simulator == 'old':
            x = 'x'
        elif self.simulator == 'carla' or self.simulator == 'svl':
            x = 'gnss_pose_x'
        else:
            print('# Wrong simulator!')
            exit()
        
        df = self.get_driving_data(idx)
        filtered_df = df[(df[x] >= lower_bound_x) & (df[x] <= upper_bound_x)]

        if len(filtered_df) != 0:
            output_start_instance = min(filtered_df['instance'])
            output_end_instance = max(filtered_df['instance'])

        return output_start_instance, output_end_instance
    
    def get_E2E_response_time(self, idx, type, first_node_name=None, last_node_name=None):
        if type != 'shortest' and type != 'longest':
            print('[ERROR] Invalidate type:', type)
            exit()

        start_instance, end_instance = self.get_instance_pair(idx)
        E2E_response_time = {}
        avg_E2E_response_time = -1.0
        max_E2E_response_time = -1.0

        if first_node_name == None:
            first_node_df = self.get_response_time(idx, self.first_node)
        else:
            first_node_df = self.get_response_time(idx, first_node_name)

        if last_node_name == None:
            last_node_df = self.get_response_time(idx, self.last_node)
        else:
            last_node_df = self.get_response_time(idx, last_node_name)

        last_node_df = last_node_df[(last_node_df['instance'] >= start_instance) & (last_node_df['instance'] <= end_instance)]
        first_node_df = first_node_df[first_node_df['instance'] >= start_instance & first_node_df['instance'].isin(last_node_df['instance'])]

        if type == 'shortest':
            first_node_df = first_node_df.drop_duplicates(subset='instance', keep='first')
            last_node_df = last_node_df.drop_duplicates(subset='instance', keep='first')
        else:
            first_node_df = first_node_df.drop_duplicates(subset='instance', keep='last')
            last_node_df = last_node_df.drop_duplicates(subset='instance', keep='last')
        # print(last_node_df)

        for instance_id in last_node_df['instance']:
            start_time = first_node_df.loc[first_node_df['instance'] == instance_id, 'start'].to_numpy()[0]
            end_time = last_node_df.loc[last_node_df['instance'] == instance_id, 'end'].to_numpy()[0]
            response_time = end_time - start_time
            E2E_response_time[instance_id] = response_time * 1000 # unit: ms

        E2E_response_time_df = pd.DataFrame(
            {
                'instance': list(E2E_response_time.keys()),
                'E2E_response_time': list(E2E_response_time.values())
            }
        )

        avg_E2E_response_time = E2E_response_time_df['E2E_response_time'].mean()
        max_E2E_response_time = max(E2E_response_time_df['E2E_response_time'])

        return E2E_response_time, max_E2E_response_time, avg_E2E_response_time
    
    def get_center_offset(self, idx):
        driving_data_df = self.get_driving_data(idx)
        driving_data_df['center_offset'] = driving_data_df['center_offset'].abs()

        max_center_offset = max(driving_data_df['center_offset'])
        avg_center_offset = driving_data_df['center_offset'].mean()

        center_offset = driving_data_df.set_index('instance')['center_offset'].to_dict()

        return center_offset, max_center_offset, avg_center_offset
    
    def get_waypoints(self, idx):
        driving_data_df = self.get_driving_data(idx)
        driving_data = driving_data_df.to_dict(orient='list')
        pose_x = None
        pose_y = None
        if self.simulator == 'old':
            pose_x = driving_data['x']
            pose_y = driving_data['y']
        elif self.simulator == 'carla' or self.simulator == 'svl':
            pose_x = driving_data['gnss_pose_x']
            pose_y = driving_data['gnss_pose_y']
        else:
            print('# Wrong simulator')
            exit()
        
        # waypoints = np.stack((pose_x, pose_y), axis=1)
        return pose_x, pose_y
    
    def get_center_line(self, idx):
        center_line_path = self.source_path + '/' + str(idx) + '/center_line.csv'
        center_line = pd.read_csv(center_line_path, sep=',')
        return center_line['center_x'], center_line['center_y']
        # return np.stack((center_line['center_x'], center_line['center_y']), axis=1)
    
    def check_matching_is_failed(self, idx):
        driving_data_df = self.get_driving_data(idx)
        if self.simulator == 'old':
            ndt_score_threshold = 1.5
            if len(driving_data_df[driving_data_df['ndt_score'] > ndt_score_threshold]) > 0:
                return True
        
        elif self.simulator == 'carla' or self.simulator == 'svl':
            fail_ratio_threshold = 0.03
            absolute_fail_threshold = 5
            fail_threshold = 1.48
            start_instance, end_instance = self.get_instance_pair(idx)
            driving_data_df = driving_data_df[(driving_data_df['instance'] >= start_instance) & (driving_data_df['instance'] <= end_instance)]
            fail_cnt = len((driving_data_df[np.sqrt(np.power(driving_data_df['gnss_pose_x'] - driving_data_df['current_pose_x'], 2) \
                                              + np.power(driving_data_df['gnss_pose_y'] - driving_data_df['current_pose_y'], 2)) > fail_threshold]))
            absolute_fail_cnt = len((driving_data_df[np.sqrt(np.power(driving_data_df['gnss_pose_x'] - driving_data_df['current_pose_x'], 2) \
                                              + np.power(driving_data_df['gnss_pose_y'] - driving_data_df['current_pose_y'], 2)) > absolute_fail_threshold]))
            if absolute_fail_cnt > 0:
                return True
            if float(fail_cnt) / float(len(driving_data_df)) > fail_ratio_threshold:
                return True
        else:
            print('# Wrong simulator!')
            exit()

        return False
    
    def get_instance_leakage(self):
        # # 조건을 만족하지 않는 행들을 검출
        # invalid_rows = df[(df['column1'] != df['column1'].min()) & (df['column1'] != df['column1'].shift(-1) - 1)]

        # # 같은 값을 가지는 행은 조건을 만족하도록 수정
        # same_value_rows = df[df['column1'].duplicated(keep=False)]

        # # 조건을 만족하지 않는 행들 중에서 같은 값을 가지는 행을 제외
        # invalid_rows = invalid_rows[~invalid_rows.index.isin(same_value_rows.index)]
        pass

    def get_velocity_info(self, idx):
        driving_data_df = self.get_driving_data(idx)
        if 'current_velocity' not in driving_data_df:
            return {}, -1
        driving_data_df['current_velocity'] = driving_data_df['current_velocity'].abs()
        avg_velocity = driving_data_df['current_velocity'].mean()
        velocity_info = driving_data_df.set_index('instance')['current_velocity'].to_dict()
        return velocity_info, avg_velocity
    
    def get_lane(self, idx, offset):
        x, y = self.get_center_line(idx)
        x_shifted = []
        y_shifted = []
        for i in range(len(x)):
            if i == 0:
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
            elif i == len(x) - 1:
                dx = x[i] - x[i-1]
                dy = y[i] - y[i-1]
            else:
                dx = (x[i+1] - x[i-1]) / 2
                dy = (y[i+1] - y[i-1]) / 2
            
            # 수직 방향으로 이동할 좌표 계산
            magnitude = math.sqrt(dx**2 + dy**2)
            perp_dx = offset * dy / magnitude
            perp_dy = - offset * dx / magnitude
            
            # 좌표 이동
            shifted_x = x[i] + perp_dx
            shifted_y = y[i] + perp_dy
            x_shifted.append(shifted_x)
            y_shifted.append(shifted_y)
        return x_shifted, y_shifted
    
    def get_perf_info(self):
        # n = 0
        # for path in os.listdir(self.source_path):
        #     if not os.path.isfile(os.path.join(self.source_path, path)): n = n + 1

        avg_memory_bandwidth_list = [] # GB/s
        l3d_cache_refill_event_cnt_of_ADAS_cores_list = []
        l3d_cache_refill_event_cnt_of_all_cores_list = []
        # n = n - 1
        for idx in range(self.iteration):
            experiment_info_path = self.source_path + '/' + str(idx) + '/experiment_info.yaml'
            with open(experiment_info_path) as f:
                experiment_info = yaml.load(f, Loader=yaml.FullLoader)
                if 'l3d_cache_refill_event_cnt_of_ADAS_cores(per sec)' not in experiment_info \
                    or 'avg_total_memory_bandwidth_usage(GB/s)' not in experiment_info \
                    or 'l3d_cache_refill_event_cnt_of_all_cores(per sec)' not in experiment_info:
                    return {}
                l3d_cache_refill_event_cnt_of_ADAS_cores_list.append(float(experiment_info['l3d_cache_refill_event_cnt_of_ADAS_cores(per sec)']))
                l3d_cache_refill_event_cnt_of_all_cores_list.append(float(experiment_info['l3d_cache_refill_event_cnt_of_all_cores(per sec)']))
                avg_memory_bandwidth_list.append(float(experiment_info['avg_total_memory_bandwidth_usage(GB/s)']))
        
        avg_l3d_cache_refill_event_cnt_of_ADAS_cores = sum(l3d_cache_refill_event_cnt_of_ADAS_cores_list)/len(l3d_cache_refill_event_cnt_of_ADAS_cores_list)
        avg_l3d_cache_refill_event_cnt_of_all_cores = sum(l3d_cache_refill_event_cnt_of_all_cores_list)/len(l3d_cache_refill_event_cnt_of_all_cores_list)
        avg_memory_bandwidth = sum(avg_memory_bandwidth_list)/len(avg_memory_bandwidth_list)
        
        perf_info = {}
        perf_info['avg_l3d_cache_refill_event_cnt_of_ADAS_cores(per sec)'] = avg_l3d_cache_refill_event_cnt_of_ADAS_cores
        perf_info['avg_l3d_cache_refill_event_cnt_of_all_cores(per sec)'] = avg_l3d_cache_refill_event_cnt_of_all_cores
        perf_info['avg_total_memory_bandwidth_usage'] = avg_memory_bandwidth

        return perf_info
    
    def origin_get_E2E_response_time(self, idx, type, first_node_name=None, last_node_name=None):
        if type != 'shortest' and type != 'longest':
            print('[ERROR] Invalidate type:', type)
            exit()

        E2E_start_instance, E2E_end_instance = self.get_instance_pair(idx)
        instance_info = {}
        start_instance = -1
        E2E_response_time = {}
        column_idx = {}

        if first_node_name == None:
            first_node_path = self.source_path + '/' + str(idx) + '/' + self.first_node + '.csv'
        else:
            first_node_path = self.source_path + '/' + str(idx) + '/' + first_node_name + '.csv'

        if last_node_name == None:
            last_node_path = self.source_path + '/' + str(idx) + '/' + self.last_node + '.csv'
        else:
            last_node_path = self.source_path + '/' + str(idx) + '/' + last_node_name + '.csv'

        # E2E Response Time
        with open(last_node_path) as f_last:
            reader = csv.reader(f_last)        
            for i, row in enumerate(reader):            
                if i == 0: 
                    column_idx = get_column_idx_from_csv(row)
                    continue
                end_time = float(row[column_idx['end']])
                instance_id = int(row[column_idx['instance']])
                if type == 'shortest':
                    if instance_id in instance_info: continue
                if i == 1: start_instance = instance_id         
                instance_info[instance_id] = {'start_time': -1.0, 'end_time': end_time}

        with open (first_node_path) as f_start:        
            reader = csv.reader(f_start)
            for i, row in enumerate(reader):
                if i == 0: 
                    column_idx = get_column_idx_from_csv(row)
                    continue       

                start_time = float(row[column_idx['start']])
                # start_time = float(row[column_idx['topic_pub_time']])
                instance_id = int(row[column_idx['instance']])
                if instance_id < start_instance: continue
                if instance_id not in instance_info: continue
                if type == 'shortest':
                    if instance_info[instance_id]['start_time'] > 0: continue
                instance_info[instance_id]['start_time'] = start_time
        for instance_id in instance_info:
            response_time = instance_info[instance_id]['end_time'] - instance_info[instance_id]['start_time']        
            E2E_response_time[instance_id] = float(response_time * 1000) # unit: ms

        keys = list(E2E_response_time.keys())

        does_start_instance_found = False
        for key in keys:
            if key >= E2E_start_instance and does_start_instance_found == False:
                E2E_start_instance = key
                does_start_instance_found = True
                continue                       
            if key >= E2E_end_instance and E2E_end_instance > 0.0: 
                E2E_end_instance = key
                break        
        remove_target = []
        for k in E2E_response_time:
            if k < E2E_start_instance or k > E2E_end_instance or k not in keys: remove_target.append(k)

        for k in remove_target: E2E_response_time.pop(k, None)
        if len(E2E_response_time) == 0:
            avg_E2E_response_time = 0.0    
            max_E2E_response_time = 0.0
        else:
            avg_E2E_response_time = get_dict_avg(E2E_response_time)
            max_E2E_response_time = get_dict_max(E2E_response_time)
            # min_E2E_response_time = get_dict_min(E2E_response_time)
            # q1_E2E_response_time = np.quantile(E2E_response_time.values(), .25)
            # q3_E2E_response_time = np.quantile(E2E_response_time.values(), .75)

        return E2E_response_time, max_E2E_response_time, avg_E2E_response_time

def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    return

def convert_boolean_list_to_int_list(list_of_booleans):    
    return [int(item) for item in list_of_booleans]

def get_idices_of_one_from_list(input, reverse=False):
    output = []
    for i, v in enumerate(input):
        if reverse: v = not v # 2**v%2
        if v: output.append(i)
    return output

def merge_binary_list_to_idx_list(a, b):
    output = []
    for i,_ in enumerate(a):
        if a[i] == 1 or b[i] == 1: output.append(i)
    return output

def stop_rosbag_record():
    _output = str(os.popen('ps au | grep rosbag').read())
    _output = _output.split('\n')
    for line in _output:    
        if not '/opt/ros/melodic/bin/rosbag' in line: continue
        pid = -1
        for v in line.split(' '):
            try: pid = int(v)
            except: continue        
            break

        if pid != -1: os.kill(pid, signal.SIGINT)

def save_dict(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    return

def subsctract_dicts(data1, data2):
    output = {}
    remove_targets = []
    keys = data1.keys()
    for k in keys:
        if k not in data2: 
            remove_targets.append(k)
            continue
        output[k] = data1[k] - data2[k]
        
    for remove_target in remove_targets:
        if remove_target in output:
            output.pop(remove_target)

    return output

def get_dict_avg(data):
    sum = 0.0
    for key in data:
        sum = sum + float(data[key])
    output = 0
    if len(data) != 0: output = sum / float(len(data))
    return output

def get_dict_max(data):
    max = 0.0
    for key in data:
        v = float(data[key])
        if v > max: max = v
    return max

def get_column_idx_from_csv(line):
    output = {}
    for i, v in enumerate(line):
        output[v] = i
    return output
