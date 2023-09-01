import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
# import scripts.autoware_analyzer_lib as aa
import scripts.autoware_analyzer_lib as aa
import numpy as np
import copy
from tqdm import tqdm
import multiprocessing as mp
import concurrent.futures
import copy
import sys

def profile_center_offset(experiment_info, idx, exp_storage):
    is_collapsed = exp_storage.is_collapsed_list[idx]

    center_offset, max_center_offset, avg_center_offset = experiment_info.get_center_offset(idx)
    output_path = 'analyzation/' + experiment_info.output_title + '/center_offset'
    if not os.path.exists(output_path): os.system('mkdir -p ' + output_path)

    # Plot graph
    x_data = list(center_offset.keys()) # Instance IDs
    y_data = list(center_offset.values()) # Center offset(m)
    exp_storage.center_offset_list[idx] = center_offset.values()
    plot_path = output_path + '/' + experiment_info.experiment_title + '_' + str(idx) + '_center_offset.png'

    plt.plot(x_data, y_data)
    plt.axhline(y = max_center_offset, color = 'r', linestyle = ':', label='Max')
    plt.axhline(y = avg_center_offset, color = 'b', linestyle = ':', label='Avg')
    plt.legend()    
    plt.xlabel('Instance ID')
    plt.ylabel('Center offset(m)')
    plt.title('is_collapsed='+str(is_collapsed))
    plt.savefig(plot_path)
    plt.close()


def profile_response_time(experiment_info, idx, exp_storage, filter=1.0):
    _profile_response_time(experiment_info, idx, exp_storage, filter, type='shortest')
    # _profile_response_time(experiment_info, idx, exp_storage, filter, type='longest')

def _profile_response_time(experiment_info, idx, exp_storage, filter, type):
    if type == 'shortest': label = 'Shortest'
    elif type == 'longest': label = 'Longest'
    else: 
        print('[ERROR] Invalidate type:', type)
        exit()

    if 'is_collapsed' not in experiment_info.get_experiment_info(idx):
        is_collapsed = experiment_info.get_experiment_info(idx)['is_collaped']
    else:
        is_collapsed = experiment_info.get_experiment_info(idx)['is_collapsed']

    is_matching_failed = experiment_info.check_matching_is_failed(idx)

    E2E_response_time, max_E2E_response_time, avg_E2E_response_time = experiment_info.get_E2E_response_time(idx, type)
    exp_storage.E2E_response_time_list[idx] = E2E_response_time
    # exp_storage.all_E2E_response_time_list.append(E2E_response_time.values())
    # exp_storage.max_E2E_response_time_list.append(max_E2E_response_time)
    
    output_path = 'analyzation/' + experiment_info.experiment_title + '/' + type + '_E2E_response_time'
    if not os.path.exists(output_path): os.system('mkdir -p ' + output_path)

    # Plot graph
    x_data = list(E2E_response_time.keys()) # Instance IDs
    y_data = list(E2E_response_time.values()) # E2E response time(ms)
    if not is_collapsed:
        x_data = x_data[:int(len(x_data) * filter)]
        y_data = y_data[:int(len(y_data) * filter)]

    plot_path = output_path + '/' + experiment_info.experiment_title + '_' + str(idx) + '_' + type + '_E2E_plot.png'

    plt.plot(x_data, y_data)
    plt.axhline(y = max_E2E_response_time, color = 'r', linestyle = ':', label='Max')
    plt.axhline(y = avg_E2E_response_time, color = 'b', linestyle = ':', label='Avg')    
    plt.legend()
    plt.ylim(0, 1000)
    plt.xlabel('Instance ID')
    plt.ylabel(label + ' E2E Response Time (ms)')
    plt.yticks(np.arange(0,1000,100))
    plt.title('E2E during avoidance\nis_collapsed='+str(is_collapsed) + '/ is_matching_failed='+str(is_matching_failed))
    plt.savefig(plot_path)
    plt.close()

def profile_miss_alignment_delay(experiment_info, idx, exp_storage, filter=1.0):

    E2E_response_time, _, _ = experiment_info.get_E2E_response_time(idx, 'shortest')
    is_collapsed = exp_storage.is_collapsed_list[idx]
    is_matching_failed = exp_storage.is_matching_failed_list[idx]

    node_response_time_list = []
    for node in experiment_info.node_chain:
        node_response_time, _, _ = experiment_info.get_E2E_response_time(idx, 'shortest', first_node_name=node, last_node_name=node)
        node_response_time_list.append(node_response_time)
    miss_alignment_delay = copy.deepcopy(E2E_response_time)

    for node_response_time in node_response_time_list:
        miss_alignment_delay = aa.subsctract_dicts(miss_alignment_delay, node_response_time)    
    
    # Plot graph
    output_dir_path = 'analyzation/' + experiment_info.output_title + '/' + 'miss_alignment_delay'
    if not os.path.exists(output_dir_path): os.system('mkdir -p ' + output_dir_path)

    x_miss_alignment_delay_data = list(miss_alignment_delay.keys()) # Instance IDs
    y_miss_alignment_delay_data = list(miss_alignment_delay.values()) # E2E response time(ms)
    if not is_collapsed:
        x_miss_alignment_delay_data = x_miss_alignment_delay_data[:int(len(x_miss_alignment_delay_data) * filter)]
        y_miss_alignment_delay_data = y_miss_alignment_delay_data[:int(len(y_miss_alignment_delay_data) * filter)]
    plt.plot(x_miss_alignment_delay_data, y_miss_alignment_delay_data, color = 'g', label = 'Miss alignment delay')
    max_miss_alignment_delay = max(y_miss_alignment_delay_data)
    avg_miss_alignment_delay = sum(y_miss_alignment_delay_data)/len(y_miss_alignment_delay_data)
    # exp_storage.max_miss_alignment_delay_list[idx] = max_miss_alignment_delay
    # exp_storage.avg_miss_alignment_delay_list[idx] = avg_miss_alignment_delay
    exp_storage.miss_alignment_delay_list[idx] = y_miss_alignment_delay_data

    x_E2E_data = list(E2E_response_time.keys()) # Instance IDs
    y_E2E_data = list(E2E_response_time.values()) # E2E response time(ms)
    color = 'r'
    if not is_collapsed:
        x_E2E_data = x_E2E_data[:int(len(x_E2E_data) * filter)]
        y_E2E_data = y_E2E_data[:int(len(y_E2E_data) * filter)]
        color = 'b'
    
    plt.plot(x_E2E_data, y_E2E_data, color = color, label = 'E2E')

    plot_path = output_dir_path + '/' + experiment_info.experiment_title + '_' + str(idx) + '_' + 'miss_alignment_delay_plot.png'
           
    plt.legend()
    plt.ylim(0, 1000)
    plt.xlabel('Instance ID')
    plt.ylabel('Time (ms)')
    plt.yticks(np.arange(0,1000,100))
    plt.title('is_collapsed='+str(is_collapsed) + '/ is_matching_failed='+str(is_matching_failed))
    plt.savefig(plot_path)
    plt.close()

    return max_miss_alignment_delay, avg_miss_alignment_delay

def profile_waypoints(experiment_info, idx, exp_storage):
    output_path = 'analyzation/' + experiment_info.output_title + '/trajectories'
    if not os.path.exists(output_path): os.system('mkdir -p ' + output_path)

    right_lane_offset = 3.1
    left_lane_offset = -3.2
    left_left_lane_offset = -9.8
    # right Lane
    right_lane_x, right_lane_y = experiment_info.get_lane(idx, right_lane_offset)
    left_lane_x, left_lane_y = experiment_info.get_lane(idx, left_lane_offset)
    left_left_lane_x, left_left_lane_y = experiment_info.get_lane(idx, left_left_lane_offset)

    plt.plot(right_lane_x, right_lane_y, 'k', label='Lane')
    plt.plot(left_lane_x, left_lane_y, 'k', linestyle='--', label='Left Lane')
    plt.plot(left_left_lane_x, left_left_lane_y, 'k', label='Left Left Lane')

    waypoints_x, waypoints_y = experiment_info.get_waypoints(idx)
    
    color = 'b'
    if exp_storage.is_collapsed_list[idx]: color = 'r'
    
    plt.plot(waypoints_x, waypoints_y, color, linewidth=1.0)

    if experiment_info.simulator == 'old':
        # Objects
        npc1_x = [6, 6, -1, -1, 6]
        npc1_y = [51, 48, 48, 51, 51]
        npc2_x = [6, 6, -1, -1, 6]
        npc2_y = [55, 52, 52, 55, 55]
        plt.plot(npc1_x, npc1_y, 'k')
        plt.plot(npc2_x, npc2_y, 'k')
    elif experiment_info.simulator == 'carla' or experiment_info.simulator == 'svl':
        pass

    # Plot
    plot_path = output_path + '/' + experiment_info.experiment_title + '_' + str(idx) + '_waypoints.png'
            
    plt.xlim([-70, 70])
    plt.ylim([-70,70])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('is_collapsed='+str(exp_storage.is_collapsed_list[idx]) + '/ is_matching_failed='+str(exp_storage.is_matching_failed_list[idx]))
    plt.legend(loc='center')
    plt.savefig(plot_path)

    plt.close()


def profile_analyzation_info(experiment_info, exp_storage):
    analyzation_info = {}
    is_collapsed_list = exp_storage.is_collapsed_list.values()
    is_matching_failed_list = exp_storage.is_matching_failed_list.values()

    collision_index_list = aa.get_idices_of_one_from_list(is_collapsed_list)
    matching_failure_index_list = aa.get_idices_of_one_from_list(is_matching_failed_list)

    perf_info = experiment_info.get_perf_info()
    max_center_offset, avg_center_offset = exp_storage.get_center_offset_for_experiment()
    max_miss_alignment_delay, avg_miss_alignment_delay = exp_storage.get_miss_alignment_delay_for_experiment()

    if len(is_collapsed_list) == 0: collision_ratio = 0
    else: collision_ratio = sum(is_collapsed_list)/len(is_collapsed_list)

    if len(is_matching_failed_list) == 0: matching_failure_ratio = 0
    else: matching_failure_ratio = sum(is_matching_failed_list)/len(is_matching_failed_list)

    analyzation_info['result'] = {}
    analyzation_info['resource_usage'] = {}

    analyzation_info['result']['avg_center_offset'] = avg_center_offset
    analyzation_info['result']['max_center_offset'] = max_center_offset
    # analyzation_info['result']['avg_velocity'] = avg_velocity
    # print(std_dev_velocity)
    # print(type(std_dev_velocity))
    # analyzation_info['result']['std_var_velocity'] = str(std_dev_velocity)
    analyzation_info['result']['collision_index'] = collision_index_list
    analyzation_info['result']['collision_ratio'] = collision_ratio
    analyzation_info['result']['matching_failure_index'] = matching_failure_index_list
    analyzation_info['result']['matching_failure_ratio'] = matching_failure_ratio
    analyzation_info['result']['max_miss_alignment_delay'] = max_miss_alignment_delay
    analyzation_info['result']['avg_miss_alignment_delay'] = avg_miss_alignment_delay

    for key in list(perf_info.keys()):
        analyzation_info['resource_usage'][key] = perf_info[key]

    analyzation_info_path = 'analyzation/' + experiment_info.output_title + '/analyzation_info.yaml'
    with open(analyzation_info_path, 'w') as f: yaml.dump(analyzation_info, f, default_flow_style=False)

    return

def profile_response_time_for_experiment(experiment_info, exp_storage, filter=1.0):

    _profile_response_time_for_experiment(experiment_info, exp_storage, type='shortest', mode='all', filter=filter)
    _profile_response_time_for_experiment(experiment_info, exp_storage, type='shortest', mode='normal', filter=filter)
    _profile_response_time_for_experiment(experiment_info, exp_storage, type='shortest', mode='collision', filter=filter)
    _profile_response_time_for_experiment(experiment_info, exp_storage, type='shortest', mode='matching_failed', filter=filter)

    # _profile_response_time_for_experiment(experiment_info, exp_storage, type='longest', mode='all', filter=filter)
    # _profile_response_time_for_experiment(experiment_info, exp_storage, type='longest', mode='all', filter=filter)
    # _profile_response_time_for_experiment(experiment_info, exp_storage, type='longest', mode='all', filter=filter)
    # _profile_response_time_for_experiment(experiment_info, exp_storage, type='longest', mode='all', filter=filter)

    return

def _profile_response_time_for_experiment(experiment_info, exp_storage, type, mode, filter=1.0):
    if type == 'shortest':
        label = 'Shortest'
    elif type == 'longest':
        label = 'Longest'
    else:
        print('[ERROR] Invalid mode:', label)
    
    available_mode = ['all', 'normal', 'collision', 'matching_failed']
    if mode not in available_mode:
        print('[ERROR] Invalidate mode:', mode)
        exit()

    is_collapsed_list = list(exp_storage.is_collapsed_list.values())
    is_matching_failed_list = list(exp_storage.is_matching_failed_list.values())

    n = len(is_collapsed_list)
    collision_cnt = sum(is_collapsed_list)
    sum_of_deadilne_miss_ratio = 0

    target_experiment_idx_list = []
    if mode == 'all': target_experiment_idx_list = range(n)
    elif mode == 'collision': target_experiment_idx_list = aa.get_idices_of_one_from_list(is_collapsed_list)
    elif mode == 'matching_failed': target_experiment_idx_list = aa.get_idices_of_one_from_list(is_matching_failed_list)        
    elif mode == 'normal':
        target_experiment_idx_list = list(range(n))
        merged_indices = aa.merge_binary_list_to_idx_list(is_collapsed_list, is_matching_failed_list)
        for remove_target in merged_indices:
            target_experiment_idx_list.remove(remove_target)
    
    all_E2E_response_time_list = []
    max_E2E_response_time_list = []
    leakage_ratio_list = []
    obj_instance_leakage_ratio_list = []
    min_response_time_list = []
    avg_response_time_list = []
    max_response_time_list = []
    min_instance_cnt_list = []
    avg_instance_cnt_list = []
    start_time_duration_list = []
    end_time_duration_list = []
    x_data = []
    max_obj_instance_leakage = -1
    for idx in target_experiment_idx_list:
        is_collapsed = is_collapsed_list[idx]        
        response_time_path = experiment_info.source_path + '/' + str(idx) + '/response_time'
        center_offset_path = experiment_info.source_path + '/' + str(idx) + '/center_offset.csv'
        first_node_path = response_time_path + '/' + experiment_info.first_node + '.csv'
        last_node_path = response_time_path + '/' + experiment_info.last_node + '.csv'
        missing_target_node_path = response_time_path + '/ndt_matching.csv'
        start_instance, end_instance = experiment_info.get_instance_pair(idx)
        if start_instance < 0: continue

        E2E_response_time = exp_storage.E2E_response_time_list[idx]
        # obj_instance_leakage_ratio, _max_obj_instance_leakage = aa.get_obj_instance_leakage(first_node_path, last_node_path, start_instance, end_instance, type)
        # min_response_time, avg_response_time, max_response_time, min_instance_cnt, avg_instance_cnt, avg_start_time_duration, avg_end_time_duration = \
        #  aa.get_node_frequency(first_node_path, first_node_path, start_instance, end_instance, type)
        # avg_start_time_duration, avg_end_time_duration = aa.get_node_frequency(first_node_path, first_node_path, start_instance, end_instance, type)
        # min_response_time_list.append(min_response_time)
        # avg_response_time_list.append(avg_response_time)
        # max_response_time_list.append(max_response_time)
        # min_instance_cnt_list.append(min_instance_cnt)
        # avg_instance_cnt_list.append(avg_instance_cnt)
        # start_time_duration_list.append(avg_start_time_duration)
        # end_time_duration_list.append(avg_end_time_duration)
        # obj_instance_leakage_ratio_list.append(obj_instance_leakage_ratio)
        # max_obj_instance_leakage = max([_max_obj_instance_leakage, max_obj_instance_leakage])

        # Profile instance leakge ratio
        instance_list = list(E2E_response_time.keys())
        # print(instance_list)
        # print(aa.get_missing_obj(missing_target_node_path, instance_list[0], instance_list[-1]))
        leakage_cnt = 0
        for i, instance in enumerate(instance_list):
            if i == 0: continue
            if instance_list[i] - instance_list[i-1] != 1: leakage_cnt = leakage_cnt + 1     
        leakage_ratio = float(leakage_cnt) / float(instance_list[-1] - instance_list[0])
        leakage_ratio_list.append(leakage_ratio)

        x_data = list(E2E_response_time.keys()) # Instance IDs
        y_data = list(E2E_response_time.values()) # E2E response time(ms)
        if not is_collapsed:
            x_data = x_data[:int(len(x_data) * filter)]
            y_data = y_data[:int(len(y_data) * filter)]
        if(len(y_data) > 0):
            all_E2E_response_time_list.extend(y_data)
            max_E2E_response_time_list.append(max(y_data))        

        # Validate miss deadline during avoidance
        deadline_miss_cnt = 0
        for instance in x_data:
            if not instance in E2E_response_time.keys(): continue
            if E2E_response_time[instance] >= experiment_info.E2E_deadline:
                deadline_miss_cnt = deadline_miss_cnt + 1
        
        if len(x_data) == 0:
            sum_of_deadilne_miss_ratio    
        else:
            sum_of_deadilne_miss_ratio = sum_of_deadilne_miss_ratio + float(deadline_miss_cnt)/float(len(x_data))

        color = 'b'
        if is_collapsed == 1: color = 'r'

        plt.plot(x_data, y_data, color, linewidth=1.0)


    # Statistics
    if len(all_E2E_response_time_list) == 0:
        max_E2E_response_time = 0    
        avg_E2E_response_time = 0
        var_E2E_response_time = 0
        avg_max_E2E_response_time = 0
        min_E2E_response_time = 0
        q1_E2E_response_time = 0
        q3_E2E_response_time = 0
    else:
        max_E2E_response_time = float(max(all_E2E_response_time_list)    )
        avg_E2E_response_time = float(sum(all_E2E_response_time_list) / len(all_E2E_response_time_list))
        var_E2E_response_time = float(np.var(all_E2E_response_time_list))
        avg_max_E2E_response_time = float(sum(max_E2E_response_time_list) / len(max_E2E_response_time_list))
        min_E2E_response_time = float(min(all_E2E_response_time_list))
        q1_E2E_response_time = float(np.quantile(all_E2E_response_time_list, .25))
        q3_E2E_response_time = float(np.quantile(all_E2E_response_time_list, .75))

    E2E_response_time_info_path = 'analyzation/' + experiment_info.output_title + '/' + experiment_info.experiment_title + '_E2E_response_time_info(' + mode + ',' + type + ').yaml'
    E2E_response_time_info = {}
    E2E_response_time_info['deadline_ms'] = experiment_info.E2E_deadline
    E2E_response_time_info['max'] = max_E2E_response_time
    E2E_response_time_info['avg'] = avg_E2E_response_time
    E2E_response_time_info['var'] = var_E2E_response_time
    E2E_response_time_info['avg_max'] = avg_max_E2E_response_time    
    E2E_response_time_info['min'] = min_E2E_response_time
    E2E_response_time_info['q1'] = q1_E2E_response_time
    E2E_response_time_info['q3'] = q3_E2E_response_time
    # E2E_response_time_info['E2E_list'] = all_E2E_response_time
    
    if collision_cnt != 0 and len(target_experiment_idx_list) != 0:
        E2E_response_time_info['avg_miss_ratio'] =  float(sum_of_deadilne_miss_ratio) / float(len(target_experiment_idx_list))
    else:
        E2E_response_time_info['avg_miss_ratio'] = 0.0    

    if len(leakage_ratio_list) == 0:
        E2E_response_time_info['avg_leakage_ratio'] = 0.0
        E2E_response_time_info['max_leakage_ratio'] = 0.0
    else:
        E2E_response_time_info['avg_leakage_ratio'] = float(sum(leakage_ratio_list)) / float(len(target_experiment_idx_list))
        E2E_response_time_info['max_leakage_ratio'] = float(max(leakage_ratio_list))

    if len(obj_instance_leakage_ratio_list) != 0:
        E2E_response_time_info['avg_obj_instance_leakage_ratio'] = float(sum(obj_instance_leakage_ratio_list)) / float(len(target_experiment_idx_list))
        E2E_response_time_info['max_obj_instance_leakage_ratio'] = float(max(obj_instance_leakage_ratio_list))
        E2E_response_time_info['max_obj_instance_leakage'] = max_obj_instance_leakage
    else:
        E2E_response_time_info['avg_obj_instance_leakage_ratio'] = 0.0
        E2E_response_time_info['max_obj_instance_leakage_ratio'] = 0.0
        E2E_response_time_info['max_obj_instance_leakage'] = 0.0

    if len(start_time_duration_list) != 0:
        E2E_response_time_info['min_response_time'] = float(min(min_response_time_list))
        E2E_response_time_info['avg_response_time'] = float(sum(avg_response_time_list)) / len(avg_response_time_list)
        E2E_response_time_info['max_response_time'] = float(max(max_response_time_list))
        E2E_response_time_info['min_instance_cnt'] = float(min(min_instance_cnt_list))
        E2E_response_time_info['avg_instance_cnt'] = float(sum(avg_instance_cnt_list)) / len(avg_instance_cnt_list)
        # min_response_time_list.append(min_response_time)
        # avg_response_time_list.append(avg_response_time)
        # max_response_time_list.append(max_response_time)
        # min_instance_cnt_list.append(min_instance_cnt)
        # avg_instance_cnt_list.append(avg_instance_cnt)
        
        E2E_response_time_info['avg_start_time_duration'] = float(sum(start_time_duration_list)) / float(len(start_time_duration_list))
        E2E_response_time_info['max_start_time_duration'] = float(max(start_time_duration_list))
        E2E_response_time_info['avg_end_time_duration'] = float(sum(end_time_duration_list)) / float(len(end_time_duration_list))
        E2E_response_time_info['max_end_time_duration'] = float(max(end_time_duration_list))
    else:
        E2E_response_time_info['avg_start_time_duration'] = 0.0
        E2E_response_time_info['max_start_time_duration'] = 0.0
        E2E_response_time_info['avg_end_time_duration'] = 0.0
        E2E_response_time_info['max_end_time_duration'] = 0.0


    with open(E2E_response_time_info_path, 'w') as f: yaml.dump(E2E_response_time_info, f, default_flow_style=False)

    if len(is_collapsed_list) == 0: collision_ratio = 0
    else: collision_ratio = sum(is_collapsed_list)/len(is_collapsed_list)

    if len(is_matching_failed_list) == 0: matching_failure_ratio = 0
    else: matching_failure_ratio = sum(is_matching_failed_list)/len(is_matching_failed_list)

    # Plot    
    plot_path = 'analyzation/' + experiment_info.output_title + '/' + experiment_info.experiment_title + '_' + mode + '_' + type + '_E2E_response_time.png'

    # plt.legend()
    plt.xlabel('Instance ID')
    plt.ylabel(label + ' E2E Response Time (ms)')
    plt.ylim(0, 1000)
    plt.yticks(np.arange(0,1000,100))
    if len(x_data) != 0:
        plt.text(min(x_data)+2, 950, 'max: ' + str(max_E2E_response_time))
        plt.text(min(x_data)+2, 900, 'avg: ' + str(avg_E2E_response_time))
        plt.text(min(x_data)+2, 850, 'var: ' + str(var_E2E_response_time))
        plt.text(min(x_data)+2, 800, 'avg_miss_ratio: ' + str(E2E_response_time_info['avg_miss_ratio']))
    plt.title('Iteration: ' + str(n) \
            + ' / Collision ratio: ' + str(collision_ratio) \
            + ' / Matching failure ratio: '+ str(matching_failure_ratio))
    plt.savefig(plot_path)
    plt.close()    

    return

def profile_waypoints_for_experiment(experiment_info, exp_storage):
    _profile_waypoints_for_experiment(experiment_info, exp_storage, mode='all')
    _profile_waypoints_for_experiment(experiment_info, exp_storage, mode='normal')
    _profile_waypoints_for_experiment(experiment_info, exp_storage, mode='collision')
    _profile_waypoints_for_experiment(experiment_info, exp_storage, mode='matching_failed')

def _profile_waypoints_for_experiment(experiment_info, exp_storage, mode='all'):
    available_mode = ['all', 'normal', 'collision', 'matching_failed']
    if mode not in available_mode:
        print('[ERROR] Invalidate mode:', mode)
        exit()
    
    is_collapsed_list = list(exp_storage.is_collapsed_list.values())
    is_matching_failed_list = list(exp_storage.is_matching_failed_list.values())

    n = len(is_collapsed_list)

    # Centerline
    center_line_x, center_line_y = experiment_info.get_center_line(0)

    plt.plot(center_line_x, center_line_y, 'k', label='Center line')
    
    target_experiment_idx_list = []
    if mode == 'all': target_experiment_idx_list = range(n)
    elif mode == 'collision':
        target_experiment_idx_list = aa.get_idices_of_one_from_list(is_collapsed_list)
    elif mode == 'matching_failed': target_experiment_idx_list = aa.get_idices_of_one_from_list(is_matching_failed_list)
    else:
        target_experiment_idx_list = list(range(n))        
        merged_indices = aa.merge_binary_list_to_idx_list(is_collapsed_list, is_matching_failed_list)
        for remove_target in merged_indices:
            target_experiment_idx_list.remove(remove_target)

    # Waypoints
    for idx in target_experiment_idx_list:        
        exp_id = str(idx)
        label = experiment_info.experiment_title + '_' + exp_id

        waypoints_x, waypoints_y = experiment_info.get_waypoints(idx)

        color = 'b'
        if is_collapsed_list[idx] == 1: color = 'r' 

        plt.plot(waypoints_x, waypoints_y, color, linewidth=1.0)


    if experiment_info.simulator == 'old':
        # Objects
        npc1_x = [6, 6, -1, -1, 6]
        npc1_y = [51, 48, 48, 51, 51]
        npc2_x = [6, 6, -1, -1, 6]
        npc2_y = [55, 52, 52, 55, 55]
        plt.plot(npc1_x, npc1_y, 'k')
        plt.plot(npc2_x, npc2_y, 'k')
    elif experiment_info.simulator == 'carla' or experiment_info.simulator == 'svl':
        pass

    if len(is_collapsed_list) == 0: collision_ratio = 0
    else: collision_ratio = sum(is_collapsed_list)/len(is_collapsed_list)

    if len(is_matching_failed_list) == 0: matching_failure_ratio = 0
    else: matching_failure_ratio = sum(is_matching_failed_list)/len(is_matching_failed_list)

    # Plot
    plot_path = 'analyzation/' + experiment_info.output_title + '/' + experiment_info.experiment_title + '_' + mode + '_waypoints.png'
    
    if experiment_info.simulator == 'old':
        plt.xlim(-70, 40)
        plt.ylim(20,75)
    elif experiment_info.simulator == 'carla' or experiment_info.simulator == 'svl':
        pass
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Iteration: ' + str(n) \
            + ' / Collision ratio: ' + str(collision_ratio) \
            + ' / Matching failure ratio: '+ str(matching_failure_ratio))
    plt.savefig(plot_path)

    plt.close()

def iteration_anaylzer(experiment_info, idx, exp_storage):
    
    #is collapsed
    is_collapsed = experiment_info.get_experiment_info(idx)
    if 'is_collapsed' not in experiment_info.get_experiment_info(idx):
        is_collapsed = experiment_info.get_experiment_info(idx)['is_collaped']
    else:
        is_collapsed = experiment_info.get_experiment_info(idx)['is_collapsed']
    exp_storage.is_collapsed_list[idx] = is_collapsed
    # print(is_collapsed)

    # Center Offset
    profile_center_offset(experiment_info, idx, exp_storage)

    # Check matching is failed
    is_matching_failed = experiment_info.check_matching_is_failed(idx)
    exp_storage.is_matching_failed_list[idx] = is_matching_failed

    # E2E response time during avoidance
    profile_response_time(experiment_info, idx, exp_storage)

    # Miss alignment delay for whold driving
    #profile miss alignment delay
    # print('hi')
    max_miss_alignment_delay, avg_miss_alignment_delay = profile_miss_alignment_delay(experiment_info, idx, exp_storage)
    exp_storage.max_miss_alignment_delay_list[idx] = max_miss_alignment_delay
    exp_storage.avg_miss_alignment_delay_list[idx] = avg_miss_alignment_delay

    # Trajectories
    profile_waypoints(experiment_info, idx, exp_storage)

def experiment_anaylzer(experiment_configs):
    experiment_manager = mp.Manager()
    experiment_info = aa.EXPERIMENT_INFO(experiment_configs)

    exp_storage = aa.EXPERIMENT_STORAGE(experiment_manager)

    idx_list = [i for i in range(experiment_info.iteration)]
    exp_pbar = tqdm(total=experiment_info.iteration, position=0)
    exp_pbar.set_description(experiment_info.output_title)
    exp_pbar.update(0)
    exp_executor = concurrent.futures.ProcessPoolExecutor()
    # print(experiment_info.get_experiment_info(3))
    exp_futures = [exp_executor.submit(iteration_anaylzer, experiment_info, idx, exp_storage) for idx in idx_list]
    # exit()
    
    for exp_future in concurrent.futures.as_completed(exp_futures):
        # result = exp_future.result()
        exp_pbar.update(1)
  
    exp_storage.is_collapsed_list = dict(sorted(exp_storage.is_collapsed_list.items()))
    exp_storage.is_matching_failed_list = dict(sorted(exp_storage.is_matching_failed_list.items()))
    exp_storage.sorting_dicts()

    profile_analyzation_info(experiment_info, exp_storage)

    profile_response_time_for_experiment(experiment_info, exp_storage)

    profile_waypoints_for_experiment(experiment_info, exp_storage)

  
    exp_pbar.clear()


if __name__ == '__main__':
    with open('yaml/autoware_analyzer.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    experiment_configs = {}
    experiment_configs['node_chain'] = configs['node_chain']
    experiment_configs['avoidance_x_range'] = configs['avoidance_x_range']
    experiment_configs['simulator'] = configs['simulator']
    experiment_configs['type'] = 'shortest'

    experiment_processes = []
    main_pbar = tqdm(total=len(configs['experiment_title']))
    main_pbar.set_description('All experiment')
    main_pbar.update(0)
    main_executor = concurrent.futures.ProcessPoolExecutor(max_workers=5)
    experiment_configs_list = []

    for i in range(len(configs['experiment_title'])):
        experiment_configs['experiment_title'] = configs['experiment_title'][i]
        experiment_configs['output_title'] = configs['output_title'][i]
        experiment_configs['first_node'] = configs['first_node'][0]
        experiment_configs['last_node'] = configs['last_node'][0]
        experiment_configs['E2E_deadline'] = configs['E2E_deadline'][0]
        experiment_configs_list.append(copy.deepcopy(experiment_configs))

    # for arg in experiment_configs_list:
    #     experiment_anaylzer(arg)
    #     main_pbar.update(1)

    main_futures = [main_executor.submit(experiment_anaylzer, arg) for arg in experiment_configs_list]

    for main_future in concurrent.futures.as_completed(main_futures):
        # result = main_future.result()
        main_pbar.update(1)
        
        
    #     experiment_process = mp.Process(target=experiment_anaylzer, args=(experiment_configs, ))
    #     experiment_processes.append(experiment_process)
    #     experiment_process.start()
    
    # for experiment_process in experiment_processes:
    #     experiment_process.join()
