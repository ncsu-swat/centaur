
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cdist_inputs():
    list_of_inputs = []
    
    x1_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2_1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    p_1 = 2.0
    compute_mode_1 = 'use_mm_for_euclid_dist_if_necessary'
    input_dict_1 = {'x1': x1_1, 'x2': x2_1, 'p': p_1, 'compute_mode': compute_mode_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    x1_2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    x2_2 = np.array([[-1.0, -2.0, -3.0]])
    p_2 = 1.0
    compute_mode_2 = 'use_mm_for_euclid_dist'
    input_dict_2 = {'x1': x1_2, 'x2': x2_2, 'p': p_2, 'compute_mode': compute_mode_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    x1_3 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    x2_3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p_3 = 3.0
    compute_mode_3 = 'donot_use_mm_for_euclid_dist'
    input_dict_3 = {'x1': x1_3, 'x2': x2_3, 'p': p_3, 'compute_mode': compute_mode_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    x1_4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    x2_4 = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])
    p_4 = 0.5
    compute_mode_4 = 'use_mm_for_euclid_dist_if_necessary'
    input_dict_4 = {'x1': x1_4, 'x2': x2_4, 'p': p_4, 'compute_mode': compute_mode_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    x1_5 = np.array([[1.0, 2.0, 3.0]])
    x2_5 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    p_5 = 2.0
    compute_mode_5 = 'donot_use_mm_for_euclid_dist'
    input_dict_5 = {'x1': x1_5, 'x2': x2_5, 'p': p_5, 'compute_mode': compute_mode_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    x1_6 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_6 = np.array([[7.0, 8.0], [9.0, 10.0]])
    p_6 = 1.5
    compute_mode_6 = 'use_mm_for_euclid_dist'
    input_dict_6 = {'x1': x1_6, 'x2': x2_6, 'p': p_6, 'compute_mode': compute_mode_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    x1_7 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    x2_7 = np.array([[5.0, -6.0], [-7.0, 8.0]])
    p_7 = 2.0
    compute_mode_7 = 'use_mm_for_euclid_dist_if_necessary'
    input_dict_7 = {'x1': x1_7, 'x2': x2_7, 'p': p_7, 'compute_mode': compute_mode_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    x1_8 = np.array([[0.0, 0.0], [0.0, 0.0]])
    x2_8 = np.array([[1.0, 1.0], [2.0, 2.0]])
    p_8 = 2.0
    compute_mode_8 = 'donot_use_mm_for_euclid_dist'
    input_dict_8 = {'x1': x1_8, 'x2': x2_8, 'p': p_8, 'compute_mode': compute_mode_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    x1_9 = np.array([[1.0, 2.0, 3.0, 4.0]])
    x2_9 = np.array([[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    p_9 = 0.5
    compute_mode_9 = 'use_mm_for_euclid_dist'
    input_dict_9 = {'x1': x1_9, 'x2': x2_9, 'p': p_9, 'compute_mode': compute_mode_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    x1_10 = np.array([[1.0], [2.0], [3.0]])
    x2_10 = np.array([[4.0], [5.0], [6.0]])
    p_10 = 2.0
    compute_mode_10 = 'use_mm_for_euclid_dist_if_necessary'
    input_dict_10 = {'x1': x1_10, 'x2': x2_10, 'p': p_10, 'compute_mode': compute_mode_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.cdist"] = cdist_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cdist'.")


check_valid('torch.cdist', generated_inputs['torch.cdist'], lib="torch", suffix=0)
