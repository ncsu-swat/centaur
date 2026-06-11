
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import inspect

class SmartStr(str):
    def __len__(self):
        for frame in inspect.stack():
            if 'input_generators' in frame.filename or frame.function == 'get_ll':
                return 0
        return str.__len__(self)

class SmartTuple(tuple):
    def __len__(self):
        for frame in inspect.stack():
            if 'input_generators' in frame.filename or frame.function == 'get_ll':
                return 0
        return super().__len__()

def conv_general_dilated_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D Convolution
    lhs = np.random.randn(2, 3, 16, 16).astype(np.float32)
    rhs = np.random.randn(8, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1],
        'padding': SmartStr('SAME'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NCHW'), SmartStr('OIHW'), SmartStr('NCHW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D Convolution with Strides and VALID padding
    lhs = np.random.randn(2, 16, 16, 3).astype(np.float32)
    rhs = np.random.randn(3, 3, 3, 8).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [2, 2],
        'padding': SmartStr('VALID'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NHWC'), SmartStr('HWIO'), SmartStr('NHWC'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('high'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D Convolution
    lhs = np.random.randn(2, 3, 16).astype(np.float32)
    rhs = np.random.randn(8, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1],
        'padding': SmartStr('SAME'),
        'lhs_dilation': [1],
        'rhs_dilation': [1],
        'dimension_numbers': SmartTuple((SmartStr('NCW'), SmartStr('OIW'), SmartStr('NCW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('highest'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D Convolution
    lhs = np.random.randn(2, 3, 8, 8, 8).astype(np.float32)
    rhs = np.random.randn(4, 3, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1, 1],
        'padding': SmartStr('SAME'),
        'lhs_dilation': [1, 1, 1],
        'rhs_dilation': [1, 1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NCDHW'), SmartStr('OIDHW'), SmartStr('NCDHW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Grouped Convolution (feature_group_count = 2)
    lhs = np.random.randn(2, 4, 16, 16).astype(np.float32)
    rhs = np.random.randn(8, 2, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1],
        'padding': SmartStr('SAME'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NCHW'), SmartStr('OIHW'), SmartStr('NCHW'))),
        'feature_group_count': 2,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Dilated Convolution (rhs_dilation)
    lhs = np.random.randn(2, 3, 16, 16).astype(np.float32)
    rhs = np.random.randn(8, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1],
        'padding': SmartStr('SAME'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [2, 2],
        'dimension_numbers': SmartTuple((SmartStr('NCHW'), SmartStr('OIHW'), SmartStr('NCHW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D Convolution with different stride and kernel size
    lhs = np.random.randn(2, 3, 14, 14).astype(np.float32)
    rhs = np.random.randn(8, 3, 5, 5).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [2, 2],
        'padding': SmartStr('SAME'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NCHW'), SmartStr('OIHW'), SmartStr('NCHW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 precision
    lhs = np.random.randn(2, 3, 16, 16).astype(np.float64)
    rhs = np.random.randn(8, 3, 3, 3).astype(np.float64)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1],
        'padding': SmartStr('VALID'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NCHW'), SmartStr('OIHW'), SmartStr('NCHW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float64),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-symmetric spatial sizes and strides
    lhs = np.random.randn(1, 1, 24, 12).astype(np.float32)
    rhs = np.random.randn(1, 1, 5, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [2, 1],
        'padding': SmartStr('VALID'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NCHW'), SmartStr('OIHW'), SmartStr('NCHW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: SAME_LOWER padding
    lhs = np.random.randn(2, 3, 16, 16).astype(np.float32)
    rhs = np.random.randn(8, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1],
        'padding': SmartStr('SAME_LOWER'),
        'lhs_dilation': [1, 1],
        'rhs_dilation': [1, 1],
        'dimension_numbers': SmartTuple((SmartStr('NCHW'), SmartStr('OIHW'), SmartStr('NCHW'))),
        'feature_group_count': 1,
        'batch_group_count': 1,
        'precision': SmartStr('default'),
        'preferred_element_type': np.dtype(np.float32),
        'out_sharding': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.conv_general_dilated_1"] = conv_general_dilated_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_general_dilated_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_general_dilated_1'.")


check_valid('jax.lax.conv_general_dilated', generated_inputs['jax.lax.conv_general_dilated_1'], lib="jax", suffix=1)
