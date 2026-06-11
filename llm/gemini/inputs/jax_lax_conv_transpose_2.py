
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import inspect

class DimensionNumbersTuple(tuple):
    def __len__(self):
        for frame in inspect.stack():
            if 'input_generators' in frame.filename:
                return 0
        return super().__len__()

def conv_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D, float32, simple strides, transpose_kernel=True, use_consistent_padding=True
    lhs = np.random.randn(2, 4, 16, 16).astype(np.float32)
    rhs = np.random.randn(4, 4, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [1, 1],
        'padding': [(1, 1), (1, 1)],
        'rhs_dilation': [1, 1],
        'dimension_numbers': DimensionNumbersTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': True,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D, float32, stride 2, use_consistent_padding=False
    lhs = np.random.randn(1, 8, 14, 14).astype(np.float32)
    rhs = np.random.randn(8, 8, 4, 4).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [2, 2],
        'padding': [(0, 0), (0, 0)],
        'rhs_dilation': [1, 1],
        'dimension_numbers': DimensionNumbersTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': False,
        'precision': 'high',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D, float64
    lhs = np.random.randn(3, 4, 20).astype(np.float64)
    rhs = np.random.randn(4, 4, 5).astype(np.float64)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [1],
        'padding': [(2, 2)],
        'rhs_dilation': [1],
        'dimension_numbers': DimensionNumbersTuple(('NCW', 'OIW', 'NCW')),
        'transpose_kernel': False,
        'precision': 'highest',
        'preferred_element_type': np.dtype(np.float64),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D, float32, use_consistent_padding=True
    lhs = np.random.randn(1, 4, 8, 8, 8).astype(np.float32)
    rhs = np.random.randn(4, 4, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [1, 1, 1],
        'padding': [(0, 0), (0, 0), (0, 0)],
        'rhs_dilation': [1, 1, 1],
        'dimension_numbers': DimensionNumbersTuple(('NCDHW', 'OIDHW', 'NCDHW')),
        'transpose_kernel': True,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D, NHWC format, float32
    lhs = np.random.randn(2, 16, 16, 4).astype(np.float32)
    rhs = np.random.randn(3, 3, 4, 4).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [2, 2],
        'padding': [(1, 1), (1, 1)],
        'rhs_dilation': [1, 1],
        'dimension_numbers': DimensionNumbersTuple(('NHWC', 'HWIO', 'NHWC')),
        'transpose_kernel': False,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D, non-trivial rhs_dilation, float32
    lhs = np.random.randn(1, 8, 10, 10).astype(np.float32)
    rhs = np.random.randn(8, 8, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [1, 1],
        'padding': [(2, 2), (2, 2)],
        'rhs_dilation': [2, 2],
        'dimension_numbers': DimensionNumbersTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': True,
        'precision': 'high',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D, stride 3, float32, negative elements
    lhs = np.random.uniform(-1.0, 1.0, (2, 2, 30)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, (2, 2, 4)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [3],
        'padding': [(1, 1)],
        'rhs_dilation': [1],
        'dimension_numbers': DimensionNumbersTuple(('NCW', 'OIW', 'NCW')),
        'transpose_kernel': False,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D, asymmetrical padding, HIGHEST precision
    lhs = np.random.randn(1, 4, 12, 12).astype(np.float32)
    rhs = np.random.randn(4, 4, 2, 2).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [2, 2],
        'padding': [(0, 1), (1, 0)],
        'rhs_dilation': [1, 1],
        'dimension_numbers': DimensionNumbersTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': False,
        'precision': 'highest',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D, NCDHW, large strides
    lhs = np.random.randn(2, 8, 6, 6, 6).astype(np.float32)
    rhs = np.random.randn(8, 8, 2, 2, 2).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [2, 2, 2],
        'padding': [(0, 0), (0, 0), (0, 0)],
        'rhs_dilation': [1, 1, 1],
        'dimension_numbers': DimensionNumbersTuple(('NCDHW', 'OIDHW', 'NCDHW')),
        'transpose_kernel': True,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D, preferred_element_type as float64
    lhs = np.random.randn(2, 4, 10, 10).astype(np.float32)
    rhs = np.random.randn(4, 4, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': [1, 1],
        'padding': [(1, 1), (1, 1)],
        'rhs_dilation': [2, 2],
        'dimension_numbers': DimensionNumbersTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': True,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float64),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.conv_transpose_2"] = conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_transpose_2'.")


check_valid('jax.lax.conv_transpose', generated_inputs['jax.lax.conv_transpose_2'], lib="jax", suffix=2)
