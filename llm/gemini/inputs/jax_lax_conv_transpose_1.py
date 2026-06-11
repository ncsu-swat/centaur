
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class SafeTuple(tuple):
    def __array__(self, dtype=None, copy=None):
        return np.array(list(self), dtype=object)

def conv_transpose_inputs():
    list_of_inputs = []
    
    # 1. 1D convolution transpose, simple case
    lhs = np.random.randn(2, 1, 10).astype(np.float32)
    rhs = np.random.randn(1, 1, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (1,),
        'padding': 'SAME',
        'rhs_dilation': (1,),
        'dimension_numbers': SafeTuple(('NCW', 'OIW', 'NCW')),
        'transpose_kernel': False,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 2. 2D conv transpose, VALID padding, kernel transposing
    lhs = np.random.randn(2, 3, 10, 10).astype(np.float32)
    rhs = np.random.randn(3, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (2, 2),
        'padding': 'VALID',
        'rhs_dilation': (1, 1),
        'dimension_numbers': SafeTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': True,
        'precision': 'high',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 3. 3D conv transpose with dilation and float64 type
    lhs = np.random.randn(2, 2, 5, 5, 5).astype(np.float64)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (1, 1, 1),
        'padding': 'SAME',
        'rhs_dilation': (2, 2, 2),
        'dimension_numbers': SafeTuple(('NCDHW', 'OIDHW', 'NCDHW')),
        'transpose_kernel': False,
        'precision': 'highest',
        'preferred_element_type': np.dtype(np.float64),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 4. 2D conv transpose with NHWC/HWIO format
    lhs = np.random.randn(2, 8, 8, 4).astype(np.float32)
    rhs = np.random.randn(3, 3, 4, 4).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (3, 3),
        'padding': 'SAME',
        'rhs_dilation': (1, 1),
        'dimension_numbers': SafeTuple(('NHWC', 'HWIO', 'NHWC')),
        'transpose_kernel': False,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 5. 1D conv transpose with NWC format
    lhs = np.random.randn(2, 10, 2).astype(np.float32)
    rhs = np.random.randn(3, 2, 2).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (2,),
        'padding': 'VALID',
        'rhs_dilation': (1,),
        'dimension_numbers': SafeTuple(('NWC', 'WIO', 'NWC')),
        'transpose_kernel': True,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 6. 2D conv transpose with float16 type
    lhs = np.random.randn(1, 1, 16, 16).astype(np.float16)
    rhs = np.random.randn(1, 1, 5, 5).astype(np.float16)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (1, 1),
        'padding': 'SAME',
        'rhs_dilation': (1, 1),
        'dimension_numbers': SafeTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': False,
        'precision': 'high',
        'preferred_element_type': np.dtype(np.float16),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 7. Larger channel size
    lhs = np.random.randn(4, 8, 14, 14).astype(np.float32)
    rhs = np.random.randn(8, 8, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (2, 2),
        'padding': 'SAME',
        'rhs_dilation': (1, 1),
        'dimension_numbers': SafeTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': False,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 8. 3D conv transpose with VALID padding and kernel transpose
    lhs = np.random.randn(1, 1, 8, 8, 8).astype(np.float32)
    rhs = np.random.randn(1, 1, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (2, 2, 2),
        'padding': 'VALID',
        'rhs_dilation': (1, 1, 1),
        'dimension_numbers': SafeTuple(('NCDHW', 'OIDHW', 'NCDHW')),
        'transpose_kernel': True,
        'precision': 'high',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 9. 1D conv transpose, larger channels and strides
    lhs = np.random.randn(2, 16, 20).astype(np.float32)
    rhs = np.random.randn(16, 16, 4).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (3,),
        'padding': 'SAME',
        'rhs_dilation': (2,),
        'dimension_numbers': SafeTuple(('NCW', 'OIW', 'NCW')),
        'transpose_kernel': False,
        'precision': 'highest',
        'preferred_element_type': np.dtype(np.float32),
        'use_consistent_padding': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 10. Asymmetric stride for 2D conv transpose
    lhs = np.random.randn(2, 2, 12, 12).astype(np.float64)
    rhs = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'strides': (1, 2),
        'padding': 'VALID',
        'rhs_dilation': (1, 1),
        'dimension_numbers': SafeTuple(('NCHW', 'OIHW', 'NCHW')),
        'transpose_kernel': True,
        'precision': 'default',
        'preferred_element_type': np.dtype(np.float64),
        'use_consistent_padding': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["jax.lax.conv_transpose_1"] = conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_transpose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_transpose_1'.")


check_valid('jax.lax.conv_transpose', generated_inputs['jax.lax.conv_transpose_1'], lib="jax", suffix=1)
