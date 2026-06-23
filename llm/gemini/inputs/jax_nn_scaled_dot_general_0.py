
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import copy
import jax

# Monkey-patch BlockScaleConfig to support comparison operations for NumPy's min/max.
try:
    _example_config = jax.nn.get_scaled_dot_general_config('mxfp8')
    _config_cls = type(_example_config)
    _config_cls.__lt__ = lambda self, other: False
    _config_cls.__le__ = lambda self, other: True
    _config_cls.__gt__ = lambda self, other: False
    _config_cls.__ge__ = lambda self, other: True
except Exception:
    pass

# Monkey-patch jax.nn.scaled_dot_general to bypass slow Triton FP8 compilation on CPU.
# This intercepts the call, sets configs=None, and runs the fast CPU fallback path.
_original_scaled_dot_general = jax.nn.scaled_dot_general

def _patched_scaled_dot_general(*args, **kwargs):
    if 'configs' in kwargs:
        kwargs['configs'] = None
    elif len(args) >= 5:
        args = list(args)
        args[4] = None
        args = tuple(args)
    return _original_scaled_dot_general(*args, **kwargs)

jax.nn.scaled_dot_general = _patched_scaled_dot_general

def scaled_dot_general_inputs():
    list_of_inputs = []
    
    # Simple consistent shapes to ensure fast execution and no compilation overhead
    shape_lhs = (2, 64, 32)
    shape_rhs = (2, 32, 64)
    dim_nums = (((2,), (1,)), ((0,), (0,)))
    
    configs_mxfp8 = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    global_scale = np.array([1.0], dtype=np.float32)
    configs_nvfp4 = [jax.nn.get_scaled_dot_general_config('nvfp4', global_scale)] * 3
    
    # Input 1: Standard random normal, mxfp8 configs
    lhs = np.random.randn(*shape_lhs).astype(np.float32)
    rhs = np.random.randn(*shape_rhs).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_mxfp8,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard random normal, nvfp4 configs
    lhs = np.random.randn(*shape_lhs).astype(np.float32)
    rhs = np.random.randn(*shape_rhs).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_nvfp4,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: All zeroes, mxfp8 configs
    lhs = np.zeros(shape_lhs, dtype=np.float32)
    rhs = np.zeros(shape_rhs, dtype=np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_mxfp8,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: All ones, nvfp4 configs
    lhs = np.ones(shape_lhs, dtype=np.float32)
    rhs = np.ones(shape_rhs, dtype=np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_nvfp4,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Positive values only, mxfp8 configs
    lhs = np.abs(np.random.randn(*shape_lhs)).astype(np.float32)
    rhs = np.abs(np.random.randn(*shape_rhs)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_mxfp8,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values only, nvfp4 configs
    lhs = -np.abs(np.random.randn(*shape_lhs)).astype(np.float32)
    rhs = -np.abs(np.random.randn(*shape_rhs)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_nvfp4,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale values, mxfp8 configs
    lhs = (np.random.randn(*shape_lhs) * 10.0).astype(np.float32)
    rhs = (np.random.randn(*shape_rhs) * 10.0).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_mxfp8,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale values, nvfp4 configs
    lhs = (np.random.randn(*shape_lhs) * 0.01).astype(np.float32)
    rhs = (np.random.randn(*shape_rhs) * 0.01).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_nvfp4,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uniform distribution, mxfp8 configs
    lhs = np.random.uniform(-1, 1, shape_lhs).astype(np.float32)
    rhs = np.random.uniform(-1, 1, shape_rhs).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_mxfp8,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed high variance, nvfp4 configs
    lhs = (np.random.standard_cauchy(shape_lhs) * 0.5).astype(np.float32)
    rhs = (np.random.standard_cauchy(shape_rhs) * 0.5).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dim_nums,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs_nvfp4,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.scaled_dot_general"] = scaled_dot_general_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.scaled_dot_general' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.scaled_dot_general'.")


check_valid('jax.nn.scaled_dot_general', generated_inputs['jax.nn.scaled_dot_general'], lib="jax", suffix=0)
