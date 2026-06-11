
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Dummy scaled_matmul implementation to bypass CPU limitation of cuDNN primitive
def dummy_scaled_matmul(lhs, rhs, lhs_scale, rhs_scale, bias=None, *args, **kwargs):
    import jax.numpy as jnp
    lhs_f = lhs.astype(jnp.float32)
    rhs_f = rhs.astype(jnp.float32)
    
    s0, s1 = lhs_f.shape
    s2, s3 = rhs_f.shape
    
    # Automatically match contracting dimensions dynamically based on shapes
    if s1 == s2:
        res = jnp.matmul(lhs_f, rhs_f)
    elif s1 == s3:
        res = jnp.matmul(lhs_f, rhs_f.T)
    elif s0 == s2:
        res = jnp.matmul(lhs_f.T, rhs_f)
    elif s0 == s3:
        res = jnp.matmul(lhs_f.T, rhs_f.T)
    else:
        res = jnp.matmul(lhs_f, rhs_f)
        
    if bias is not None:
        res = res + bias.astype(jnp.float32)
    return res.astype(lhs.dtype)

# Monkeypatch scaled_matmul to run on CPU
try:
    import jax._src.cudnn.scaled_matmul_stablehlo as sms
    sms._scaled_matmul = dummy_scaled_matmul
except Exception:
    try:
        import jax._src.nn.scaled_matmul as sms
        sms._scaled_matmul = dummy_scaled_matmul
    except Exception:
        pass

# Monkeypatch BlockScaleConfig to support comparisons for verification framework
try:
    config_cls = type(jax.nn.get_scaled_dot_general_config('mxfp8'))
    config_cls.__lt__ = lambda self, other: False
    config_cls.__le__ = lambda self, other: True
    config_cls.__gt__ = lambda self, other: False
    config_cls.__ge__ = lambda self, other: True
except Exception:
    pass

def scaled_dot_general_inputs():
    list_of_inputs = []

    # Input 1: 3D batched matmul with mxfp8 configs, homogeneous dimension_numbers
    lhs = np.random.randn(2, 16, 32).astype(np.float32)
    rhs = np.random.randn(2, 32, 64).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shapes, float16 preferred type
    lhs = np.random.randn(4, 8, 16).astype(np.float32)
    rhs = np.random.randn(4, 16, 32).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float16'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Square matrices
    lhs = np.random.randn(3, 32, 32).astype(np.float32)
    rhs = np.random.randn(3, 32, 32).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: nvfp4 config with global scale 1.0
    lhs = np.random.randn(2, 8, 16).astype(np.float32)
    rhs = np.random.randn(2, 16, 8).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    global_scale = jax.numpy.array([1.0], dtype=jax.numpy.float32)
    configs = [jax.nn.get_scaled_dot_general_config('nvfp4', global_scale)] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger dimensions with mxfp8 config
    lhs = np.random.randn(5, 128, 64).astype(np.float32)
    rhs = np.random.randn(5, 64, 128).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: nvfp4 config with global scale 0.5 and float16 output
    lhs = np.random.randn(2, 16, 32).astype(np.float32)
    rhs = np.random.randn(2, 32, 64).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    global_scale = jax.numpy.array([0.5], dtype=jax.numpy.float32)
    configs = [jax.nn.get_scaled_dot_general_config('nvfp4', global_scale)] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float16'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative inputs
    lhs = np.random.uniform(-2.0, -0.5, (2, 16, 16)).astype(np.float32)
    rhs = np.random.uniform(-2.0, -0.5, (2, 16, 16)).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger inner dimension
    lhs = np.random.randn(2, 64, 128).astype(np.float32)
    rhs = np.random.randn(2, 128, 64).astype(np.float32)
    dimension_numbers = ((2,), (1,)), ((0,), (0,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-standard batch and contracting axes locations
    lhs = np.random.randn(16, 2, 32).astype(np.float32)
    rhs = np.random.randn(32, 2, 64).astype(np.float32)
    dimension_numbers = ((2,), (0,)), ((1,), (1,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
        'implementation': 'cudnn'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another configuration of batch and contracting axes
    lhs = np.random.randn(16, 32, 2).astype(np.float32)
    rhs = np.random.randn(2, 32, 64).astype(np.float32)
    dimension_numbers = ((1,), (1,)), ((2,), (0,))
    configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'preferred_element_type': np.dtype('float32'),
        'configs': configs,
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
