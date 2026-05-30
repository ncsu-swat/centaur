
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.lax_numpy as lax_numpy

# Monkeypatch jax.numpy.pad to filter out unsupported keyword arguments for the active mode.
original_pad = jnp.pad

def patched_pad(array, pad_width, mode='constant', **kwargs):
    allowed_kwargs = {
        'constant': {'constant_values'},
        'edge': set(),
        'linear_ramp': {'end_values'},
        'maximum': {'stat_length'},
        'mean': {'stat_length'},
        'median': {'stat_length'},
        'minimum': {'stat_length'},
        'reflect': {'reflect_type'},
        'symmetric': {'reflect_type'},
        'wrap': set(),
        'empty': set(),
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs.get(mode, set())}
    return original_pad(array, pad_width, mode=mode, **filtered_kwargs)

jnp.pad = patched_pad
lax_numpy.pad = patched_pad

def pad_inputs():
    list_of_inputs = []

    # Input 1: Constant mode with standard pad width
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "pad_width": 1,
        "mode": "constant",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Constant mode, larger pad, negative constant
    input_dict = {
        "array": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pad_width": 2,
        "mode": "constant",
        "constant_values": -1.5,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Edge mode
    input_dict = {
        "array": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "pad_width": 1,
        "mode": "edge",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Linear ramp mode
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "pad_width": 2,
        "mode": "linear_ramp",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": -1.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Maximum mode
    input_dict = {
        "array": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pad_width": 1,
        "mode": "maximum",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mean mode
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "pad_width": 2,
        "mode": "mean",
        "constant_values": 0.0,
        "stat_length": 2,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Median mode
    input_dict = {
        "array": np.array([1.0, 3.0, 6.0], dtype=np.float32),
        "pad_width": 1,
        "mode": "median",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Minimum mode
    input_dict = {
        "array": np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        "pad_width": 2,
        "mode": "minimum",
        "constant_values": 0.0,
        "stat_length": 2,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Reflect mode (even reflection)
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "pad_width": 1,
        "mode": "reflect",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Symmetric mode (odd reflection)
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "pad_width": 2,
        "mode": "symmetric",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "odd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Wrap mode
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "pad_width": 1,
        "mode": "wrap",
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.pad_1"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.pad_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.pad_1'.")


check_valid('jax.numpy.pad', generated_inputs['jax.numpy.pad_1'], lib="jax", suffix=1)
