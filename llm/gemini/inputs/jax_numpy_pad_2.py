
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkey-patch jax.numpy.pad to filter out unsupported kwargs based on the active mode.
# This resolves the conflict between the test framework requiring all signature keys
# and JAX throwing a ValueError for extraneous kwargs.
_original_pad = jnp.pad

def _wrapped_pad(array, pad_width, mode='constant', **kwargs):
    allowed_kwargs = {
        'constant': {'constant_values'},
        'empty': set(),
        'edge': set(),
        'wrap': set(),
        'linear_ramp': {'end_values'},
        'maximum': {'stat_length'},
        'mean': {'stat_length'},
        'median': {'stat_length'},
        'minimum': {'stat_length'},
        'reflect': {'reflect_type'},
        'symmetric': {'reflect_type'},
    }
    active_allowed = allowed_kwargs.get(mode, set())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in active_allowed}
    return _original_pad(array, pad_width, mode=mode, **filtered_kwargs)

jnp.pad = _wrapped_pad
jax.numpy.pad = _wrapped_pad

def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D array, constant mode
    input_dict = {
        'array': np.arange(6, dtype=np.float32),
        'pad_width': ((2, 3),),
        'mode': 'constant',
        'constant_values': 1.5,
        'stat_length': ((1, 1),),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, constant mode
    input_dict = {
        'array': np.ones((3, 3), dtype=np.float32),
        'pad_width': ((1, 1), (2, 2)),
        'mode': 'constant',
        'constant_values': -1.0,
        'stat_length': ((1, 1), (1, 1)),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, edge mode
    input_dict = {
        'array': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'pad_width': ((1, 1),),
        'mode': 'edge',
        'constant_values': 0.0,
        'stat_length': ((1, 1),),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, reflect mode
    input_dict = {
        'array': np.random.randn(4, 4).astype(np.float32),
        'pad_width': ((2, 2), (1, 1)),
        'mode': 'reflect',
        'constant_values': 0.0,
        'stat_length': ((1, 1), (1, 1)),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, symmetric mode
    input_dict = {
        'array': np.random.randn(3, 3, 3).astype(np.float32),
        'pad_width': ((1, 1), (1, 1), (1, 1)),
        'mode': 'symmetric',
        'constant_values': 0.0,
        'stat_length': ((1, 1), (1, 1), (1, 1)),
        'end_values': 0.0,
        'reflect_type': 'odd'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array, linear_ramp mode
    input_dict = {
        'array': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'pad_width': ((2, 2),),
        'mode': 'linear_ramp',
        'constant_values': 0.0,
        'stat_length': ((1, 1),),
        'end_values': 10.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, maximum mode
    input_dict = {
        'array': np.random.randn(5, 5).astype(np.float32),
        'pad_width': ((1, 1), (1, 1)),
        'mode': 'maximum',
        'constant_values': 0.0,
        'stat_length': ((2, 2), (2, 2)),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, mean mode
    input_dict = {
        'array': np.random.randn(6).astype(np.float32),
        'pad_width': ((3, 3),),
        'mode': 'mean',
        'constant_values': 0.0,
        'stat_length': ((3, 3),),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, minimum mode
    input_dict = {
        'array': np.random.randn(4, 4).astype(np.float32),
        'pad_width': ((1, 2), (2, 1)),
        'mode': 'minimum',
        'constant_values': 0.0,
        'stat_length': ((2, 2), (2, 2)),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, median mode
    input_dict = {
        'array': np.random.randn(8).astype(np.float32),
        'pad_width': ((2, 2),),
        'mode': 'median',
        'constant_values': 0.0,
        'stat_length': ((4, 4),),
        'end_values': 0.0,
        'reflect_type': 'even'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.pad_2"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.pad_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.pad_2'.")


check_valid('jax.numpy.pad', generated_inputs['jax.numpy.pad_2'], lib="jax", suffix=2)
