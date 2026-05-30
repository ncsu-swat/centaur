
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Save original pad and monkey-patch it to filter out irrelevant kwargs for each mode
_original_pad = jnp.pad

def patched_pad(array, pad_width, mode="constant", **kwargs):
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
    if isinstance(mode, str) and mode in allowed_kwargs:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs[mode] and v is not None}
    else:
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return _original_pad(array, pad_width, mode=mode, **filtered_kwargs)

jax.numpy.pad = patched_pad
jnp.pad = patched_pad

def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D array, constant mode
    list_of_inputs.append({
        "array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "pad_width": [[1, 2]],
        "mode": "constant",
        "constant_values": 0.0,
        "stat_length": None,
        "end_values": None,
        "reflect_type": None
    })

    # Input 2: 2D array, constant mode with non-zero float constant
    list_of_inputs.append({
        "array": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pad_width": [[1, 1], [2, 2]],
        "mode": "constant",
        "constant_values": -1.5,
        "stat_length": None,
        "end_values": None,
        "reflect_type": None
    })

    # Input 3: 1D array, edge mode
    list_of_inputs.append({
        "array": np.array([5.0, 10.0, 15.0], dtype=np.float32),
        "pad_width": [[2, 2]],
        "mode": "edge",
        "constant_values": None,
        "stat_length": None,
        "end_values": None,
        "reflect_type": None
    })

    # Input 4: 2D array, linear_ramp mode with float end_values
    list_of_inputs.append({
        "array": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "pad_width": [[1, 1], [2, 2]],
        "mode": "linear_ramp",
        "constant_values": None,
        "stat_length": None,
        "end_values": 0.0,
        "reflect_type": None
    })

    # Input 5: 1D array, maximum mode with stat_length
    list_of_inputs.append({
        "array": np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32),
        "pad_width": [[2, 2]],
        "mode": "maximum",
        "constant_values": None,
        "stat_length": 2,
        "end_values": None,
        "reflect_type": None
    })

    # Input 6: 2D array, mean mode with stat_length
    list_of_inputs.append({
        "array": np.arange(9, dtype=np.float32).reshape(3, 3),
        "pad_width": [[1, 1], [1, 1]],
        "mode": "mean",
        "constant_values": None,
        "stat_length": 1,
        "end_values": None,
        "reflect_type": None
    })

    # Input 7: 1D array, reflect mode with reflect_type
    list_of_inputs.append({
        "array": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "pad_width": [[2, 2]],
        "mode": "reflect",
        "constant_values": None,
        "stat_length": None,
        "end_values": None,
        "reflect_type": "even"
    })

    # Input 8: 2D array, symmetric mode with reflect_type
    list_of_inputs.append({
        "array": np.arange(4, dtype=np.float32).reshape(2, 2),
        "pad_width": [[1, 1], [1, 1]],
        "mode": "symmetric",
        "constant_values": None,
        "stat_length": None,
        "end_values": None,
        "reflect_type": "odd"
    })

    # Input 9: 3D array, wrap mode
    list_of_inputs.append({
        "array": np.ones((2, 2, 2), dtype=np.float32),
        "pad_width": [[1, 1], [1, 1], [1, 1]],
        "mode": "wrap",
        "constant_values": None,
        "stat_length": None,
        "end_values": None,
        "reflect_type": None
    })

    # Input 10: 1D array, minimum mode with stat_length
    list_of_inputs.append({
        "array": np.array([10.0, 2.0, 8.0, 4.0], dtype=np.float32),
        "pad_width": [[1, 1]],
        "mode": "minimum",
        "constant_values": None,
        "stat_length": 2,
        "end_values": None,
        "reflect_type": None
    })

    return list_of_inputs

generated_inputs["jax.numpy.pad_3"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.pad_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.pad_3'.")


check_valid('jax.numpy.pad', generated_inputs['jax.numpy.pad_3'], lib="jax", suffix=3)
