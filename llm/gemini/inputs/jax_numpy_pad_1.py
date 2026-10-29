
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class CustomPad(str):

    def __call__(self, row, pad_width, iaxis, kwargs):
        return row


def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, pad_width=1
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "pad_width": 1,
        "mode": CustomPad("custom_pad_1"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, pad_width=2
    input_dict = {
        "array": np.arange(12, dtype=np.float32).reshape(3, 4),
        "pad_width": 2,
        "mode": CustomPad("custom_pad_2"),
        "constant_values": -1.5,
        "stat_length": 2,
        "end_values": 5.0,
        "reflect_type": "odd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32, pad_width=3
    input_dict = {
        "array": np.array([10, 20, 30, 40, 50], dtype=np.int32),
        "pad_width": 3,
        "mode": CustomPad("custom_pad_3"),
        "constant_values": 100.0,
        "stat_length": 1,
        "end_values": -10.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64, pad_width=1
    input_dict = {
        "array": np.arange(9, dtype=np.int64).reshape(3, 3),
        "pad_width": 1,
        "mode": CustomPad("custom_pad_4"),
        "constant_values": 42.0,
        "stat_length": 3,
        "end_values": 1.0,
        "reflect_type": "odd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32, pad_width=1
    input_dict = {
        "array": np.ones((2, 3, 2), dtype=np.float32),
        "pad_width": 1,
        "mode": CustomPad("custom_pad_5"),
        "constant_values": 3.14,
        "stat_length": 1,
        "end_values": 2.71,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64, pad_width=2
    input_dict = {
        "array": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        "pad_width": 2,
        "mode": CustomPad("custom_pad_6"),
        "constant_values": -0.5,
        "stat_length": 2,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64, pad_width=1
    input_dict = {
        "array": np.eye(4, dtype=np.float64),
        "pad_width": 1,
        "mode": CustomPad("custom_pad_7"),
        "constant_values": 1.0,
        "stat_length": 1,
        "end_values": -1.0,
        "reflect_type": "odd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int16, pad_width=0
    input_dict = {
        "array": np.array([5, 10, 15], dtype=np.int16),
        "pad_width": 0,
        "mode": CustomPad("custom_pad_8"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32, pad_width=1
    input_dict = {
        "array": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "pad_width": 1,
        "mode": CustomPad("custom_pad_9"),
        "constant_values": 7.0,
        "stat_length": 1,
        "end_values": 14.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32, pad_width=3
    input_dict = {
        "array": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "pad_width": 3,
        "mode": CustomPad("custom_pad_10"),
        "constant_values": -99.0,
        "stat_length": 2,
        "end_values": 99.0,
        "reflect_type": "odd",
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
