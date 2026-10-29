
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PadMode(str):

    def __call__(self, row, pad_width, iaxis, kwargs):
        return row

    def __deepcopy__(self, memo):
        return self


def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    input_dict = {
        "array": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "pad_width": [[1, 2]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array
    input_dict = {
        "array": np.ones((4, 4), dtype=np.float32),
        "pad_width": [[1, 1], [2, 2]],
        "mode": PadMode("custom"),
        "constant_values": -1.5,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 array
    input_dict = {
        "array": np.linspace(1.0, 10.0, 6, dtype=np.float64),
        "pad_width": [[2, 2]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 5.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array
    input_dict = {
        "array": np.arange(25, dtype=np.float32).reshape((5, 5)),
        "pad_width": [[1, 1], [1, 1]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array
    input_dict = {
        "array": np.arange(36, dtype=np.int32).reshape((6, 6)),
        "pad_width": [[2, 1], [1, 2]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "odd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array
    input_dict = {
        "array": np.zeros((3, 4, 5), dtype=np.float32),
        "pad_width": [[1, 1], [0, 1], [2, 0]],
        "mode": PadMode("custom"),
        "constant_values": 3.14,
        "stat_length": 2,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array
    input_dict = {
        "array": np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32),
        "pad_width": [[3, 3]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "odd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float64 array
    input_dict = {
        "array": np.ones((4, 5), dtype=np.float64),
        "pad_width": [[2, 2], [1, 1]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": -2.5,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array
    input_dict = {
        "array": np.arange(64, dtype=np.float64).reshape((4, 4, 4)),
        "pad_width": [[1, 1], [1, 1], [1, 1]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with asymmetric padding
    input_dict = {
        "array": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        "pad_width": [[0, 4]],
        "mode": PadMode("custom"),
        "constant_values": 100.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int64 array
    input_dict = {
        "array": np.arange(25, dtype=np.int64).reshape((5, 5)),
        "pad_width": [[1, 0], [0, 2]],
        "mode": PadMode("custom"),
        "constant_values": 0.0,
        "stat_length": 1,
        "end_values": 0.0,
        "reflect_type": "even",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
