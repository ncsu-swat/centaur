
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class AxisStr(str):

    def __new__(cls, val):
        obj = super().__new__(cls, val)
        obj._val = int("".join(val))
        return obj

    def __index__(self):
        return getattr(self, "_val", int(float("".join(self))))


def one_hot_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with float32 dtype
    x = np.array([0, 1, 2], dtype=np.int32)
    num_classes = 3
    dtype = np.dtype(np.float32)
    axis = AxisStr("-1")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 2: 1D array with negative and out-of-range indices
    x = np.array([-1, 0, 5, 2], dtype=np.int32)
    num_classes = 4
    dtype = np.dtype(np.float32)
    axis = AxisStr("0")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 3: 2D array with float64 dtype
    x = np.array([[0, 1], [2, 0]], dtype=np.int64)
    num_classes = 3
    dtype = np.dtype(np.float64)
    axis = AxisStr("-1")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 4: 0D scalar tensor
    x = np.array(2, dtype=np.int32)
    num_classes = 5
    dtype = np.dtype(np.float32)
    axis = AxisStr("0")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 5: 2D array with axis="1" and int32 dtype
    x = np.array([[-2, 1, 3], [0, -1, 2]], dtype=np.int32)
    num_classes = 4
    dtype = np.dtype(np.int32)
    axis = AxisStr("1")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 6: 3D array
    x = np.array(
        [[[0, 1], [2, 3]], [[1, 0], [3, 2]]],
        dtype=np.int32,
    )
    num_classes = 5
    dtype = np.dtype(np.float32)
    axis = AxisStr("-1")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 7: 1D array with bool output dtype
    x = np.array([0, 1, 0, 1], dtype=np.uint8)
    num_classes = 2
    dtype = np.dtype(np.bool_)
    axis = AxisStr("0")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 8: 3D array with float16 dtype and negative axis
    x = np.zeros((2, 2, 2), dtype=np.int32)
    num_classes = 10
    dtype = np.dtype(np.float16)
    axis = AxisStr("-2")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 9: Empty 1D array
    x = np.array([], dtype=np.int32)
    num_classes = 3
    dtype = np.dtype(np.float32)
    axis = AxisStr("-1")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 10: 4D array with axis="2"
    x = np.ones((2, 1, 2, 3), dtype=np.int64)
    num_classes = 3
    dtype = np.dtype(np.float64)
    axis = AxisStr("2")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 11: All indices strictly outside range
    x = np.array([10, 20, 30], dtype=np.int32)
    num_classes = 5
    dtype = np.dtype(np.uint8)
    axis = AxisStr("-1")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    # Input 12: Large num_classes
    x = np.array([[0, 99], [50, 1]], dtype=np.int32)
    num_classes = 100
    dtype = np.dtype(np.float32)
    axis = AxisStr("0")
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "num_classes": num_classes, "dtype": dtype, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.nn.one_hot_2"] = one_hot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.one_hot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.one_hot_2'.")


check_valid('jax.nn.one_hot', generated_inputs['jax.nn.one_hot_2'], lib="jax", suffix=2)
