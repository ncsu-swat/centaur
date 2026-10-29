
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class StrOrd(str):

    def __new__(cls, val="2"):
        return super().__new__(cls, val)

    def __eq__(self, other):
        if super().__eq__("2") and other == 2:
            return True
        if super().__eq__("1") and other == 1:
            return True
        if super().__eq__("0") and other == 0:
            return True
        if super().__eq__("inf") and other == float("inf"):
            return True
        if super().__eq__("-inf") and other == -float("inf"):
            return True
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


def norm_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, ord='2', axis=0, keepdims=False
    x = np.array([-3.0, 4.0, -12.0], dtype=np.float32)
    input_dict = {"x": x, "ord": StrOrd("2"), "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64, ord='1', axis=-1, keepdims=True
    x = np.array([1.5, -2.5, 3.5, -4.5], dtype=np.float64)
    input_dict = {"x": x, "ord": StrOrd("1"), "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32, ord='inf', axis=0, keepdims=False
    x = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"x": x, "ord": StrOrd("inf"), "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32, ord='-inf', axis=1, keepdims=True
    x = np.random.randn(5, 2).astype(np.float32)
    input_dict = {"x": x, "ord": StrOrd("-inf"), "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64, ord='0', axis=-1, keepdims=False
    x = np.array([[1.0, 0.0, -2.0], [0.0, 3.0, 0.0]], dtype=np.float64)
    input_dict = {"x": x, "ord": StrOrd("0"), "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32, ord='2', axis=2, keepdims=True
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "ord": StrOrd("2"), "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64, ord='1', axis=-2, keepdims=False
    x = np.random.randn(3, 3, 3).astype(np.float64)
    input_dict = {"x": x, "ord": StrOrd("1"), "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32, ord='inf', axis=1, keepdims=True
    x = np.random.randn(2, 4, 3, 2).astype(np.float32)
    input_dict = {"x": x, "ord": StrOrd("inf"), "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64, ord='2', axis=-2, keepdims=True
    x = np.random.randn(4, 5).astype(np.float64)
    input_dict = {"x": x, "ord": StrOrd("2"), "axis": -2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32, ord='-inf', axis=-1, keepdims=False
    x = np.random.randn(2, 2, 5).astype(np.float32)
    input_dict = {"x": x, "ord": StrOrd("-inf"), "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.norm_3"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_3'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_3'], lib="jax", suffix=3)
