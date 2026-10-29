
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class Order(str):
    def __new__(cls, val):
        instance = super().__new__(cls, str(val))
        instance.val = val
        return instance

    def __eq__(self, other):
        return other == self.val or super().__eq__(other)

    def __hash__(self):
        return hash(self.val)

    def lower(self):
        return self

    def upper(self):
        return self

    def __int__(self):
        return int(self.val)

    def __float__(self):
        return float(self.val)

    def __reduce__(self):
        return (Order, (self.val,))


def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis 0, keepdims False, ord 2
    x = np.array([3.0, -4.0, 5.0, -12.0], dtype=np.float32)
    input_dict = {"x": x, "axis": 0, "keepdims": False, "ord": Order(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float64, axis 1, keepdims True, ord 1
    x = np.random.randn(3, 5).astype(np.float64)
    input_dict = {"x": x, "axis": 1, "keepdims": True, "ord": Order(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, negative axis -1, keepdims False, ord 2
    x = np.array([[-1.5, 2.5, 0.0], [4.0, -3.0, 1.2]], dtype=np.float32)
    input_dict = {"x": x, "axis": -1, "keepdims": False, "ord": Order(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with positive values, axis 0, keepdims True, ord 1
    x = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "axis": 0, "keepdims": True, "ord": Order(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis 2, keepdims False, ord 2
    x = np.random.randn(4, 2, 6).astype(np.float64)
    input_dict = {"x": x, "axis": 2, "keepdims": False, "ord": Order(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis 1, keepdims True, ord 1
    x = np.random.randn(2, 4, 3, 2).astype(np.float32)
    input_dict = {"x": x, "axis": 1, "keepdims": True, "ord": Order(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with negative axis -1, keepdims True, ord 2
    x = np.array([-7.0, 24.0], dtype=np.float32)
    input_dict = {"x": x, "axis": -1, "keepdims": True, "ord": Order(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex array, axis 0, keepdims False, ord 2
    x = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
        np.complex64
    )
    input_dict = {"x": x, "axis": 0, "keepdims": False, "ord": Order(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with negative axis -2, keepdims False, ord 1
    x = np.random.uniform(0.5, 5.0, size=(3, 5, 2)).astype(np.float32)
    input_dict = {"x": x, "axis": -2, "keepdims": False, "ord": Order(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, axis 3, keepdims True, ord 2
    x = np.random.randn(2, 1, 3, 4, 2).astype(np.float32)
    input_dict = {"x": x, "axis": 3, "keepdims": True, "ord": Order(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array, axis -2, keepdims True, ord 1
    x = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"x": x, "axis": -2, "keepdims": True, "ord": Order(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.vector_norm_2"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vector_norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vector_norm_2'.")


check_valid('jax.numpy.linalg.vector_norm', generated_inputs['jax.numpy.linalg.vector_norm_2'], lib="jax", suffix=2)
