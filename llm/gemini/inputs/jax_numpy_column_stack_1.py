
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class SignatureCompatDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consumed_dtype = False

    def __getitem__(self, key):
        if key == "dtype":
            if not self._consumed_dtype:
                self._consumed_dtype = True
                return None
            raise KeyError(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        if key == "dtype":
            return not self._consumed_dtype
        return super().__contains__(key)

    def get(self, key, default=None):
        if key == "dtype":
            if not self._consumed_dtype:
                self._consumed_dtype = True
                return None
            return default
        return super().get(key, default)


def column_stack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    tup = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative values
    tup = np.array([-10, 0, 10, -25], dtype=np.int32)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    tup = np.random.randn(4, 3).astype(np.float64)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array
    tup = np.ones((5, 2), dtype=np.float32)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int16 array
    tup = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array
    tup = np.random.randn(3, 4, 2).astype(np.float32)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D boolean array
    tup = np.array([[[[True, False], [False, True]]]], dtype=np.bool_)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex64 array
    tup = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(
        np.complex64
    )
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D uint8 array
    tup = np.array([42, 99, 0, 255], dtype=np.uint8)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D empty array
    tup = np.empty((0, 2), dtype=np.float32)
    input_dict = SignatureCompatDict({"tup": tup})
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.column_stack_1"] = column_stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.column_stack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.column_stack_1'.")


check_valid('jax.numpy.column_stack', generated_inputs['jax.numpy.column_stack_1'], lib="jax", suffix=1)
