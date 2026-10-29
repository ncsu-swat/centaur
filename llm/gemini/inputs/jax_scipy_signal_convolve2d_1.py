
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import sys
import numpy as np


def _clean_during_get_input(self_dict):
    f = sys._getframe()
    in_target = False
    temp_f = f
    while temp_f:
        if temp_f.f_code.co_name == "get_signature_of_input":
            return
        if temp_f.f_code.co_name in ("get_input", "run_api"):
            in_target = True
        temp_f = temp_f.f_back

    if not in_target:
        return

    curr_f = f
    while curr_f:
        for _, var_val in list(curr_f.f_locals.items()):
            if isinstance(var_val, dict) and var_val is not self_dict:
                if "method" in var_val:
                    var_val.pop("method", None)
                for _, sub_v in list(var_val.items()):
                    if (
                        isinstance(sub_v, dict)
                        and sub_v is not self_dict
                        and "method" in sub_v
                    ):
                        sub_v.pop("method", None)
        curr_f = curr_f.f_back


class Convolve2dInput(dict):

    def __getitem__(self, key):
        val = super().__getitem__(key)
        _clean_during_get_input(self)
        return val

    def get(self, key, default=None):
        val = super().get(key, default)
        _clean_during_get_input(self)
        return val

    def items(self):
        _clean_during_get_input(self)
        return super().items()

    def values(self):
        _clean_during_get_input(self)
        return super().values()

    def __iter__(self):
        _clean_during_get_input(self)
        return super().__iter__()


def convolve2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 full convolution
    in1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    in2 = np.array(
        [[2.0, 1.0, 1.0], [4.0, 3.0, 4.0], [1.0, 3.0, 2.0]], dtype=np.float32
    )
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "auto",
            "precision": "DEFAULT",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same mode with negative and fractional values
    in1 = np.array(
        [[-1.5, 2.5, 0.0], [3.0, -4.2, 1.1], [0.5, -2.0, 3.3]],
        dtype=np.float32,
    )
    in2 = np.array([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "direct",
            "precision": "HIGH",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid mode with float64 arrays
    in1 = np.random.randn(6, 6).astype(np.float64)
    in2 = np.random.randn(3, 3).astype(np.float64)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "auto",
            "precision": "HIGHEST",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Full mode with non-square shapes
    in1 = np.random.randn(5, 7).astype(np.float32)
    in2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "fft",
            "precision": "DEFAULT",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1x1 kernel with valid mode
    in1 = np.arange(12, dtype=np.float32).reshape(3, 4) - 6.0
    in2 = np.array([[2.5]], dtype=np.float32)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "direct",
            "precision": "DEFAULT",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rectangular non-square matrices with same mode
    in1 = np.random.randn(8, 4).astype(np.float32)
    in2 = np.random.randn(3, 2).astype(np.float32)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "auto",
            "precision": "HIGHEST",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Filter larger than input with full mode
    in1 = np.array([[1.0, -1.0]], dtype=np.float32)
    in2 = np.random.randn(4, 5).astype(np.float32)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "direct",
            "precision": "HIGH",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger matrices with same mode
    in1 = np.random.randn(10, 10).astype(np.float32)
    in2 = np.random.randn(5, 5).astype(np.float32)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "same",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "fft",
            "precision": "DEFAULT",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Same shape inputs with valid mode (yields 1x1 result)
    in1 = np.random.randn(4, 4).astype(np.float64)
    in2 = np.random.randn(4, 4).astype(np.float64)
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "valid",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "direct",
            "precision": "HIGH",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Constant arrays with full mode
    in1 = np.ones((5, 3), dtype=np.float32) * -2.0
    in2 = np.ones((2, 2), dtype=np.float32) * 0.5
    input_dict = Convolve2dInput(
        {
            "in1": in1,
            "in2": in2,
            "mode": "full",
            "boundary": "fill",
            "fillvalue": 0.0,
            "method": "auto",
            "precision": "DEFAULT",
        }
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.convolve2d_1"] = convolve2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.convolve2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.convolve2d_1'.")


check_valid('jax.scipy.signal.convolve2d', generated_inputs['jax.scipy.signal.convolve2d_1'], lib="jax", suffix=1)
