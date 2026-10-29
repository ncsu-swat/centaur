
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class ConvolveDict(dict):

    def __init__(self, *args, method="auto", precision=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.precision = precision

    def __getitem__(self, key):
        if key == "method":
            return self.method
        if key == "precision":
            return self.precision
        return super().__getitem__(key)

    def __contains__(self, key):
        if key in ("method", "precision"):
            return False
        return super().__contains__(key)


def convolve2d_inputs():
    list_of_inputs = []

    # Input 1: Basic full convolution with float32
    in1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    in2 = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="full",
        boundary="fill",
        fillvalue=0.0,
        method="auto",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Same mode with 5x5 and 3x3 arrays
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
        method="direct",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Valid mode with float64 arrays
    in1 = np.random.randn(6, 4).astype(np.float64)
    in2 = np.random.randn(3, 2).astype(np.float64)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="valid",
        boundary="fill",
        fillvalue=0.0,
        method="fft",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values and rectangular dimensions, full mode
    in1 = np.array(
        [[-1.5, 2.0, -3.2], [4.1, -5.0, 0.5], [-2.2, 1.1, -4.4], [0.0, -1.0, 2.0]],
        dtype=np.float32,
    )
    in2 = np.array([[-0.5, 1.5], [2.0, -1.0]], dtype=np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="full",
        boundary="fill",
        fillvalue=0.0,
        method="direct",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Same mode with 7x5 array and 4x3 kernel
    in1 = np.random.randn(7, 5).astype(np.float32)
    in2 = np.random.randn(4, 3).astype(np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
        method="auto",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Valid mode with larger 8x8 input and 2x2 kernel
    in1 = np.random.randn(8, 8).astype(np.float64)
    in2 = np.random.randn(2, 2).astype(np.float64)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="valid",
        boundary="fill",
        fillvalue=0.0,
        method="direct",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Minimal 1x1 inputs
    in1 = np.array([[3.5]], dtype=np.float32)
    in2 = np.array([[-2.0]], dtype=np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="full",
        boundary="fill",
        fillvalue=0.0,
        method="fft",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Valid mode with identical shapes
    in1 = np.random.randn(5, 7).astype(np.float32)
    in2 = np.random.randn(5, 7).astype(np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="valid",
        boundary="fill",
        fillvalue=0.0,
        method="auto",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Same mode with 10x12 matrix and 3x4 kernel, float64
    in1 = np.random.randn(10, 12).astype(np.float64)
    in2 = np.random.randn(3, 4).astype(np.float64)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
        method="direct",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid mode with 1x1 kernel
    in1 = np.random.randn(6, 6).astype(np.float32)
    in2 = np.array([[0.5]], dtype=np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="valid",
        boundary="fill",
        fillvalue=0.0,
        method="fft",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Full mode with 4x4 matrix and 3x3 filter
    in1 = np.random.randn(4, 4).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = ConvolveDict(
        in1=in1,
        in2=in2,
        mode="full",
        boundary="fill",
        fillvalue=0.0,
        method="auto",
        precision=(),
    )
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.convolve2d_2"] = convolve2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.convolve2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.convolve2d_2'.")


check_valid('jax.scipy.signal.convolve2d', generated_inputs['jax.scipy.signal.convolve2d_2'], lib="jax", suffix=2)
