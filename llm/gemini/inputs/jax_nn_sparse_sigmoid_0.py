
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sparse_sigmoid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array spanning the threshold boundaries (-1 and 1)
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array with random standard normal values
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar (0D) float32 array
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array
    x = np.random.uniform(-2.0, 2.0, size=(2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: float64 array for higher precision
    x = np.array([-1.5, -0.2, 0.2, 1.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: float16 array for lower precision
    x = np.array([-0.9, -0.1, 0.1, 0.9], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D float32 array representing batch of images (B, H, W, C)
    x = np.random.randn(2, 8, 8, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: All values below the lower bound (-1) to test saturation at 0
    x = np.array([-5.0, -10.0, -100.0, -1.1], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: All values above the upper bound (1) to test saturation at 1
    x = np.array([1.1, 5.0, 10.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Empty/zero elements in complex shape, float32
    x = np.zeros((1, 5, 1), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.sparse_sigmoid"] = sparse_sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.sparse_sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.sparse_sigmoid'.")


check_valid('jax.nn.sparse_sigmoid', generated_inputs['jax.nn.sparse_sigmoid'], lib="jax", suffix=0)
