
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def piecewise_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with 2 conditions and constant functions
    x1 = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    condlist1 = np.array([x1 < 0, x1 >= 0])
    funclist1 = [0, 1]
    input_dict1 = {"x": x1, "condlist": condlist1, "funclist": funclist1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D array with 2 conditions and a default constant function
    x2 = np.array([-5, -1, 0, 1, 5], dtype=np.int32)
    condlist2 = np.array([x2 < -2, x2 > 2])
    funclist2 = [-1, 1, 0]
    input_dict2 = {"x": x2, "condlist": condlist2, "funclist": funclist2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D array with 2 conditions and constant values
    x3 = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    condlist3 = np.array([x3 < 0, x3 >= 0])
    funclist3 = [10, 20]
    input_dict3 = {"x": x3, "condlist": condlist3, "funclist": funclist3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float32 array with float constant values
    x4 = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    condlist4 = np.array([x4 < 0, x4 >= 0])
    funclist4 = [0.0, 1.0]
    input_dict4 = {"x": x4, "condlist": condlist4, "funclist": funclist4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float32 array with distinct float constants
    x5 = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    condlist5 = np.array([x5 < 0.0, x5 >= 0.0])
    funclist5 = [-10.0, 10.0]
    input_dict5 = {"x": x5, "condlist": condlist5, "funclist": funclist5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 3D array with extreme float constants
    x6 = np.random.randn(2, 3, 4).astype(np.float32)
    condlist6 = np.array([x6 < 0.0, x6 >= 0.0])
    funclist6 = [-100.0, 100.0]
    input_dict6 = {"x": x6, "condlist": condlist6, "funclist": funclist6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 1D array with 3 conditions and 4 constant functions
    x7 = np.array([-10, -5, 0, 5, 10], dtype=np.float32)
    condlist7 = np.array([x7 < -5, (x7 >= -5) & (x7 < 5), x7 >= 5])
    funclist7 = [1.0, 2.0, 3.0, 4.0]
    input_dict7 = {"x": x7, "condlist": condlist7, "funclist": funclist7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Float64 1D array with 2 conditions and float constants
    x8 = np.linspace(-1, 1, 10, dtype=np.float64)
    condlist8 = np.array([x8 < 0.0, x8 >= 0.0])
    funclist8 = [-5.5, 5.5]
    input_dict8 = {"x": x8, "condlist": condlist8, "funclist": funclist8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 4D array with 2 conditions and 3 constant functions
    x9 = np.random.uniform(-5, 5, (2, 2, 2, 2)).astype(np.float32)
    condlist9 = np.array([x9 < -1.0, x9 > 1.0])
    funclist9 = [0.1, 0.2, 0.3]
    input_dict9 = {"x": x9, "condlist": condlist9, "funclist": funclist9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Int64 1D array with large constant integers
    x10 = np.array([-100, 0, 100], dtype=np.int64)
    condlist10 = np.array([x10 < 0, x10 >= 0])
    funclist10 = [-50, 50]
    input_dict10 = {"x": x10, "condlist": condlist10, "funclist": funclist10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["jax.numpy.piecewise_1"] = piecewise_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.piecewise_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.piecewise_1'.")


check_valid('jax.numpy.piecewise', generated_inputs['jax.numpy.piecewise_1'], lib="jax", suffix=1)
