
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ndim_inputs():
    list_of_inputs = []

    # 1. Scalar boolean
    a = np.bool_(True)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 2. 0D boolean array
    a = np.array(True, dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 3. 1D boolean array
    a = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 4. 2D boolean array
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 5. 3D boolean array
    a = np.ones((2, 2, 2), dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 6. 4D boolean array
    a = np.zeros((1, 3, 2, 1), dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 7. 5D boolean array
    a = np.ones((2, 1, 2, 1, 2), dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 8. Large 2D boolean array
    a = np.random.choice([True, False], size=(10, 10)).astype(np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 9. 1D boolean array with single element
    a = np.array([False], dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # 10. 3D boolean array with shape containing 0
    a = np.empty((0, 2, 3), dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs

generated_inputs["jax.numpy.ndim_4"] = ndim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ndim_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ndim_4'.")


check_valid('jax.numpy.ndim', generated_inputs['jax.numpy.ndim_4'], lib="jax", suffix=4)
