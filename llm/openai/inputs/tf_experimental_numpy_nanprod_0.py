
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_nanprod_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with NaN values
    a = np.array([[1., 2., np.nan], [4., 5., 6.]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with NaN values
    a = np.array([[[1., 2.], [np.nan, 4.]], [[5., 6.], [7., np.nan]]])
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with NaN values
    a = np.array([1., 2., np.nan, 4., 5.])
    input_dict = {
        "a": a,
        "axis": None,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with all NaN values in one axis
    a = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with negative values
    a = np.array([-1., -2., 3., -4.])
    input_dict = {
        "a": a,
        "axis": None,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with mixed values including NaN
    a = np.array([[1., np.nan, 3.], [4., 5., np.nan]])
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": None,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with multiple NaN values
    a = np.array([[[np.nan, 2.], [3., 4.]], [[5., 6.], [np.nan, 8.]]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with large values
    a = np.array([100., 200., 300., np.nan])
    input_dict = {
        "a": a,
        "axis": None,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with zero values
    a = np.array([[1., 0.], [3., 4.]])
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": None,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with float dtype
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float64,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.nanprod"] = generate_nanprod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.nanprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.nanprod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.nanprod', generated_inputs['tf.experimental.numpy.nanprod'], lib="tf", suffix=0)
