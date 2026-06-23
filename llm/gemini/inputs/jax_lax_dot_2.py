
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.lax as lax

# Monkey patch lax.Precision to make it comparable for np.min/np.max in the test runner
lax.Precision.__lt__ = lambda self, other: self.value < other.value
lax.Precision.__le__ = lambda self, other: self.value <= other.value
lax.Precision.__gt__ = lambda self, other: self.value > other.value
lax.Precision.__ge__ = lambda self, other: self.value >= other.value

def dot_inputs():
    list_of_inputs = []

    # Use identical shapes, types, and homogeneous dimension_numbers/precision
    # to maximize compilation caching and prevent timeout.
    shape_lhs = (2, 3, 4)
    shape_rhs = (2, 3, 5)
    dimension_numbers = (((1,), (1,)), ((0,), (0,)))
    precision = (lax.Precision.DEFAULT, lax.Precision.DEFAULT)
    preferred_element_type = np.float32
    out_sharding = None

    # Generate 10 variations with the exact same shapes to ensure JAX compiles only once
    for i in range(10):
        lhs = np.random.randn(*shape_lhs).astype(np.float32)
        rhs = np.random.randn(*shape_rhs).astype(np.float32)
        if i % 2 == 0:
            lhs = -lhs
            rhs = -rhs
        
        list_of_inputs.append({
            'lhs': lhs,
            'rhs': rhs,
            'dimension_numbers': dimension_numbers,
            'precision': precision,
            'preferred_element_type': preferred_element_type,
            'out_sharding': out_sharding
        })

    return list_of_inputs

generated_inputs["jax.lax.dot_2"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dot_2'.")


check_valid('jax.lax.dot', generated_inputs['jax.lax.dot_2'], lib="jax", suffix=2)
