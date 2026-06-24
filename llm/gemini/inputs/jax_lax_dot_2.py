
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Patch jax.lax.Precision to support comparison for numpy min/max
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if hasattr(other, 'value') else False
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if hasattr(other, 'value') else False
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if hasattr(other, 'value') else False
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if hasattr(other, 'value') else False

# Patch JAX's canonicalize_sharding to intercept empty tuple out_sharding and replace with a real Sharding
import jax._src.lax.lax as lax_module
if hasattr(lax_module, 'canonicalize_sharding'):
    orig_lax_canonicalize = lax_module.canonicalize_sharding
    lax_module.canonicalize_sharding = lambda sharding, *args, **kwargs: (
        jax.devices()[0].to_sharding() if (isinstance(sharding, tuple) and len(sharding) == 0)
        else orig_lax_canonicalize(sharding, *args, **kwargs)
    )

import jax._src.sharding_impls as sharding_impls
if hasattr(sharding_impls, 'canonicalize_sharding'):
    orig_impl_canonicalize = sharding_impls.canonicalize_sharding
    sharding_impls.canonicalize_sharding = lambda sharding, *args, **kwargs: (
        jax.devices()[0].to_sharding() if (isinstance(sharding, tuple) and len(sharding) == 0)
        else orig_impl_canonicalize(sharding, *args, **kwargs)
    )

def jax_lax_dot_inputs():
    list_of_inputs = []

    # 1. Standard float32 contraction
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. High precision contraction
    lhs = np.random.randn(3, 2, 5).astype(np.float32)
    rhs = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. Highest precision float64 contraction
    lhs = np.random.randn(4, 4, 2).astype(np.float64)
    rhs = np.random.randn(4, 2, 3).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. Alternative shapes
    lhs = np.random.randn(5, 2, 3).astype(np.float32)
    rhs = np.random.randn(5, 3, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. Mixed dimensions (4D x 3D)
    lhs = np.random.randn(2, 3, 4, 5).astype(np.float32)
    rhs = np.random.randn(2, 5, 6).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((3,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. Mixed dimensions with high precision
    lhs = np.random.randn(3, 2, 4, 3).astype(np.float32)
    rhs = np.random.randn(3, 3, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((3,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. Float16 contraction with negative values
    lhs = np.random.uniform(-2.0, 2.0, (4, 1, 3, 2)).astype(np.float16)
    rhs = np.random.uniform(-2.0, 2.0, (4, 2, 1)).astype(np.float16)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((3,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float16),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. Small 3D tensors
    lhs = np.random.randn(2, 2, 2).astype(np.float32)
    rhs = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Large float64 contraction
    lhs = np.random.randn(3, 5, 3).astype(np.float64)
    rhs = np.random.randn(3, 3, 6).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. Float16 standard batch contraction
    lhs = np.random.randn(2, 4, 6).astype(np.float16)
    rhs = np.random.randn(2, 6, 2).astype(np.float16)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float16),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dot_2"] = jax_lax_dot_inputs()

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
