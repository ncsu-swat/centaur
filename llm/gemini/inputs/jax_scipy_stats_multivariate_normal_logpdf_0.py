
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multivariate_normal_logpdf_inputs():
    list_of_inputs = []

    def make_spd_cov(dim, dtype):
        A = np.random.randn(dim, dim).astype(dtype)
        return A @ A.T + np.eye(dim, dtype=dtype) * 0.5

    np.random.seed(42)

    # Input 1: 1D vector (dim=1), float32
    x = np.array([0.5], dtype=np.float32)
    mean = np.array([0.0], dtype=np.float32)
    cov = np.array([[1.0]], dtype=np.float32)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 2: 1D vector (dim=2), float64, standard normal
    x = np.array([1.0, -1.0], dtype=np.float64)
    mean = np.array([0.0, 0.0], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 3: 1D vector (dim=3) with non-zero mean and non-diagonal cov, float32
    x = np.array([-1.2, 0.8, 2.3], dtype=np.float32)
    mean = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    cov = make_spd_cov(3, np.float32)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 4: Batched x of shape (5, 2), float32
    x = np.random.randn(5, 2).astype(np.float32)
    mean = np.zeros(2, dtype=np.float32)
    cov = make_spd_cov(2, np.float32)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 5: Batched x of shape (10, 4), float64
    x = np.random.randn(10, 4).astype(np.float64)
    mean = np.array([1.0, -2.0, 0.5, -0.5], dtype=np.float64)
    cov = make_spd_cov(4, np.float64)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 6: Multi-dimensional batch of shape (3, 4, 2), float32
    x = np.random.randn(3, 4, 2).astype(np.float32)
    mean = np.array([0.1, -0.2], dtype=np.float32)
    cov = np.array([[2.0, 0.3], [0.3, 1.5]], dtype=np.float32)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 7: Higher dimension (dim=6), single vector, float64
    x = np.random.randn(6).astype(np.float64)
    mean = np.random.randn(6).astype(np.float64)
    cov = make_spd_cov(6, np.float64)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 8: Batch of 1 sample (1, 3), float32
    x = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
    mean = np.zeros(3, dtype=np.float32)
    cov = np.eye(3, dtype=np.float32) * 2.0
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 9: Large batch (50, 2), float32
    x = np.random.randn(50, 2).astype(np.float32)
    mean = np.array([-1.0, 1.0], dtype=np.float32)
    cov = make_spd_cov(2, np.float32)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 10: 1D vector (dim=5) evaluated at distribution mean, float64
    mean = np.array([0.5, -1.2, 3.4, -2.1, 0.0], dtype=np.float64)
    x = mean.copy()
    cov = make_spd_cov(5, np.float64)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    # Input 11: Multi-dimensional batch (2, 3, 2, 2), float64
    x = np.random.randn(2, 3, 2, 2).astype(np.float64)
    mean = np.zeros(2, dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    allow_singular = None
    list_of_inputs.append(
        copy.deepcopy(
            {"x": x, "mean": mean, "cov": cov, "allow_singular": allow_singular}
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.multivariate_normal.logpdf"] = (
    multivariate_normal_logpdf_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.multivariate_normal.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.multivariate_normal.logpdf'.")


check_valid('jax.scipy.stats.multivariate_normal.logpdf', generated_inputs['jax.scipy.stats.multivariate_normal.logpdf'], lib="jax", suffix=0)
