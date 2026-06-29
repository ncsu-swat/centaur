
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_pd_matrix(d, dtype=np.float32):
    A = np.random.randn(d, d).astype(dtype)
    return np.dot(A, A.T) + np.eye(d).astype(dtype) * 0.1

def logpdf_inputs():
    list_of_inputs = []

    # Case 1: d=2, single point
    x = np.array([0.5, -0.5], dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    cov = generate_pd_matrix(2, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 2: d=2, batch of 5
    x = np.random.randn(5, 2).astype(np.float32)
    mean = np.array([1.0, -1.0], dtype=np.float32)
    cov = generate_pd_matrix(2, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 3: d=3, single point (float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cov = generate_pd_matrix(3, dtype=np.float64)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 4: d=3, batch of 10
    x = np.random.randn(10, 3).astype(np.float32)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    cov = generate_pd_matrix(3, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 5: d=4, single point
    x = np.random.randn(4).astype(np.float32)
    mean = np.zeros(4, dtype=np.float32)
    cov = generate_pd_matrix(4, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 6: d=4, batch of 3
    x = np.random.randn(3, 4).astype(np.float32)
    mean = np.ones(4, dtype=np.float32)
    cov = generate_pd_matrix(4, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 7: d=5, single point (float64)
    x = np.random.randn(5).astype(np.float64)
    mean = np.zeros(5, dtype=np.float64)
    cov = generate_pd_matrix(5, dtype=np.float64)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 8: d=5, batch of 2 (float64)
    x = np.random.randn(2, 5).astype(np.float64)
    mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    cov = generate_pd_matrix(5, dtype=np.float64)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 9: d=6, single point
    x = np.random.randn(6).astype(np.float32)
    mean = np.zeros(6, dtype=np.float32)
    cov = generate_pd_matrix(6, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    # Case 10: d=6, batch of 15
    x = np.random.randn(15, 6).astype(np.float32)
    mean = np.ones(6, dtype=np.float32)
    cov = generate_pd_matrix(6, dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "mean": mean,
        "cov": cov,
        "allow_singular": None
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.multivariate_normal.logpdf"] = logpdf_inputs()

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
