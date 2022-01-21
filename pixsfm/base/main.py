interpolation_default_conf = {
    'nodes': [[0.0, 0.0]],
    'mode': 'BICUBIC',
    'l2_normalize': True,
    'ncc_normalize': False,
    "use_float_simd": False
}

solver_default_conf = {
    'function_tolerance': 0.0,
    'gradient_tolerance': 0.0,
    'parameter_tolerance': 0.0,
    'minimizer_progress_to_stdout': False,
    'max_num_iterations': 100,
    'max_linear_solver_iterations': 200,
    'max_num_consecutive_invalid_steps': 10,
    'max_consecutive_nonmonotonic_steps': 10,
    'use_inner_iterations': False,
    'use_nonmonotonic_steps': False,
    'update_state_every_iteration': False,
    'num_threads': -1,
}
