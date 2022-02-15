// Ceres Solver Python Bindings
// Copyright Nikolaus Mitchell. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: nikolausmitchell@gmail.com (Nikolaus Mitchell)
// Edited by: philipp.lindenberger@math.ethz.ch (Philipp Lindenberger)
#include <ceres/ceres.h>
#include <pybind11/pybind11.h>

// // If the library is built by linking to ceres then we need to
// // access the defined compiler options(USE_SUITESPARSE,threading model
// // ...)
// #ifdef CERES_IS_LINKED
// #include<ceres/internal/port.h>
// #endif
//#include <pybind11/exception.h>
namespace py = pybind11;

#include <chrono>
#include <thread>

#include "_pixsfm/src/helpers.h"
#include "util/src/log_exceptions.h"

void init_solver(py::module& m) {
  // The main Solve function
  m.def("solve",
        overload_cast_<const ceres::Solver::Options&, ceres::Problem*,
                       ceres::Solver::Summary*>()(&ceres::Solve),
        py::call_guard<py::gil_scoped_release>());

  using s_options = ceres::Solver::Options;
  auto so =
      py::class_<ceres::Solver::Options>(m, "SolverOptions")
          .def(py::init<>())
          .def("IsValid", &s_options::IsValid)
          .def_readwrite("callbacks", &s_options::callbacks)
          .def_readwrite("minimizer_type", &s_options::minimizer_type)
          .def_readwrite("line_search_direction_type",
                         &s_options::line_search_direction_type)
          .def_readwrite("line_search_type", &s_options::line_search_type)
          .def_readwrite("nonlinear_conjugate_gradient_type",
                         &s_options::nonlinear_conjugate_gradient_type)
          .def_readwrite("max_lbfgs_rank", &s_options::max_lbfgs_rank)
          .def_readwrite("use_approximate_eigenvalue_bfgs_scaling",
                         &s_options::use_approximate_eigenvalue_bfgs_scaling)
          .def_readwrite("line_search_interpolation_type",
                         &s_options::line_search_interpolation_type)
          .def_readwrite("min_line_search_step_size",
                         &s_options::min_line_search_step_size)
          .def_readwrite("line_search_sufficient_function_decrease",
                         &s_options::line_search_sufficient_function_decrease)
          .def_readwrite("max_line_search_step_contraction",
                         &s_options::max_line_search_step_contraction)
          .def_readwrite("min_line_search_step_contraction",
                         &s_options::min_line_search_step_contraction)
          .def_readwrite("max_num_line_search_step_size_iterations",
                         &s_options::max_num_line_search_step_size_iterations)
          .def_readwrite("max_num_line_search_direction_restarts",
                         &s_options::max_num_line_search_direction_restarts)
          .def_readwrite("line_search_sufficient_curvature_decrease",
                         &s_options::line_search_sufficient_curvature_decrease)
          .def_readwrite("max_line_search_step_expansion",
                         &s_options::max_line_search_step_expansion)
          .def_readwrite("trust_region_strategy_type",
                         &s_options::trust_region_strategy_type)
          .def_readwrite("dogleg_type", &s_options::dogleg_type)
          .def_readwrite("use_nonmonotonic_steps",
                         &s_options::use_nonmonotonic_steps)
          .def_readwrite("max_consecutive_nonmonotonic_steps",
                         &s_options::max_consecutive_nonmonotonic_steps)
          .def_readwrite("max_num_iterations", &s_options::max_num_iterations)
          .def_readwrite("max_solver_time_in_seconds",
                         &s_options::max_solver_time_in_seconds)
          .def_property(
              "num_threads",
              [](const s_options& self) { return self.num_threads; },
              [](s_options& self, int n_threads) {
                self.num_threads = n_threads;
#if CERES_VERSION_MAJOR < 2
                self.num_linear_solver_threads = n_threads;
#endif  // CERES_VERSION_MAJOR
              })
          .def_readwrite("initial_trust_region_radius",
                         &s_options::initial_trust_region_radius)
          .def_readwrite("max_trust_region_radius",
                         &s_options::max_trust_region_radius)
          .def_readwrite("min_trust_region_radius",
                         &s_options::min_trust_region_radius)
          .def_readwrite("min_relative_decrease",
                         &s_options::min_relative_decrease)
          .def_readwrite("min_lm_diagonal", &s_options::min_lm_diagonal)
          .def_readwrite("max_lm_diagonal", &s_options::max_lm_diagonal)
          .def_readwrite("max_num_consecutive_invalid_steps",
                         &s_options::max_num_consecutive_invalid_steps)
          .def_readwrite("function_tolerance", &s_options::function_tolerance)
          .def_readwrite("gradient_tolerance", &s_options::gradient_tolerance)
          .def_readwrite("parameter_tolerance", &s_options::parameter_tolerance)
          .def_readwrite("linear_solver_type", &s_options::linear_solver_type)
          .def_readwrite("preconditioner_type", &s_options::preconditioner_type)
          .def_readwrite("visibility_clustering_type",
                         &s_options::visibility_clustering_type)
          .def_readwrite("dense_linear_algebra_library_type",
                         &s_options::dense_linear_algebra_library_type)
          .def_readwrite("sparse_linear_algebra_library_type",
                         &s_options::sparse_linear_algebra_library_type)
          // .def_readwrite("num_linear_solver_threads",
          // &s_options::num_linear_solver_threads)
          .def_readwrite("use_explicit_schur_complement",
                         &s_options::use_explicit_schur_complement)
          .def_readwrite("use_postordering", &s_options::use_postordering)
          .def_readwrite("dynamic_sparsity", &s_options::dynamic_sparsity)
          .def_readwrite("use_inner_iterations",
                         &s_options::use_inner_iterations)
          .def_readwrite("inner_iteration_tolerance",
                         &s_options::inner_iteration_tolerance)
          .def_readwrite("min_linear_solver_iterations",
                         &s_options::min_linear_solver_iterations)
          .def_readwrite("max_linear_solver_iterations",
                         &s_options::max_linear_solver_iterations)
          .def_readwrite("eta", &s_options::eta)
          .def_readwrite("jacobi_scaling", &s_options::jacobi_scaling)
          .def_readwrite("logging_type", &s_options::logging_type)
          .def_readwrite("minimizer_progress_to_stdout",
                         &s_options::minimizer_progress_to_stdout)
          .def_readwrite("trust_region_problem_dump_directory",
                         &s_options::trust_region_problem_dump_directory)
          .def_readwrite("trust_region_problem_dump_format_type",
                         &s_options::trust_region_problem_dump_format_type)
          .def_readwrite("check_gradients", &s_options::check_gradients)
          .def_readwrite("gradient_check_relative_precision",
                         &s_options::gradient_check_relative_precision)
          .def_readwrite(
              "gradient_check_numeric_derivative_relative_step_size",
              &s_options::gradient_check_numeric_derivative_relative_step_size)
          .def_readwrite("update_state_every_iteration",
                         &s_options::update_state_every_iteration);
  make_dataclass(so);

  using s_summary = ceres::Solver::Summary;
  auto summary =
      py::class_<ceres::Solver::Summary>(m, "SolverSummary")
          .def(py::init<>())
          .def("BriefReport", &s_summary::BriefReport)
          .def("FullReport", &s_summary::FullReport)
          .def("IsSolutionUsable", &s_summary::IsSolutionUsable)
          .def_readwrite("minimizer_type", &s_summary::minimizer_type)
          .def_readwrite("termination_type", &s_summary::termination_type)
          .def_readwrite("message", &s_summary::message)
          .def_readwrite("initial_cost", &s_summary::initial_cost)
          .def_readwrite("final_cost", &s_summary::final_cost)
          .def_readwrite("fixed_cost", &s_summary::fixed_cost)
          .def_readwrite("num_successful_steps",
                         &s_summary::num_successful_steps)
          .def_readwrite("num_unsuccessful_steps",
                         &s_summary::num_unsuccessful_steps)
          .def_readwrite("num_inner_iteration_steps",
                         &s_summary::num_inner_iteration_steps)
          .def_readwrite("num_line_search_steps",
                         &s_summary::num_line_search_steps)
          .def_readwrite("preprocessor_time_in_seconds",
                         &s_summary::preprocessor_time_in_seconds)
          .def_readwrite("minimizer_time_in_seconds",
                         &s_summary::minimizer_time_in_seconds)
          .def_readwrite("postprocessor_time_in_seconds",
                         &s_summary::postprocessor_time_in_seconds)
          .def_readwrite("total_time_in_seconds",
                         &s_summary::total_time_in_seconds)
          .def_readwrite("linear_solver_time_in_seconds",
                         &s_summary::linear_solver_time_in_seconds)
          .def_readwrite("num_linear_solves", &s_summary::num_linear_solves)
          .def_readwrite("residual_evaluation_time_in_seconds",
                         &s_summary::residual_evaluation_time_in_seconds)
          .def_readwrite("num_residual_evaluations",
                         &s_summary::num_residual_evaluations)
          .def_readwrite("jacobian_evaluation_time_in_seconds",
                         &s_summary::jacobian_evaluation_time_in_seconds)
          .def_readwrite("num_jacobian_evaluations",
                         &s_summary::num_jacobian_evaluations)
          .def_readwrite("inner_iteration_time_in_seconds",
                         &s_summary::inner_iteration_time_in_seconds)
          .def_readwrite(
              "line_search_cost_evaluation_time_in_seconds",
              &s_summary::line_search_cost_evaluation_time_in_seconds)
          .def_readwrite(
              "line_search_gradient_evaluation_time_in_seconds",
              &s_summary::line_search_gradient_evaluation_time_in_seconds)
          .def_readwrite(
              "line_search_polynomial_minimization_time_in_seconds",
              &s_summary::line_search_polynomial_minimization_time_in_seconds)
          .def_readwrite("line_search_total_time_in_seconds",
                         &s_summary::line_search_total_time_in_seconds)
          .def_readwrite("num_parameter_blocks",
                         &s_summary::num_parameter_blocks)
          .def_readwrite("num_parameters", &s_summary::num_parameters)
          .def_readwrite("num_effective_parameters",
                         &s_summary::num_effective_parameters)
          .def_readwrite("num_residual_blocks", &s_summary::num_residual_blocks)
          .def_readwrite("num_residuals", &s_summary::num_residuals)
          .def_readwrite("num_parameter_blocks_reduced",
                         &s_summary::num_parameter_blocks_reduced)
          .def_readwrite("num_parameters_reduced",
                         &s_summary::num_parameters_reduced)
          .def_readwrite("num_effective_parameters_reduced",
                         &s_summary::num_effective_parameters_reduced)
          .def_readwrite("num_residual_blocks_reduced",
                         &s_summary::num_residual_blocks_reduced)
          .def_readwrite("num_residuals_reduced",
                         &s_summary::num_residuals_reduced)
          .def_readwrite("is_constrained", &s_summary::is_constrained)
          .def_readwrite("num_threads_given", &s_summary::num_threads_given)
          .def_readwrite("num_threads_used", &s_summary::num_threads_used)
#if CERES_VERSION_MAJOR < 2
          .def_readwrite("num_linear_solver_threads_given",
                         &s_summary::num_linear_solver_threads_given)
          .def_readwrite("num_linear_solver_threads_used",
                         &s_summary::num_linear_solver_threads_used)
#endif
          .def_readwrite("linear_solver_type_given",
                         &s_summary::linear_solver_type_given)
          .def_readwrite("linear_solver_type_used",
                         &s_summary::linear_solver_type_used)
          .def_readwrite("schur_structure_given",
                         &s_summary::schur_structure_given)
          .def_readwrite("schur_structure_used",
                         &s_summary::schur_structure_used)
          .def_readwrite("inner_iterations_given",
                         &s_summary::inner_iterations_given)
          .def_readwrite("inner_iterations_used",
                         &s_summary::inner_iterations_used)
          .def_readwrite("preconditioner_type_given",
                         &s_summary::preconditioner_type_given)
          .def_readwrite("preconditioner_type_used",
                         &s_summary::preconditioner_type_used)
          .def_readwrite("visibility_clustering_type",
                         &s_summary::visibility_clustering_type)
          .def_readwrite("trust_region_strategy_type",
                         &s_summary::trust_region_strategy_type)
          .def_readwrite("dogleg_type", &s_summary::dogleg_type)
          .def_readwrite("dense_linear_algebra_library_type",
                         &s_summary::dense_linear_algebra_library_type)
          .def_readwrite("sparse_linear_algebra_library_type",
                         &s_summary::sparse_linear_algebra_library_type)
          .def_readwrite("line_search_direction_type",
                         &s_summary::line_search_direction_type)
          .def_readwrite("line_search_type", &s_summary::line_search_type)
          .def_readwrite("line_search_interpolation_type",
                         &s_summary::line_search_interpolation_type)
          .def_readwrite("nonlinear_conjugate_gradient_type",
                         &s_summary::nonlinear_conjugate_gradient_type)
          .def_readwrite("max_lbfgs_rank", &s_summary::max_lbfgs_rank);
  make_dataclass(summary);

  using it_sum = ceres::IterationSummary;
  auto it_summary =
      py::class_<ceres::IterationSummary>(m, "IterationSummary")
          .def(py::init<>())
          .def_readonly("iteration", &it_sum::iteration)
          .def_readonly("step_is_valid", &it_sum::step_is_valid)
          .def_readonly("step_is_nonmonotonic", &it_sum::step_is_nonmonotonic)
          .def_readonly("step_is_successful", &it_sum::step_is_successful)
          .def_readonly("cost", &it_sum::cost)
          .def_readonly("cost_change", &it_sum::cost_change)
          .def_readonly("gradient_max_norm", &it_sum::gradient_max_norm)
          .def_readonly("gradient_norm", &it_sum::gradient_norm)
          .def_readonly("step_norm", &it_sum::step_norm)
          .def_readonly("relative_decrease", &it_sum::relative_decrease)
          .def_readonly("trust_region_radius", &it_sum::trust_region_radius)
          .def_readonly("eta", &it_sum::eta)
          .def_readonly("step_size", &it_sum::step_size)
          .def_readonly("line_search_function_evaluations",
                        &it_sum::line_search_function_evaluations)
          .def_readonly("line_search_gradient_evaluations",
                        &it_sum::line_search_gradient_evaluations)
          .def_readonly("line_search_iterations",
                        &it_sum::line_search_iterations)
          .def_readonly("linear_solver_iterations",
                        &it_sum::linear_solver_iterations)
          .def_readonly("iteration_time_in_seconds",
                        &it_sum::iteration_time_in_seconds)
          .def_readonly("step_solver_time_in_seconds",
                        &it_sum::step_solver_time_in_seconds)
          .def_readonly("cumulative_time_in_seconds",
                        &it_sum::cumulative_time_in_seconds);
}
