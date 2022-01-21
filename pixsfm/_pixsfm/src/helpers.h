#pragma once

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "util/src/log_exceptions.h"
#include <third-party/half.h>

#include <iostream>
#include <regex>
#include <string>

using float16 = half_float::half;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Make pybind11 support float16 conversion
// https://github.com/eacousineau/repro/blob/43407e3/python/pybind11/custom_tests/test_numpy_issue1776.cc
namespace pybind11 {
namespace detail {

#define SINGLE_ARG(...) __VA_ARGS__

#define ASSIGN_PYDICT_ITEM(dict, key, type) \
  if (dict.contains(#key)) key = dict[#key].cast<type>();

#define ASSIGN_PYDICT_ITEM_TO_MEMBER(obj, dict, key, type) \
  if (dict.contains(#key)) obj.key = dict[#key].cast<type>();

template <typename T>
struct npy_scalar_caster {
  PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
  using Array = array_t<T>;

  bool load(handle src, bool convert) {
    // Taken from Eigen casters. Permits either scalar dtype or scalar array.
    handle type = dtype::of<T>().attr("type");  // Could make more efficient.
    if (!convert && !isinstance<Array>(src) && !isinstance(src, type))
      return false;
    Array tmp = Array::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }
    return false;
  }

  static handle cast(T src, return_value_policy, handle) {
    Array tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});
    // You could also just return the array if you want a scalar array.
    object scalar = tmp[tuple()];
    return scalar.release();
  }
};

}  // namespace detail
}  // namespace pybind11

static_assert(sizeof(float16) == 2, "Bad size");

namespace pybind11 {
namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

// Kinda following:
// https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <>
struct type_caster<float16> : npy_scalar_caster<float16> {
  static constexpr auto name = _("float16");
};

}  // namespace detail
}  // namespace pybind11

template <typename T>
inline T pyStringToEnum(const py::enum_<T>& enm, const std::string& value) {
  auto values = enm.attr("__members__").template cast<py::dict>();
  auto strVal = py::str(value);
  if (values.contains(strVal)) {
    return T(values[strVal].template cast<T>());
  }
  std::string msg =
      "ERROR: Invalid string value " + value + " for enum " +
      std::string(enm.attr("__name__").template cast<std::string>());
  THROW_EXCEPTION(std::out_of_range, msg.c_str());
  T t;
  return t;
}

// automatic conversion of python list to opaque std::vector
// Involves copy!

template <typename T>
inline void AddListToVectorConstructor(
    py::class_<std::vector<T>, std::unique_ptr<std::vector<T>>>& cls) {
  cls.def(py::init<>([](py::list list) {
    std::vector<T> vec;
    for (const auto& it : list) {
      vec.push_back(list.cast<T>());
    }
    return vec;
  }));
  py::implicitly_convertible<py::list, std::vector<T>>();
}

// automatic conversion of strings to enum by name
template <typename T>
inline void AddStringToEnumConstructor(py::enum_<T>& enm) {
  enm.def(py::init([enm](const std::string& value) {
    return pyStringToEnum(enm, py::str(value));  // str constructor
  }));
  py::implicitly_convertible<std::string, T>();
}

// This function adds the following methods to a class:
//  def __init__(self, dict): recursively merges options in the dictionary into
//  the defaults. def __init__(self, **kwargs): recursivelymerges keyword
//  options into the defaults. def mergedict(self, dict): recursively merges the
//  dictionary into the defaults. def summary(self, write_type: bool): returns a
//  nicely formatted string with current options. def todict(self): returns all
//  options as a python dict.

// The merge performs type-checks and casts options to the specific type if a
// conversion is defined. Merge is strict, i.e. it fails if no conversion can be
// found, or unknown options are found in the input dict.

// The function also registers implicit conversions from dict/kwargs to the
// specific class.
template <typename T>
inline void make_dataclass(py::class_<T> cls) {
  // dict-constructor
  cls.def(py::init([cls](py::dict dict) {
    auto self = py::object(cls());
    self.attr("mergedict").attr("__call__")(dict);
    return self.cast<T>();
  }));
  py::implicitly_convertible<py::dict, T>();

  // kwargs-constructor
  cls.def(py::init([cls](py::kwargs kwargs) {
    py::dict dict = kwargs.cast<py::dict>();
    auto self = py::object(cls(dict));
    return self.cast<T>();
  }));

  // Merge a dict into the defaults
  py::implicitly_convertible<py::kwargs, T>();
  cls.def("mergedict", [cls](py::object& self, py::dict dict) {
    for (auto& it : dict) {
      try {
        if (py::hasattr(self.attr(it.first), "mergedict")) {
          self.attr(it.first).attr("mergedict").attr("__call__")(it.second);
        } else {
          self.attr(it.first) = it.second;
        }
      } catch (const py::error_already_set& ex) {
        if (ex.matches(PyExc_TypeError)) {
          // If fail we try bases of the class
          py::list bases = self.attr(it.first)
                               .attr("__class__")
                               .attr("__bases__")
                               .cast<py::list>();
          bool success_on_base = false;
          for (auto& base : bases) {
            try {
              self.attr(it.first) = base(it.second);
              success_on_base = true;
              break;
            } catch (const py::error_already_set& ex) {
              continue;  // We anyway throw afterwards
            }
          }
          if (success_on_base) {
            continue;
          }
          std::stringstream ss;
          ss << cls.attr("__name__").template cast<std::string>() << "."
             << py::str(it.first).template cast<std::string>()
             << ": Could not convert "
             << py::type::of(it.second.cast<py::object>())
                    .attr("__name__")
                    .template cast<std::string>()
             << ": " << py::str(it.second).template cast<std::string>()
             << " to '"
             << py::type::of(self.attr(it.first))
                    .attr("__name__")
                    .template cast<std::string>()
             << "'.";
          // We write the err message to give info even if exceptions
          // is catched outside, e.g. in function overload resolve
          std::cerr << "Internal TypeError: " << ss.str() << std::endl;
          throw(
              py::type_error(std::string("Failed to merge dict into class: ") +
                             "Could not assign " +
                             py::str(it.first).template cast<std::string>()));
        } else if (ex.matches(PyExc_AttributeError) &&
                   py::str(ex.value()).cast<std::string>() ==
                       std::string("can't set attribute")) {
          std::stringstream ss;
          ss << cls.attr("__name__").template cast<std::string>() << "."
             << py::str(it.first).template cast<std::string>()
             << " defined readonly.";
          throw py::attribute_error(ss.str());
        } else if (ex.matches(PyExc_AttributeError)) {
          std::cerr << "Internal AttributeError: "
                    << py::str(ex.value()).cast<std::string>() << std::endl;
          throw;
        } else {
          throw;
        }
      }
    }
  });

  // Pretty-formatted string summary. If write_type==true, write also the type
  // string for each entry.
  cls.def(
      "summary",
      [cls](const T& self, bool write_type) {
        std::stringstream ss;
        auto pyself = py::cast(self);
        std::string prefix = "    ";
        bool after_subsummary = false;
        ss << cls.attr("__name__").template cast<std::string>() << ":\n";
        for (auto& handle : pyself.attr("__dir__")()) {
          std::string attribute = py::str(handle);
          auto member = pyself.attr(attribute.c_str());

          if (attribute.find("__") != 0 &&
              attribute.rfind("__") == std::string::npos &&
              !py::hasattr(member, "__func__")) {
            if (py::hasattr(member, "summary")) {
              std::string summ = member.attr("summary")
                                     .attr("__call__")(write_type)
                                     .template cast<std::string>();
              summ = std::regex_replace(summ, std::regex("\n"), "\n" + prefix);
              if (!after_subsummary) {
                ss << prefix;
              }
              ss << attribute << ": " << summ;
              after_subsummary = true;
            } else {
              if (!after_subsummary) {
                ss << prefix;
              }
              ss << attribute;
              if (write_type) {
                ss << ": "
                   << py::type::of(member)
                          .attr("__name__")
                          .template cast<std::string>();
              }
              ss << " = " << py::str(member).template cast<std::string>()
                 << "\n";
              after_subsummary = false;
            }
          }
        }
        return ss.str();
      },
      py::arg("write_type") = false);

  // Convert all options to a python dictionary, and return it.
  cls.def("todict", [cls](const T& self) {
    auto pyself = py::cast(self);
    py::dict dict;
    for (auto& handle : pyself.attr("__dir__")()) {
      std::string attribute = py::str(handle);
      auto member = pyself.attr(attribute.c_str());
      if (attribute.find("__") != 0 &&
          attribute.rfind("__") == std::string::npos &&
          !py::hasattr(member, "__func__")) {
        if (py::hasattr(member, "todict")) {
          dict[attribute.c_str()] = member.attr("todict")
                                        .attr("__call__")()
                                        .template cast<py::dict>();
        } else {
          dict[attribute.c_str()] = member;
        }
      }
    }
    return dict;
  });
}