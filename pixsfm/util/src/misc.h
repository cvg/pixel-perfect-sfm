#pragma once

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/H5Object.hpp>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

// https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
inline bool ends_with(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline bool ends_with(std::string const& value,
                      std::vector<std::string>& endings) {
  for (std::string& ending : endings) {
    if (ends_with(value, ending)) {
      return true;
    }
  }
  return false;
}

inline std::vector<std::string> GetImageKeys(
    HighFive::Group group, std::string current_path = "",
    std::vector<std::string> viable_endings = {".png", ".jpeg", ".jpg", ".JPEG",
                                               ".JPG"}) {
  std::vector<std::string> valid_keys;

  std::vector<std::string> next_keys = group.listObjectNames();
  std::string sep = (current_path.length() > 0) ? "/" : "";
  for (auto& key : next_keys) {
    if (ends_with(key, viable_endings)) {
      valid_keys.push_back(current_path + sep + key);
    } else if (group.getObjectType(key) == HighFive::ObjectType::Group) {
      auto sub_valid_keys = GetImageKeys(
          group.getGroup(key), current_path + sep + key, viable_endings);
      valid_keys.insert(valid_keys.end(), sub_valid_keys.begin(),
                        sub_valid_keys.end());
    }
  }
  return valid_keys;
}

template <typename idx_t, typename val_t>
inline val_t AccumulateValues(const std::unordered_map<idx_t, val_t>& datamap) {
  val_t out = static_cast<val_t>(0);
  for (auto& pair : datamap) {
    out += pair.second;
  }
  return out;
}