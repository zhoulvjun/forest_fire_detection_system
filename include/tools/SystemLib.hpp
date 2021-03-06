/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: SystemLib.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-25
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef INCLUDE_TOOLS_SYSTEMLIB_HPP_
#define INCLUDE_TOOLS_SYSTEMLIB_HPP_

#include <ros/ros.h>
#include <sys/time.h>
#include <tools/PrintControl/PrintCtrlImp.h>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace FFDS {
namespace TOOLS {

/* return as second */
inline float getRosTimeInterval(const ros::Time& begin) {
  ros::Time time_now = ros::Time::now();
  float currTimeSec = time_now.sec - begin.sec;
  float currTimenSec = time_now.nsec / 1e9 - begin.nsec / 1e9;
  return (currTimeSec + currTimenSec);
}

/* return as ms */
inline int32_t getSysTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

/* return as ms */
inline int32_t getTimeInterval(const int32_t begin_time) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000 + tv.tv_usec / 1000 - begin_time);
}

template <typename T>
void write2Files(const std::string file_path_name, const std::string item,
                 const T data) {
  int32_t time_stamp = getSysTime();
  std::fstream oufile;

  oufile.open(file_path_name.c_str(), std::ios::app | std::ios::out);

  oufile << std::fixed << time_stamp << "\t"
         << "\t" << item;
  oufile << "\t" << data;
  oufile << std::endl;

  if (!oufile) PRINT_ERROR("something wrong to open or write!");
  oufile.close();
}

inline std::string addStr(const std::string a, const std::string b) {
  return a + b;
}

template <typename T>
T getParam(const YAML::Node& node, const std::string& paramName,
           const T& defaultValue) {
  T v;
  try {
    v = node[paramName].as<T>();
    ROS_INFO_STREAM("Found parameter: " << paramName << ", value: " << v);
  } catch (std::exception e) {
    v = defaultValue;
    ROS_WARN_STREAM("Cannot find value for parameter: "
                    << paramName << ", assigning default: " << v);
  }
  return v;
}

}  // namespace TOOLS
}  // namespace FFDS

#endif  // INCLUDE_TOOLS_SYSTEMLIB_HPP_
