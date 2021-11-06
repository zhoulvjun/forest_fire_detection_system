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

#ifndef __SYSTEMLIB_HPP__
#define __SYSTEMLIB_HPP__

#include <ros/ros.h>
#include <string.h>
#include <sys/time.h>
#include <tools/PrintControl/PrintCtrlImp.h>

#include <fstream>
#include <iostream>

namespace FFDS {
namespace TOOLS {

using namespace std;

/* return as second */
inline float getRosTimeInterval(ros::Time begin) {
  ros::Time time_now = ros::Time::now();
  float currTimeSec = time_now.sec - begin.sec;
  float currTimenSec = time_now.nsec / 1e9 - begin.nsec / 1e9;
  return (currTimeSec + currTimenSec);
}

/* return as ms */
inline long getSysTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

/* return as ms */
inline long getTimeInterval(const long begin_time) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000 + tv.tv_usec / 1000 - begin_time);
}

template <typename T>
void write2Files(string file_path_name, string item, T data) {
  long time_stamp = getSysTime();
  fstream oufile;

  oufile.open(file_path_name.c_str(), ios::app | ios::out);

  oufile << fixed << time_stamp << "\t"
         << "\t" << item;
  oufile << "\t" << data;
  oufile << endl;

  if (!oufile) PRINT_ERROR("something wrong to open or write!");
  oufile.close();
}

/**
 * @Input: string
 * @Output:string
 * @Description:将两个string合并
 */
inline string addStr(string a, string b) { return a + b; }
}  // namespace TOOLS
}  // namespace FFDS

#endif
