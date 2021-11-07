/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: CommonTypes.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-26
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef INCLUDE_COMMON_COMMONTYPES_HPP_
#define INCLUDE_COMMON_COMMONTYPES_HPP_

#include <cmath>
#include <iostream>
#include <string>
#include <tools/SystemLib.hpp>

namespace FFDS {
namespace COMMON {

/* Local earth-fixed coordinates position */
template <typename T>
struct LocalPosition {
  T x{0.0};
  T y{0.0};
  T z{0.0};
};

/* the WpV2 mission state code */
/*
 * 0x0:ground station not start.
 * 0x1:mission prepared.
 * 0x2:enter mission.
 * 0x3:execute flying route mission.
 * 0x4:pause state.
 * 0x5:enter mission after ending pause.
 * 0x6:exit mission.
 * */
enum WpV2MissionState {};

struct CameraParams {
  CameraParams(float fullImgWidthPix_, float fullImgHeightPix_)
      : fullImgWidthPix(fullImgWidthPix_), fullImgHeightPix(fullImgHeightPix_) {
    splitImgWidthPix = fullImgWidthPix / 2;
    splitImgHeightPix = fullImgHeightPix / 2;
  }

  explicit CameraParams(std::string configPath) {
    YAML::Node node = YAML::LoadFile(configPath);
    fullImgWidthPix = FFDS::TOOLS::getParam(node, "full_img_width", 1920);
    fullImgHeightPix = FFDS::TOOLS::getParam(node, "full_img_height", 1080);

    splitImgWidthPix = FFDS::TOOLS::getParam(node, "split_img_width", 960);
    splitImgHeightPix = FFDS::TOOLS::getParam(node, "split_img_height", 540);
  }

  ~CameraParams() {}

  float fullImgWidthPix;
  float fullImgHeightPix;
  float splitImgWidthPix;
  float splitImgHeightPix;
};

}  // namespace COMMON
}  // namespace FFDS

#endif  // INCLUDE_COMMON_COMMONTYPES_HPP_
