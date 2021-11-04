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

#ifndef __COMMONTYPES_HPP__
#define __COMMONTYPES_HPP__

#include <cmath>
#include <iostream>

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

/* defalut is for H20T IR camera */
struct IRCameraParams {
  IRCameraParams(float orgImgWidthPix_, float orgImgHeightPix_,
                 float focalLength_, float equivalentFocalLength_,
                 float equivalentCrossLineInMM_)
      : orgImgWidthPix(orgImgWidthPix_),
        orgImgHeightPix(orgImgHeightPix_),
        focalLength(focalLength_),
        equivalentFocalLength(equivalentFocalLength_),
        equivalentCrossLineInMM(equivalentCrossLineInMM_) {
    float crossLinePix = std::sqrt(orgImgWidthPix * orgImgWidthPix +
                                    orgImgHeightPix * orgImgHeightPix);
    eachPixInMM = equivalentCrossLineInMM / crossLinePix;
  }

  IRCameraParams() {
    float crossLinePix = std::sqrt(orgImgWidthPix * orgImgWidthPix +
                                    orgImgHeightPix * orgImgHeightPix);
    eachPixInMM = equivalentCrossLineInMM / crossLinePix;
  }

  ~IRCameraParams() {}

  float orgImgWidthPix{640};
  float orgImgHeightPix{512};
  float focalLength{13.5};
  float equivalentFocalLength{58};
  float equivalentCrossLineInMM{42.27};
  float eachPixInMM;
};

}  // namespace COMMON
}  // namespace FFDS

#endif /* COMMONTYPES_HPP */
