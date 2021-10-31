/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GimbalCameraController.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-31
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef __GIMBALCAMERACONTROLLER_HPP__
#define __GIMBALCAMERACONTROLLER_HPP__

#include <PX4-Matrix/matrix/Vector2.hpp>

namespace FFDS {

namespace MODULES {

class GimbalCameraController {
  public:
    void rotatePayload();
    void zoomCamera();
  private:
    matrix::Vector2<int> currentPos;
    matrix::Vector2<int> setPos;

};

} // namespace MODULES
} // namespace FFDS

#endif /* GIMBALCAMERACONTROLLER_HPP */
