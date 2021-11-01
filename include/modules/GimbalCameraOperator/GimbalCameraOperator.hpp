/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GimbalCameraOperator.hpp
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

#ifndef __GIMBALCAMERAOPERATOR_HPP__
#define __GIMBALCAMERAOPERATOR_HPP__

#include <PX4-Matrix/matrix/Vector2.hpp>
#include <modules/BasicController/IncPIDController.hpp>

namespace FFDS {

namespace MODULES {

typedef matrix::Vector2<int> PixelPos;

class GimbalCameraOperator {
public:
  GimbalCameraOperator() : pidController(0.01, 0.01, 0.01){};
  void rotatePayload();
  void zoomCamera();

private:
  IncPIDController pidController;
  PixelPos currentPos;
  PixelPos setPos;
};

} // namespace MODULES
} // namespace FFDS

#endif /* GIMBALCAMERAOPERATOR_HPP */
