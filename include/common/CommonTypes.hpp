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

#include <iostream>

namespace FFDS {
namespace COMMON {

/* Local earth-fixed coordinates position */
template<typename T>
struct LocalPosition {
  T x{0.0};
  T y{0.0};
  T z{0.0};
};

} // namespace COMMON
} // namespace FFDS

#endif /* COMMONTYPES_HPP */
