/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: MathLib.hpp
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

#ifndef __MATHLIB_HPP__
#define __MATHLIB_HPP__

#include <iostream>
#include <math.h>

namespace FFDS {
namespace TOOLS {

using namespace std;

#define CONSTANTS_RADIUS_OF_EARTH (double)6378137.0
#define EARTH_R (double)6378137.0
#define CONSTANTS_ONE_G (double)9.80665

float AbsNum(float a) {
  float result;
  if (a < 0)
    result = -a;

  else
    result = a;

  return result;
}

bool isFinite(float a) {
  if ((AbsNum(a) > 0.02) && (AbsNum(a) < 1000)) {
    return true;
  } else
    return false;
}

float Constrain(float val, float min, float max) {
  return (val < min) ? min : ((val > max) ? max : val);
}

float Max(const float a, const float b) { return (a > b) ? a : b; }

float Min(const float a, const float b) { return (a < b) ? a : b; }

void Quaternion2Euler(float quat[4], float angle[3]) {
  angle[0] = atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                   1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]));
  angle[1] = asin(2.0 * (quat[2] * quat[0] - quat[3] * quat[1]));
  angle[2] = atan2(2.0 * (quat[3] * quat[0] + quat[1] * quat[2]),
                   -1.0 + 2.0 * (quat[0] * quat[0] + quat[1] * quat[1]));
}

void Euler2Quaternion(float angle[3], float quat[4]) {
  double cosPhi_2 = cos(double(angle[0]) / 2.0);

  double sinPhi_2 = sin(double(angle[0]) / 2.0);

  double cosTheta_2 = cos(double(angle[1]) / 2.0);

  double sinTheta_2 = sin(double(angle[1]) / 2.0);

  double cosPsi_2 = cos(double(angle[2]) / 2.0);

  double sinPsi_2 = sin(double(angle[2]) / 2.0);

  quat[0] = float(cosPhi_2 * cosTheta_2 * cosPsi_2 +
                  sinPhi_2 * sinTheta_2 * sinPsi_2);

  quat[1] = float(sinPhi_2 * cosTheta_2 * cosPsi_2 -
                  cosPhi_2 * sinTheta_2 * sinPsi_2);

  quat[2] = float(cosPhi_2 * sinTheta_2 * cosPsi_2 +
                  sinPhi_2 * cosTheta_2 * sinPsi_2);

  quat[3] = float(cosPhi_2 * cosTheta_2 * sinPsi_2 -
                  sinPhi_2 * sinTheta_2 * cosPsi_2);
}

void MatrixPlusVector3(float vector_a[3], float rotmax[3][3],
                          float vector_b[3]) {
  vector_a[0] = rotmax[0][0] * vector_b[0] + rotmax[0][1] * vector_b[1] +
                rotmax[0][2] * vector_b[2];

  vector_a[1] = rotmax[1][0] * vector_b[0] + rotmax[1][1] * vector_b[1] +
                rotmax[1][2] * vector_b[2];

  vector_a[2] = rotmax[2][0] * vector_b[0] + rotmax[2][1] * vector_b[1] +
                rotmax[2][2] * vector_b[2];
}

/**
 * create rotation matrix for the quaternion
 */
void Quat2Rotmax(float q[4], float R[3][3]) {
  float aSq = q[0] * q[0];
  float bSq = q[1] * q[1];
  float cSq = q[2] * q[2];
  float dSq = q[3] * q[3];
  R[0][0] = aSq + bSq - cSq - dSq;
  R[0][1] = 2.0f * (q[1] * q[2] - q[0] * q[3]);
  R[0][2] = 2.0f * (q[0] * q[2] + q[1] * q[3]);
  R[1][0] = 2.0f * (q[1] * q[2] + q[0] * q[3]);
  R[1][1] = aSq - bSq + cSq - dSq;
  R[1][2] = 2.0f * (q[2] * q[3] - q[0] * q[1]);
  R[2][0] = 2.0f * (q[1] * q[3] - q[0] * q[2]);
  R[2][1] = 2.0f * (q[0] * q[1] + q[2] * q[3]);
  R[2][2] = aSq - bSq - cSq + dSq;
}

float Rad2Deg(float rad) {
  float deg;
  deg = rad * 180 / M_PI;
  return deg;
}

float Deg2Rad(float deg) {
  float rad;
  rad = deg * M_PI / 180;
  return rad;
}

// ref,result---lat,long,alt
void Meter2LatLongAlt(double ref[3], float x, float y, float z,
                          double result[3]) {

  if (x == 0 && y == 0) {
    result[0] = ref[0];
    result[1] = ref[1];
  } else {
    double local_radius = cos(Deg2Rad(ref[0])) * EARTH_R; // lat是

    result[0] =
        ref[0] +
        Rad2Deg(x / EARTH_R); //得到的是lat，x是北向位置，所以在大圆上

    result[1] = ref[1] + Rad2Deg(y / local_radius); //得到的是long，在维度圆上
  }

  result[2] = ref[2] + z; //高度
}

void LatLong2Meter(double a_pos[2], double b_pos[2],
                      double m[2]) { //参考点是a点，lat，long，alt
  double lat1 = a_pos[0];
  double lon1 = a_pos[1];

  double lat2 = b_pos[0];
  double lon2 = b_pos[1];

  double n_distance = Deg2Rad(lat2 - lat1) *
                      EARTH_R; //涉及到ned是向北增加，且纬度向北也增加

  double r_at_ref1 = cos(Deg2Rad(lat1)) * EARTH_R;

  double e_distance = Deg2Rad(lon2 - lon1) *
                      r_at_ref1; //涉及到ned是向东增加，但是经度向东减少

  m[0] = n_distance;
  m[1] = e_distance;
}
} // namespace TOOLS
} // namespace FFDS

#endif /* MATHLIB_HPP */
