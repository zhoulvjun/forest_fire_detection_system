/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: IncPIDController.hpp
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

#ifndef __INCPIDCONTROLLER_HPP__
#define __INCPIDCONTROLLER_HPP__

namespace FFDS {
namespace MODULES {

class IncPIDController {

public:
  IncPIDController(float p, float i, float d) : Kp(p), Ki(i), Kd(d){};

  void ctrl(float in);
  float fullOutput();
  float incOutput();
  void reset();
  void setPrevOutput(float prev);

private:
  float Kp;
  float Ki;
  float Kd;

  float input{0.0};
  float prev_input{0.0};
  float prev2_input{0.0};

  float increment{0.0};
  float output{0.0};
  float prev_output{0.0};

  void updateInput();
};

inline float IncPIDController::incOutput() { return increment; }

/**
 * @Input:
 * @Output:
 * @Description: 用于第一次进入时与其他控制方式的衔接
 */
inline void IncPIDController::setPrevOutput(float prev) { prev_output = prev; }

inline float IncPIDController::fullOutput() {

  output = prev_output + increment;
  prev_output = output;

  return output;
}

inline void IncPIDController::reset() {

  prev_input = 0.0;
  prev2_input = 0.0;
  output = 0.0;
}

inline void IncPIDController::updateInput() {

  prev2_input = prev_input;
  prev_input = input;
}

inline void IncPIDController::ctrl(float in) {

  input = in;
  float param_p = Kp * (input - prev_input);
  float param_i = Ki * input;
  float param_d = Kd * (input - 2 * prev_input + prev2_input);
  increment = param_p + param_i + param_d;

  updateInput();
}

} // namespace MODULES

} // namespace FFDS

#endif /* INCPIDCONTROLLER_HPP */
