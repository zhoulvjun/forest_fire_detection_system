/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PrintCtrlImp.h
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-10
 *
 *   @Description:
 *
 *******************************************************************************/

/* NOTE: namespace can not control macros in Cpp! */

#ifndef INCLUDE_TOOLS_PRINTCONTROL_PRINTCTRLIMP_H_
#define INCLUDE_TOOLS_PRINTCONTROL_PRINTCTRLIMP_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

/** 调试输出级别 */
#define NONE 0
#define ERROR 1
#define WARN 2
#define INFO 3
#define ENTRY 4
#define DEBUG 5

#define PRINT_LEVEL DEBUG

#define COLOR(color, msg) "\033[0;1;" #color "m" msg "\033[0m"
#define RED 31
#define GREEN 32
#define YELLOW 33
#define PURPLE 34
#define PINK 35
#define BLUE 36

#define FILENAME(x) strrchr(x, '/') ? strrchr(x, '/') + 1 : x

#define PRINT_PURE(level, ...)              \
  do {                                      \
    if (level <= PRINT_LEVEL) {             \
      printf("[" #level "]>>" __VA_ARGS__); \
      printf("\n");                         \
    }                                       \
  } while (0);

#define PRINT(color, level, ...)                                 \
  do {                                                           \
    if (level <= PRINT_LEVEL) {                                  \
      printf(COLOR(36, "[File:%s  Line:%d  Function:%s]\n"),     \
             FILENAME(__FILE__), __LINE__, __PRETTY_FUNCTION__); \
      printf(COLOR(color, "[" #level "]") __VA_ARGS__);          \
      printf("\n");                                              \
    }                                                            \
  } while (0);

#define PRINT_ERROR(...)             \
  do {                               \
    PRINT(31, ERROR, ##__VA_ARGS__); \
  } while (0);

#define PRINT_WARN(...)             \
  do {                              \
    PRINT(33, WARN, ##__VA_ARGS__); \
  } while (0);

#define PRINT_INFO(...)             \
  do {                              \
    PRINT(32, INFO, ##__VA_ARGS__); \
  } while (0);

#define PRINT_ENTRY(...)             \
  do {                               \
    PRINT(34, ENTRY, ##__VA_ARGS__); \
  } while (0);

#define PRINT_DEBUG(...)             \
  do {                               \
    PRINT(35, DEBUG, ##__VA_ARGS__); \
  } while (0);

#endif  // INCLUDE_TOOLS_PRINTCONTROL_PRINTCTRLIMP_H_
