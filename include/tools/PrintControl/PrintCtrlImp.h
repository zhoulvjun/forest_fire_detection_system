#ifndef _PRINT_CTRL_IMP_H_
#define _PRINT_CTRL_IMP_H_

/**
 *
 *@file Print control
 *
 *@author lee-shun
 *
 *@data 2020-08-10
 *
 *@version
 *
 *@brief control the output level
 *
 *@copyright
 *
 */

/* NOTE: namespace can not control macros! */

#include <cstdio>
#include <cstdlib>

/** 调试输出级别 */
#define NONE 0
#define ERROR 1
#define WARN 2
#define INFO 3
#define ENTRY 4
#define DEBUG 5

/**
 *TODO:
 * 需要定义PRINT_LEVEL
 * */

#define PRINT_LEVEL DEBUG

/* 颜色的定义
 * TODO: 此处的#color有两个宏定义初始化先后顺序的问题
 * */

#define COLOR(color, msg) "\033[0;1;" #color "m" msg "\033[0m"
#define RED 31
#define GREEN 32
#define YELLOW 33
#define PURPLE 34
#define PINK 35
#define BLUE 36

#define PRINT_PURE(level, ...)                                                 \
  do {                                                                         \
    if (level <= PRINT_LEVEL) {                                                \
      printf("[" #level "]>>" __VA_ARGS__);                                    \
      printf("\n");                                                            \
    }                                                                          \
  } while (0);

#define PRINT(color, level, ...)                                               \
  do {                                                                         \
    if (level <= PRINT_LEVEL) {                                                \
      printf(COLOR(36, "[File:%s  Line:%d  Function:%s]\n"), __FILE__,         \
             __LINE__, __PRETTY_FUNCTION__);                                   \
      printf(COLOR(color, "[" #level "]") __VA_ARGS__);                        \
      printf("\n");                                                            \
    }                                                                          \
  } while (0);

#define PRINT_ERROR(...)                                                       \
  do {                                                                         \
    PRINT(31, ERROR, ##__VA_ARGS__);                                           \
  } while (0);

#define PRINT_WARN(...)                                                        \
  do {                                                                         \
    PRINT(32, WARN, ##__VA_ARGS__);                                            \
  } while (0);

#define PRINT_INFO(...)                                                        \
  do {                                                                         \
    PRINT(33, INFO, ##__VA_ARGS__);                                            \
  } while (0);

#define PRINT_ENTRY(...)                                                       \
  do {                                                                         \
    PRINT(34, ENTRY, ##__VA_ARGS__);                                           \
  } while (0);

#define PRINT_DEBUG(...)                                                       \
  do {                                                                         \
    PRINT(35, DEBUG, ##__VA_ARGS__);                                           \
  } while (0);

#endif /*头文件 */
