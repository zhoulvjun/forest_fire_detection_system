#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: CMakeLists.txt
#
#   @Author: Shun Li
#
#   @Date: 2021-10-22
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------

file(GLOB_RECURSE APP_PATH_LIST *.cpp)

foreach(APP_PATH ${APP_PATH_LIST})

    # get the name without "*.cpp" or "path.."
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" APP_NAME ${APP_PATH})

    # generate the library
    add_library(${APP_NAME} ${APP_PATH})
    target_link_libraries(${APP_NAME}
        ${catkin_LIBRARIES}
        )

endforeach()
