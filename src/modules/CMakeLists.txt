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

file(GLOB_RECURSE MODULE_PATH_LIST *.cpp)

foreach(MODULE_PATH ${MODULE_PATH_LIST})

    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" MODULE_NAME ${MODULE_PATH})

    # generate the library
    add_library(${MODULE_NAME} ${MODULE_PATH})
    target_link_libraries(${MODULE_NAME}
        ${catkin_LIBRARIES}
        )

endforeach()
