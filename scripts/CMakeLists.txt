## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  detection.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

