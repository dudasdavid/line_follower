cmake_minimum_required(VERSION 2.8)
project(line_follower)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS nodes/line_controller.py
  nodes/line_controller_cnn.py
  nodes/line_controller_cnn_pi.py
  nodes/line_controller_pi.py
  nodes/line_detector.py
  nodes/line_detector_cnn.py
  nodes/line_detector_cnn_pi.py
  nodes/line_detector_pi.py
  nodes/save_traning_pictures.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)