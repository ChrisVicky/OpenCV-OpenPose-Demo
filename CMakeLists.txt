cmake_minimum_required(VERSION 3.25)
project(OpenPoseTest)

# Find Packages
FIND_PACKAGE( OpenCV REQUIRED )

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("./logsrc/")
INCLUDE_DIRECTORIES("./openpose")

add_subdirectory("./logsrc")
add_subdirectory("./openpose")

ADD_EXECUTABLE(run main.cpp)

TARGET_LINK_LIBRARIES(run ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(run openpose)
TARGET_LINK_LIBRARIES(run logger)

CONFIGURE_FILE(group.jpg group.jpg COPYONLY)
CONFIGURE_FILE(./default.xml default.xml COPYONLY)

FILE(GLOB allCopyFiles  "./pose/*")
FILE(COPY ${allCopyFiles} DESTINATION ./pose)
