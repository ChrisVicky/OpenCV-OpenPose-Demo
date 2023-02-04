# This file if for logger libraries
add_compile_options(-lpthread -ldl)
add_library(openpose multi-person-openpose.cpp)
