### LIBRARIES ####
add_library(SLucAM_library SHARED
SLucAM_dataset.cpp
SLucAM_geometry.cpp
SLucAM_image.cpp
SLucAM_initialization.cpp
SLucAM_measurement.cpp
SLucAM_state.cpp)


### EXECUTABLE ###
add_executable(SLucAM main.cpp)


### LINKING ###
target_link_libraries(SLucAM
SLucAM_library
${OpenCV_LIBS}
${EIGEN3_LIBS})