cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
project(FeatureCompression)


###############################################################################
# GLOBAL CONFIG
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

###############################################################################
# Configure OpenCV here
find_package(OpenCV REQUIRED)

###############################################################################
# Boost
find_package(Boost REQUIRED COMPONENTS filesystem system thread program_options)


###############################################################################
# RPATH Handling
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1") 

###############################################################################
# Sources
set(SOURCE_FILES 
${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/ac_extended.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/ImgBufferEntry.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/feature_coder.cpp)

set(DBoW_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib/libDBoW2.so)
	set(DBoW_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib/libDBoW2.so)
endif()



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
add_library(LBFC2 SHARED ${SOURCE_FILES})
install(TARGETS LBFC2 DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
target_link_libraries(LBFC2 ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
