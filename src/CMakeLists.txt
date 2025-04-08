cmake_minimum_required(VERSION 3.20.1)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(ValidateTritonClient VERSION 0.1.0 LANGUAGES C CXX)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_FLAGS "-Wno-error=format-truncation")

find_package(TritonCommon CONFIG REQUIRED)
find_package(TritonClient CONFIG REQUIRED COMPONENTS TritonClient::triton-client)

add_executable(test_resnet50 test_resnet50.cxx)
target_link_libraries(test_resnet50 PUBLIC tbb)
set_target_properties(test_resnet50
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)

target_link_libraries(
  test_resnet50
  PRIVATE
    libgrpcclient.so
    libprotobuf.a
)

# install packages
install(TARGETS test_resnet50
        DESTINATION bin
)
