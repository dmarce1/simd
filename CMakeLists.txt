cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(simd CXX)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
)

execute_process(
	COMMAND rm -rf ./generated_code
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)	

execute_process(
	COMMAND ${CMAKE_CXX_COMPILER} -g -O0 ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -pthread -lmpfr -lgmp -o codegen
  	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
execute_process(
  	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/codegen
)

FILE(GLOB SRC_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*.cpp)
add_library(simd ${SRC_FILES} ${GENERATED_HEADER_FILES})
target_include_directories(simd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(simd PUBLIC -march=native -ffast-math)
set_target_properties(simd PROPERTIES PUBLIC_HEADER "include/simd.hpp")

add_executable(simd_test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(simd_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(simd_test PUBLIC simd pthread mpfr gmp)
target_compile_options(simd_test PUBLIC -pthread -ffast-math -march=native)
install(TARGETS simd_test DESTINATION bin)
INSTALL(TARGETS simd 
        LIBRARY DESTINATION simd
        PUBLIC_HEADER DESTINATION include
)


