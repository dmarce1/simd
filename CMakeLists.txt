cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(simd CXX)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(SIMD_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_code")
set(SIMD_GENERATED_MATH "${SIMD_GENERATED_DIR}/src/math.cpp")

# The code generator is an ordinary target rather than something compiled and
# run by execute_process() while configuring. This keeps `cmake` quiet and
# fast, lets the generator be rebuilt with the project's toolchain, and makes
# regeneration a build step that can be re-run when its inputs change.
add_executable(simd_codegen ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp)
target_compile_definitions(
	simd_codegen PRIVATE SIMD_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(simd_codegen PRIVATE -O3)
target_link_libraries(simd_codegen PRIVATE mpfr gmp pthread)

# Regenerated whenever the generator or the sources it splices in change.
add_custom_command(
	OUTPUT ${SIMD_GENERATED_MATH}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SIMD_GENERATED_DIR}/src
	COMMAND simd_codegen
	DEPENDS simd_codegen
	        ${CMAKE_CURRENT_SOURCE_DIR}/include/code.hpp
	        ${CMAKE_CURRENT_SOURCE_DIR}/include/code64.hpp
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating SIMD math functions"
	VERBATIM
)

add_library(simd ${SIMD_GENERATED_MATH})
add_library(simd::simd ALIAS simd)
target_include_directories(simd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(simd PUBLIC -march=native -mavx2)
set_target_properties(simd PROPERTIES PUBLIC_HEADER "include/simd.hpp")

add_executable(simd_test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_include_directories(simd_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simd_test PUBLIC simd pthread mpfr gmp)
target_compile_options(simd_test PUBLIC -pthread -march=native)

install(TARGETS simd_test DESTINATION bin)
install(TARGETS simd
        LIBRARY DESTINATION simd
        PUBLIC_HEADER DESTINATION include
)
