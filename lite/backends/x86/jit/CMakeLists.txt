set(jit_file ${PADDLE_BINARY_DIR}/lite/backends/x86/jit/kernels.h)
file(WRITE ${jit_file} "// Generated by the lite/backends/x86/jit/CMakeLists.txt.  DO NOT EDIT!\n\n")
file(APPEND ${jit_file} "\#pragma once\n")
file(APPEND ${jit_file} "\#include \"lite/backends/x86/jit/helper.h\"\n")
file(APPEND ${jit_file} "\#include \"lite/backends/x86/jit/registry.h\"\n\n")

set(X86_MATH_DEPS  ${X86_MATH_DEPS} cblas gflags xxhash CACHE INTERNAL "")

# refer must go first
add_subdirectory(refer)
add_subdirectory(more)
if(WITH_XBYAK AND NOT APPLE AND NOT WIN32)
    add_subdirectory(gen)
endif()
