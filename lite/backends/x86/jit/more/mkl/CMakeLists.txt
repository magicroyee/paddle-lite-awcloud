file(GLOB jit_mkl_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
set(X86_JIT_MORE_SRC ${X86_JIT_MORE_SRC} ${jit_mkl_src} CACHE INTERNAL "")

# use mkl kernels by name and type
USE_JITKERNEL_MORE_LITE(kMatMul, mkl)
USE_JITKERNEL_MORE_LITE(kVMul, mkl)
USE_JITKERNEL_MORE_LITE(kVAdd, mkl)
USE_JITKERNEL_MORE_LITE(kVScal, mkl)
USE_JITKERNEL_MORE_LITE(kStrideScal, mkl)
USE_JITKERNEL_MORE_LITE(kVExp, mkl)
USE_JITKERNEL_MORE_LITE(kVSquare, mkl)
USE_JITKERNEL_MORE_LITE(kVCopy, mkl)
USE_JITKERNEL_MORE_LITE(kVSigmoid, mkl)
USE_JITKERNEL_MORE_LITE(kVTanh, mkl)
USE_JITKERNEL_MORE_LITE(kSeqPool, mkl)
USE_JITKERNEL_MORE_LITE(kSoftmax, mkl)
USE_JITKERNEL_MORE_LITE(kEmbSeqPool, mkl)
USE_JITKERNEL_MORE_LITE(kSgd, mkl)
USE_JITKERNEL_MORE_LITE(kVBroadcast, mkl)
