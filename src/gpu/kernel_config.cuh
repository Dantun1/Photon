#pragma once


namespace photon::gpu::kernelcfg {
  constexpr int REDUCTION_THREADS_PB = 256;
  // Occupancy vs overserialisation tradeoff. 
  // More blocks are heavy but we need to maximise occupancy via finer grain parallelism
  constexpr int REDUCTION_WAVES = 2;
  // min number of elements per thread to launch a new block 
  constexpr int REDUCTION_MIN_ELEMS_PER_THREAD = 4;
}
