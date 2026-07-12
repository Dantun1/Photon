#include <backend_gpu.cuh>
#include <cuda_runtime.h>

namespace photon {
  namespace gpu {
    const DeviceManager& DeviceManager::get() {
      // access/construct singleton thread safe 
      static DeviceManager instance;
      return instance;
    }
    
    DeviceManager::DeviceManager() {
      // per device attrs 
      int device_id;
      cudaGetDevice(&device_id);

      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
      cudaDeviceGetAttribute(&max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
      cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id);
    }
  }
}
