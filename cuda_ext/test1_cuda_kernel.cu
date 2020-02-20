#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
/*
    define your own cuda extension.

    This example is just add N to the original image.
*/
namespace {
template <typename scalar_t>
__global__ void test1_cuda_kernel(
    scalar_t* __restrict__ image,
    size_t N,
    size_t batch_size,
    size_t channel,
    size_t image_height,
    size_t image_width) {

        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int num_threads = blockDim.x * gridDim.x;

        while(idx < batch_size*channel*image_height*image_width) {
            image[idx] = image[idx] + N;
            idx += num_threads;
        }
    }
}

at::Tensor test1_cuda(
        at::Tensor image,
        size_t N) {
    const auto batch_size = image.size(0);
    const auto channel = image.size(1);
    const auto image_height = image.size(2);
    const auto image_width = image.size(3);

    const int threads = 32;
    const dim3 blocks ((batch_size * channel - 1) / threads + 1);

     // 注意, AT_DISPATCH_FLOATING_TYPES的第2个参数必须和所在函数体的名称一样! 否则会就无法dispatch.
     AT_DISPATCH_FLOATING_TYPES(image.type(), "test1_cuda", ([&] {
      test1_cuda_kernel<scalar_t><<<blocks, threads>>>(
          image.data<scalar_t>(),
          N,
          batch_size,
          channel,
          image_height,
          image_width);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in test1: %s\n", cudaGetErrorString(err));
    return image;
}