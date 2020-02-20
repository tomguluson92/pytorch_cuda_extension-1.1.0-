#include <torch/torch.h>

/*
    define your own cuda extension.

    This example is just add N to the original image.
*/

// CUDA forward declarations

at::Tensor test1_cuda(
        at::Tensor image,
        size_t N);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor test1(
        at::Tensor image,
        size_t N) {

    CHECK_INPUT(image);

    return test1_cuda(image, N);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test1", &test1, "test1 (CUDA)");
}