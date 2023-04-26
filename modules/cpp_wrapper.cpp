#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cudnn_convolution_backward", &at::cudnn_convolution_backward, "CUDNN convolution backward");
    try{
        m.def("cudnn_convolution_backward_input", &at::cudnn_convolution_backward_input, "CUDNN convolution backward for input");
        m.def("cudnn_convolution_backward_weight", &at::cudnn_convolution_backward_weight, "CUDNN convolution backward for weight");
    } catch (std::exception &e) {
        std::cerr << "cudnn_convolution_backward_input and cudnn_convolution_backward_weight not loaded!";
    }
}