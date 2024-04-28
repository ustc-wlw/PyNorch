#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data;
    int* strides;
    int* shape;
    int ndim;
    int size;
} Tensor;

extern "C" {
    Tensor* create_tensor(float* data, int* shape, int ndim);
    float get_item(Tensor* tensor, int* indices);
    Tensor* create_tensor(float* data, int* shape, int ndim);
    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* sub_tensor(Tensor* tensor1, Tensor* tensor2);
    Tensor* elementwise_mul_tensor(Tensor* tensor1, Tensor* tensor2);
    void reshape_tensor(Tensor* tensor, int* new_shape, int new_ndim);
    Tensor* matmul_tensor(Tensor* tensor1, Tensor* tensor2);
}

#endif /* TENSOR_H */
