#include <vector>
#include "caffe/layers/crowd_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void CrowdDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // 1. Reshape to loaded crowd image and copy it to top[0].
  top[0]->ReshapeLike(batch->data_);
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  // 2. Reshape to loaded density map and copy it to top[1].
  top[1]->ReshapeLike(batch->label_);
  caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
      top[1]->mutable_gpu_data());
  DLOG(INFO) << "Prefetch copied";
  // 3. Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);

}

INSTANTIATE_LAYER_GPU_FORWARD(CrowdDataLayer);

}  // namespace caffe
