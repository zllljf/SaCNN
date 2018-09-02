#ifndef CAFFE_CROWD_DATA_LAYER_HPP_
#define CAFFE_CROWD_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/**
 * @brief Provides data and density map to the Net.
 *  top[0]: crowd image
 *  top[1]: density map
 */

using namespace cv;

template <typename Dtype>
struct HeadLocation {
  Dtype x;
  Dtype y;
};

template <typename Dtype>
class CrowdDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CrowdDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~CrowdDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CrowdData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void ReadLocationFromTextFile(const std::string filename);
  virtual void ShuffleData();
  virtual void load_batch(Batch<Dtype>* batch);
  virtual void Transform(const cv::Mat &cv_img, Blob<Dtype> *transformed_blob,
          const bool do_mirror, const int offset);
  virtual unsigned int PrefetchRand();
  virtual Mat ReadCrowdImageToCVMat(const string& filename, const bool is_color);
  virtual vector<int> GetImageBlobShape(const Mat& cv_img, const bool is_color);
  virtual vector<int> GetDmapBlobShape(const Mat& cv_img, const int ds_times);
  virtual Mat fspecial(const int size, const float sigma);
  virtual void GetDensityMap(Mat& cv_dmap, const vector<HeadLocation<int> >& head_location,
          const float sigma);


  int lines_id_;
  int downsamp_times_;
  Dtype transform_scale_;
  Dtype base_sigma_;
  vector<std::pair<std::string, std::string> > lines_;
  vector<HeadLocation<int> > gt_loc_;
  vector<Dtype> mean_values_;

};


}  // namespace caffe

#endif  // CAFFE_CROWD_DATA_LAYER_HPP_
