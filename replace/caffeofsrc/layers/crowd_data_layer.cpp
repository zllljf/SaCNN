//  Create on: 2016/9/19 ShanghaiTech
//  Author:    Yingying Zhang
#ifdef USE_OPENCV
#include <fstream>  
#include <iostream>  
#include <string>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/crowd_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

using namespace cv;

template <typename Dtype>
CrowdDataLayer<Dtype>::~CrowdDataLayer<Dtype>() {
  this->StopInternalThread();
}

//Read ground truth head location of crowd images
template <typename Dtype>
void CrowdDataLayer<Dtype>::ReadLocationFromTextFile(const std::string filename) {
  DLOG(INFO) << "Opening file " << filename;
  std::ifstream infile(filename.c_str());
  CHECK(infile.good()) << "Failed to open file " << filename;
  int num_crowd;
  CHECK(infile >> num_crowd);
  CHECK_GE(num_crowd, 0) << "Number of crowd must be positive!";
  gt_loc_.clear();
  gt_loc_.resize(num_crowd);
  for (int i = 0; i < num_crowd; i++) {
    HeadLocation<int> location;
    CHECK(infile >> location.x >> location.y);
    CHECK(0 <= location.x && 0 <= location.y);
    for (int j = 0; j < downsamp_times_; ++j) {
        location.x = location.x / 2;
        location.y = location.y / 2;
    }
    //LOG(INFO)<<"i="<<i<<", ("<<location.x<<", "<<location.y<<")";
    gt_loc_[i] = location;
  }
  infile.close();
}

// Read crowd image
template <typename Dtype>
Mat CrowdDataLayer<Dtype>::ReadCrowdImageToCVMat(const string& filename,
        const bool is_color) {
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
     cv::Mat cv_img_dest;
  Mat cv_img = imread(filename, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
  }
     cv_img_dest=cv_img;
     //cv:: resize(cv_img,cv_img_dest,cv::Size(1024,512));
     cv_img=cv_img_dest;
  return cv_img;
}

// Compute initial shape for crowd image
template <typename Dtype>
vector<int> CrowdDataLayer<Dtype>::GetImageBlobShape(const Mat& cv_img,
        const bool is_color) {
  const int channels = is_color ? 3 : 1;
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(height, 0);
  CHECK_GT(width, 0);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  return shape;
}

// Compute initial shape for density map
template <typename Dtype>
vector<int> CrowdDataLayer<Dtype>::GetDmapBlobShape(const Mat& cv_img,
        const int ds_times) {
  const int channels = 1;
  int height = cv_img.rows;
  int width = cv_img.cols;
  for (int i = 0; i < ds_times; ++i) {
      height = height / 2 + (height % 2);
      width = width / 2 + (width % 2);
  }
  // Check dimensions.
  CHECK_GT(height, 0);
  CHECK_GT(width, 0);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  return shape;
}

//2D Gaussain kernel, similar to function fspecial in Matlab
//Input size must be odd number
template <typename Dtype>
Mat CrowdDataLayer<Dtype>::fspecial(const int size, const float sigma) {
  int r_size = (size-1)/ 2;
  Mat kernel(size, size, CV_32F);
  float simga_2 = float(2) * sigma * sigma;
  for(int i = -r_size; i <= r_size; ++i) {
    int h = i + r_size;
    for (int j = (-r_size); j <= r_size; ++j) {
      int w = j + r_size;
      float v = exp(-(static_cast<float>(i*i) + static_cast<float>(j*j)) / simga_2);
      kernel.ptr<float>(h)[w] = v;
    }
  }
  Scalar sum_value = sum(kernel);
  Mat gaussian_kernel;
  kernel.convertTo(gaussian_kernel, CV_32F, (1/sum_value[0]));
  return gaussian_kernel;
}

// Generate gaussian kernel based density map
template <typename Dtype>
void CrowdDataLayer<Dtype>::GetDensityMap(Mat& cv_dmap,
       const vector<HeadLocation<int> >& head_location,
       const float sigma) {
  const int k_size = 25;
  const int r_size = (k_size - 1) / 2;
  Mat gaussian_kernel = fspecial(k_size, sigma);
  Mat extend_dmap(cv_dmap.rows + k_size - 1, cv_dmap.cols + k_size - 1, CV_32F, Scalar(0));
  const int num_head = head_location.size();
  for (int i = 0; i < num_head; ++i) {
    int loc_x = head_location[i].x;
    int loc_y = head_location[i].y;
    CHECK_GE(loc_x, 0) << "invalid head location: x!";
    CHECK_GE(loc_y, 0) << "invalid head location: y!";
    CHECK_LE(loc_x, cv_dmap.cols) << "invalid head location: x!";
    CHECK_LE(loc_y, cv_dmap.rows) << "invalid head location: y!";
    Rect location_box;
    location_box.x = loc_x;
    location_box.y = loc_y;
    location_box.width = k_size;
    location_box.height = k_size;
    Mat kernel_location(extend_dmap, location_box);
    kernel_location = kernel_location + gaussian_kernel;
  }
  Rect dmap_loc;
  dmap_loc.x = r_size;
  dmap_loc.y = r_size;
  dmap_loc.width = cv_dmap.cols;
  dmap_loc.height = cv_dmap.rows;
  Mat dstroi = extend_dmap(dmap_loc);
  dstroi.convertTo(cv_dmap, CV_32F, 1, 0);
}

//Shuffle crowd image and ground truth location.
template <typename Dtype>
void CrowdDataLayer<Dtype>::ShuffleData() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

//Return an unsigned integrate number
template <typename Dtype>
unsigned int CrowdDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

//Convert cvMat to data blob
template <typename Dtype>
void CrowdDataLayer<Dtype>::Transform(const Mat &cv_img,
  Blob<Dtype> *transformed_blob, const bool do_mirror,
  const int offset) {
  // 1. Check dimensions
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  CHECK_EQ(channels, img_channels);
  CHECK_EQ(height, img_height);
  CHECK_EQ(width, img_width);
  CHECK(cv_img.depth() == CV_8U || cv_img.depth() == CV_32F);
  // 2. Copy image (or density map data from cvMat to  data blob
  Dtype *data = transformed_blob->mutable_cpu_data() + offset;
  for (int h = 0; h < img_height; h++) {
    const uchar *psrc_u = NULL;
    const float *psrc_f = NULL;
    if(cv_img.depth() == CV_8U) {
      psrc_u = cv_img.ptr<uchar>(h);
    }
    if(cv_img.depth() == CV_32F) {
      psrc_f = cv_img.ptr<float>(h);
    }
    for (int w = 0; w < img_width; w++) {
      for (int c = 0; c < channels; c++) {
        if (do_mirror) {
          // For crowd image
          if(cv_img.depth() == CV_8U)
            data[(c * height + h) * width + w] = static_cast<Dtype> (
              (psrc_u[(img_width - 1 - w) * channels + c] - mean_values_[c])
              * transform_scale_);
          // For density map
          if (cv_img.depth() == CV_32F)
            data[(c * height + h) * width + w] = static_cast<Dtype> (
              psrc_f[(img_width - 1 - w) * channels + c]);
        } else {
          // For crowd image
          if(cv_img.depth() == CV_8U)
            data[(c * height + h) * width + w] = static_cast<Dtype> (
              (psrc_u[w * channels + c] - mean_values_[c])
              * transform_scale_);
          // For density map
          if (cv_img.depth() == CV_32F)
            data[(c * height + h) * width + w] = static_cast<Dtype> (
              psrc_f[w * channels + c]);
        }
      }
    }
  }
}

template <typename Dtype>
void CrowdDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  downsamp_times_ = this->layer_param_.image_data_param().downsamp_times();
  base_sigma_ = this->layer_param_.image_data_param().base_sigma();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  // 1.  Configure mean values and transform scale
  const bool has_mean_values = this->transform_param_.mean_value_size() > 0;
  int channel = is_color ? 3 : 1;
  if (has_mean_values) {
    const int mean_channel = this->transform_param_.mean_value_size();
    CHECK(mean_channel == 1 || mean_channel == channel) <<
        "Specify either 1 mean_value or as many as channels: " << channel;
    if (channel > 1 && mean_channel == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 0; c < channel; ++c) {
        mean_values_.push_back(this->transform_param_.mean_value(0));
      }
    } else {
      for (int c = 0; c < channel; ++c) {
        mean_values_.push_back(this->transform_param_.mean_value(c));
      }
    }
  } else {
    // If not specify mean value, set to zero
    for (int c = 0; c < channel; ++c) {
      mean_values_.push_back(Dtype(0));
    }
  }
  transform_scale_ = this->transform_param_.scale();

  // 2. Read the image and location file with filenames.
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string img_filename;
  string loc_filename;
  while (infile >> img_filename >> loc_filename) {
    //LOG(INFO)<<"img_filename:"<<img_filename<<", loc_filename:"<<loc_filename;
    lines_.push_back(std::make_pair(img_filename, loc_filename));
  }

  // 3. Randomly shuffle data.
  if (this->layer_param_.image_data_param().shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleData();
  }
  LOG(INFO) << "A total of " << lines_.size() << " crowd images.";

  // 4. Read an crowd image to initialize the top blob.
  // 4.1. crowd image.
  lines_id_ = 0;
  Mat cv_img = ReadCrowdImageToCVMat(root_folder + lines_[lines_id_].first,
          is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  vector<int> data_shape = GetImageBlobShape(cv_img, is_color);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(data_shape);
  }
  top[0]->Reshape(data_shape);
  LOG(INFO) << "output crowd image size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // 4.2. density map.
  vector<int> dmap_shape = GetDmapBlobShape(cv_img, downsamp_times_);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(dmap_shape);
  }
  top[1]->Reshape(dmap_shape);
  LOG(INFO) << "output density map size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();

}

// This function is called on prefetch thread.
template <typename Dtype>
void CrowdDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  double dmap_time = 0;
  CPUTimer timer;

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const bool is_color = image_data_param.is_color();
  const bool mirror = this->transform_param_.mirror();
  string root_folder = image_data_param.root_folder();
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  // 1. Reshape each batch according the image read currently.
  // Because each crowd image has different size, the batch_size needs
  // to be set to one.
  // Maybe use batch_size more than one in future.
  // 1.1. Reshape crowd image (data) and density map (label) of batch.
  int batch_size  = 1;
  Mat cv_img = ReadCrowdImageToCVMat(root_folder + lines_[lines_id_].first,
          is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  vector<int> data_shape = GetImageBlobShape(cv_img, is_color);
  data_shape[0] = batch_size;
  batch->data_.Reshape(data_shape);
  vector<int> dmap_shape = GetDmapBlobShape(cv_img, downsamp_times_);
  dmap_shape[0] = batch_size;
  batch->label_.Reshape(dmap_shape);

  // 2. Load crowd images and generate density maps.
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // 2.1 Load crowd images
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    Mat cv_img = ReadCrowdImageToCVMat(root_folder + lines_[lines_id_].first,
          is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    // 2.2 Apply transformations (mirror,maybe crop in future...) to the image
    timer.Start();
    int data_offset = item_id * (batch->data_.count(1));
    bool do_mirror = mirror;      // for validation and testing
    if (this->phase_ == TRAIN) {  // for training, do mirror jittering
      do_mirror = mirror && PrefetchRand() % 2;
    }
    Transform(cv_img, &batch->data_, do_mirror, data_offset);
    trans_time += timer.MicroSeconds();
    // 2.3 Generate density map
    timer.Start();
    int dmap_offset = item_id * (batch->label_.count(1));
    Mat dmap(dmap_shape[2], dmap_shape[3], CV_32F, Scalar(0));
    ReadLocationFromTextFile(root_folder + lines_[lines_id_].second);
    GetDensityMap(dmap, gt_loc_, base_sigma_);
    // For Verification.
    float sum_crowd = 0;
    for (int h=0; h<dmap.rows; h++) {
       for (int w=0; w<dmap.cols; w++) {
         sum_crowd = sum_crowd + *(dmap.ptr<float>(h,w));
       //std::cout<< *(densitymap.ptr<float>(h,w))<<" ";
       }
    }
    //LOG(INFO)<<"FILE: "<<root_folder + lines_[lines_id_].second<<", SUM_CROWED: "<<sum_crowd;
    Transform(dmap, &batch->label_, do_mirror, dmap_offset);
    // 2.4 Go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleData();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  DLOG(INFO) << "Generate Density map: " << dmap_time / 1000 << "ms.";
}

INSTANTIATE_CLASS(CrowdDataLayer);
REGISTER_LAYER_CLASS(CrowdData);

}  // namespace caffe
#endif  // USE_OPENCV
