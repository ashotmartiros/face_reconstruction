#include <iostream>

#include "LandmarkDetection.h"
#include "DenseOptimizer.h"
#include "RGBD_Image.h"

int main(int argc, char** argv) {
	DenseOptimizer optimizer;

	cv::Mat img = cv::imread("../data/frontal_face.jpg");
	//cv::Mat img = cv::imread("../data/putin.png");
	std::vector<dlib::full_object_detection> landmarks;
	DetectLandmarks(img, landmarks);
	optimizer.optimize(img, landmarks);

    //cv::Mat img = cv::imread("../data/RGBD_data/Test1/000_00_image.png");
	//std::vector<dlib::full_object_detection> landmarks;
	//DetectLandmarks(img, landmarks);
    //RGBD_Image *rgbd = new RGBD_Image(img, "../data/RGBD_data/Test1/000_00_cloud.bin");
	//optimizer.optimize_rgbd(rgbd, landmarks);
	return 0;
}
