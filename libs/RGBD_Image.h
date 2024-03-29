#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <stdio.h>
#include <iostream>     
#include <fstream>
#include <utils/points.h>

/**
 * ICP optimizer - using Ceres for optimization.
 */
class RGBD_Image {
public:
	std::vector<Point3D> points;
	cv::Mat image;

	RGBD_Image(cv::Mat img, std::string depth_path) {
		this->image = img;
		//this->landmarks = DetectLandmarks(rgb_path, true, true);
	    //DetectLandmarks(this->image, this->landmarks);
		load_data(depth_path);
	};

	double get_depth(int x, int y) {
		assert(!this->points.empty());
		for(Point3D p : this->points){
			if((int)p.x == x && (int)p.y == y) return p.z;
		}
		return 0.; 
	}
	//Method to load the binary data into a vector of 3D points where x and y are already projected into 2D	
	void load_data(const std::string path){  
		std::ifstream inBinFile; 
		inBinFile.open(path, std::ios::out | std::ios::binary);
		// Define projection matrix from 3D to 2D
		//P matrix is in camera_info.yaml*/
		Eigen::Matrix<float, 3,4> P;
			P <<  1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721, 536.2206151001486, 0, 0, 0, 1, 0;

		for(int i =0; i<960;i++){
			for(int j =0; j<540;j++){
				double pointx;
				double pointy;
				double pointz;
				inBinFile.read(reinterpret_cast<char*>(&pointx), sizeof pointx);
				inBinFile.read(reinterpret_cast<char*>(&pointy), sizeof pointy);
				inBinFile.read(reinterpret_cast<char*>(&pointz), sizeof pointz);

				Eigen::Vector4f homogeneous_point(pointx, pointy, pointz,1);
				Eigen::Vector3f output = P * homogeneous_point;

				output[0] /= output[2];
				output[1] /= output[2];

				Point3D p{output[0],output[1],output[2]};
				//If there is no depth data at this pixel 
				if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)){   
					continue;
				}

				points.push_back(p);
			}
		}
		inBinFile.close();
	}
};
