#pragma once

#include <dlib/opencv.h>
#include "BFM.h"
#include "Renderer.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "ceres/cubic_interpolation.h"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

#include <opencv2/imgproc.hpp>

#include "RGBD_Image.h"

#include <unordered_set>

struct SparseCost {
	SparseCost(BFM bfm_, Vector2d observed_landmark_, int vertex_id_, int image_width_, int image_height_) :
		bfm{bfm_}, observed_landmark{observed_landmark_}, vertex_id{vertex_id_}, image_width{image_width_}, image_height{image_height_}
	{}
	template <typename T>
	bool operator()(T const* camera_rotation, T const* camera_translation, T const* shape_weights, T const* exp_weights, T* residuals) const {
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> shape_pca_basis_full(bfm.shape_pca_basis, 85764, 199);
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> exp_pca_basis_full(bfm.exp_pca_basis, 85764, 100);

		Matrix<T, 1, 3> face_model;

		face_model(0, 0) = T(bfm.shape_mean[vertex_id * 3]) + T(bfm.exp_mean[vertex_id * 3]);
		face_model(0, 1) = T(bfm.shape_mean[vertex_id * 3 + 1]) + T(bfm.exp_mean[vertex_id * 3 + 1]);
		face_model(0, 2) = T(bfm.shape_mean[vertex_id * 3 + 2]) + T(bfm.exp_mean[vertex_id * 3 + 2]);

		for (int i = 0; i < 199; i++) {
			T value = T(sqrt(bfm.shape_pca_var[i])) * shape_weights[i];
			face_model(0, 0) += T(shape_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(shape_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(shape_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}

		for (int i = 0; i < 100; i++) {
			T value = T(sqrt(bfm.exp_pca_var[i])) * exp_weights[i];
			face_model(0, 0) += T(exp_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(exp_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(exp_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}
				
		Matrix<T, 3, 1> translation = { camera_translation[0], camera_translation[1], camera_translation[2] };
		Quaternion<T> rotation = { camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3] };
		
		auto transformation_matrix = calculate_transformation_matrix<T>(translation, rotation);
		auto projection = calculate_transformation_perspective<T>(image_width, image_height, transformation_matrix, face_model);
		
		T w = projection(0, 3);
		T transformed_x = (projection(0, 0) / w + T(1)) / T(2)* T(image_width);
		T transformed_y = (projection(0, 1) / w + T(1)) / T(2)* T(image_height);
		transformed_y = T(image_height) - transformed_y;
		
		residuals[0] = transformed_x - T(observed_landmark[0]);
		residuals[1] = transformed_y - T(observed_landmark[1]);
		
		return true;
	}
private:
	const BFM bfm;
	const Vector2d observed_landmark;
	const int vertex_id;
	const int image_width;
	const int image_height;
};

struct DenseRGBCost {
	DenseRGBCost(BFM bfm_, cv::Mat* image_, int vertex_id_) :
		bfm{ bfm_ }, image{ image_ }, vertex_id{ vertex_id_ }
	{}
	template <typename T>
	bool operator()(T const* camera_rotation, T const* camera_translation, T const* shape_weights, T const* exp_weights, T const* color_weights, T* residuals) const {
		int image_width = image->cols;
		int image_height = image->rows;
		
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> shape_pca_basis_full(bfm.shape_pca_basis, 85764, 199);
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> exp_pca_basis_full(bfm.exp_pca_basis, 85764, 100);
	    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> color_pca_basis_full(bfm.color_pca_basis, 85764, 199);
        
		Matrix<T, 1, 3> face_model;
		Matrix<T, 1, 3> face_color;

		face_model(0, 0) = T(bfm.shape_mean[vertex_id * 3]) + T(bfm.exp_mean[vertex_id * 3]);
		face_model(0, 1) = T(bfm.shape_mean[vertex_id * 3 + 1]) + T(bfm.exp_mean[vertex_id * 3 + 1]);
		face_model(0, 2) = T(bfm.shape_mean[vertex_id * 3 + 2]) + T(bfm.exp_mean[vertex_id * 3 + 2]);

		face_color(0, 0) = T(bfm.color_mean[vertex_id * 3]);
		face_color(0, 1) = T(bfm.color_mean[vertex_id * 3 + 1]);
		face_color(0, 2) = T(bfm.color_mean[vertex_id * 3 + 2]);

		for (int i = 0; i < 199; i++) {
			T value = T(sqrt(bfm.shape_pca_var[i])) * shape_weights[i];
			face_model(0, 0) += T(shape_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(shape_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(shape_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}

		for (int i = 0; i < 100; i++) {
			T value = T(sqrt(bfm.exp_pca_var[i])) * exp_weights[i];
			face_model(0, 0) += T(exp_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(exp_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(exp_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}

		for (int i = 0; i < 199; i++) {
			T value = T(sqrt(bfm.color_pca_var[i])) * color_weights[i];
			face_color(0, 0) += T(color_pca_basis_full(vertex_id * 3, i)) * value;
			face_color(0, 1) += T(color_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_color(0, 2) += T(color_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}


		Matrix<T, 3, 1> translation = { camera_translation[0], camera_translation[1], camera_translation[2] };
		Quaternion<T> rotation = { camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3] };

		auto transformation_matrix = calculate_transformation_matrix<T>(translation, rotation);
		auto projection = calculate_transformation_perspective<T>(image_width, image_height, transformation_matrix, face_model);

		T w = projection(0, 3);
		T transformed_x = (projection(0, 0) / w + T(1)) / T(2) * T(image_width);
		T transformed_y = (projection(0, 1) / w + T(1)) / T(2) * T(image_height);
		transformed_y = T(image_height) - transformed_y;

		ceres::Grid2D<uchar, 3> grid(image->ptr(0), 0, image->rows, 0, image->cols);
		ceres::BiCubicInterpolator<ceres::Grid2D<uchar, 3>> interpolator(grid);
		T observed_colour[3];
		interpolator.Evaluate(transformed_y, transformed_x, &observed_colour[0]);

		residuals[0] = face_color(0, 0) * 255.0 - T(observed_colour[2]);
		residuals[1] = face_color(0, 1) * 255.0 - T(observed_colour[1]);
		residuals[2] = face_color(0, 2) * 255.0 - T(observed_colour[0]);

		return true;
	}
private:
	const BFM bfm;
	const cv::Mat* image;
	const int vertex_id;
};


struct DenseRGBDepthCost {
	DenseRGBDepthCost(BFM bfm_, RGBD_Image* rgbd_, int triangle_id_, MatrixXd* transformed_vertices_
	,double model_depth_min_
	,double model_depth_max_
	,double image_depth_min_
	,double image_depth_max_) :
		bfm{ bfm_ }, rgbd{ rgbd_ }, triangle_id{ triangle_id_ }, transformed_vertices{ transformed_vertices_ },
		model_depth_min{model_depth_min_},
		model_depth_max{model_depth_max_},
		image_depth_min{image_depth_min_},
		image_depth_max{image_depth_max_}
	{}
	template <typename T>
	bool operator()(T const* camera_rotation, T const* camera_translation, T const* shape_weights, T const* exp_weights, T const* color_weights, T* residuals) const {
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> color_pca_basis_full(bfm.color_pca_basis, 85764, 199);
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> shape_pca_basis_full(bfm.shape_pca_basis, 85764, 199);
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> exp_pca_basis_full(bfm.exp_pca_basis, 85764, 100);

		Matrix<T, 3, 3> face_color;
		Matrix<T, 1, 3> face_model;

		Matrix<T, 3, 3> positions;
		Matrix<double, 3, 2> positions_double;
		cv::Point pts[1][3];
        
        std::cout << "O1" << std::endl;
		for (int t = 0; t < 3; t++) {
			int vertex_id = bfm.triangles[triangle_id + t * 56572];

			face_color(t, 0) = T(bfm.color_mean[vertex_id * 3]);
			face_color(t, 1) = T(bfm.color_mean[vertex_id * 3 + 1]);
			face_color(t, 2) = T(bfm.color_mean[vertex_id * 3 + 2]);

			face_model(0, 0) = T(bfm.shape_mean[vertex_id * 3]) + T(bfm.exp_mean[vertex_id * 3]);
			face_model(0, 1) = T(bfm.shape_mean[vertex_id * 3 + 1]) + T(bfm.exp_mean[vertex_id * 3 + 1]);
			face_model(0, 2) = T(bfm.shape_mean[vertex_id * 3 + 2]) + T(bfm.exp_mean[vertex_id * 3 + 2]);

            std::cout << "O2" << std::endl;
			for (int i = 0; i < 199; i++) {
				T value = color_weights[i];
				face_color(t, 0) += T(color_pca_basis_full(vertex_id * 3, i)) * value;
				face_color(t, 1) += T(color_pca_basis_full(vertex_id * 3 + 1, i)) * value;
				face_color(t, 2) += T(color_pca_basis_full(vertex_id * 3 + 2, i)) * value;
			}
            std::cout << "O2.1" << std::endl;

			for (int i = 0; i < 199; i++) {
				T value = shape_weights[i];
				face_model(0, 0) += T(shape_pca_basis_full(vertex_id * 3, i)) * value;
				face_model(0, 1) += T(shape_pca_basis_full(vertex_id * 3 + 1, i)) * value;
				face_model(0, 2) += T(shape_pca_basis_full(vertex_id * 3 + 2, i)) * value;
			}

			for (int i = 0; i < 100; i++) {
				T value = exp_weights[i];
				face_model(0, 0) += T(exp_pca_basis_full(vertex_id * 3, i)) * value;
				face_model(0, 1) += T(exp_pca_basis_full(vertex_id * 3 + 1, i)) * value;
				face_model(0, 2) += T(exp_pca_basis_full(vertex_id * 3 + 2, i)) * value;
			}

			Matrix<T, 3, 1> translation = { camera_translation[0], camera_translation[1], camera_translation[2] };
			Quaternion<T> rotation = { camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3] };

			auto transformation_matrix = calculate_transformation_matrix<T>(translation, rotation);
			auto projection = calculate_transformation_perspective<T>(rgbd->image.cols, rgbd->image.rows, transformation_matrix, face_model);

			T w = projection(0, 3);
			T transformed_x = (projection(0, 0) / w + T(1)) / T(2) * T(rgbd->image.cols);
			T transformed_y = (projection(0, 1) / w + T(1)) / T(2) * T(rgbd->image.rows);
			transformed_y = T(rgbd->image.rows) - transformed_y;
			T transformed_z = projection(0, 2);

			positions(t, 0) = transformed_x;
			positions(t, 1) = transformed_y;
			positions(t, 2) = (transformed_z - T(model_depth_min)) / T(model_depth_max - model_depth_min);

			double dw = (*transformed_vertices)(vertex_id, 3);
			positions_double(t, 0) = ((*transformed_vertices)(vertex_id, 0) / dw + 1) / 2.0 * rgbd->image.cols;
			positions_double(t, 1) = rgbd->image.rows - (((*transformed_vertices)(vertex_id, 1) / dw + 1) / 2.0 * rgbd->image.rows);
			pts[0][t] = cv::Point(positions_double(t, 0), positions_double(t, 1));
		}

        std::cout << "O3" << std::endl;
		// Calculate the triangle
		const cv::Point* points[1] = { pts[0] };
		int npoints = 3;
		cv::Mat1b mask(rgbd->image.rows, rgbd->image.cols, uchar(0));
		cv::fillPoly(mask, points, &npoints, 1, cv::Scalar(255));

		auto result = cv::mean(rgbd->image, mask);

		residuals[0] = ((face_color(0, 0) * 255.0) + (face_color(1, 0) * 255.0) + (face_color(2, 0) * 255.0)) / T(3.0) - T(result[2]);
		residuals[1] = ((face_color(0, 1) * 255.0) + (face_color(1, 1) * 255.0) + (face_color(2, 1) * 255.0)) / T(3.0) - T(result[1]);
		residuals[2] = ((face_color(0, 2) * 255.0) + (face_color(1, 2) * 255.0) + (face_color(2, 2) * 255.0)) / T(3.0) - T(result[0]);
		// +depth
		//T depth0 = (T(rgbd->get_depth((int)positions_double(0, 0) / 2, (int)positions_double(0, 1) / 2)) - T(image_depth_min)) / T(image_depth_max - image_depth_min);
		//T depth1 = (T(rgbd->get_depth((int)positions_double(1, 0) / 2, (int)positions_double(1, 1) / 2)) - T(image_depth_min)) / T(image_depth_max - image_depth_min);
		//T depth2 = (T(rgbd->get_depth((int)positions_double(2, 0) / 2, (int)positions_double(2, 1) / 2)) - T(image_depth_min)) / T(image_depth_max - image_depth_min);

		//residuals[3] = positions(0, 2) - depth0 + positions(1, 2) - depth1 + positions(2, 2) - depth2;

		return true;
	}
private:
	const BFM bfm;
	RGBD_Image* rgbd;
	const int triangle_id;
	const MatrixXd* transformed_vertices;
	double model_depth_min;
	double model_depth_max;
	double image_depth_min;
	double image_depth_max;
};


struct DenseColorCost {
	DenseColorCost(BFM bfm_, cv::Mat* image_, int triangle_id_, int x_, int y_, MatrixXd* transformed_vertices_) :
		bfm{ bfm_ }, image{ image_ }, triangle_id{ triangle_id_ }, x{ x_ }, y{ y_ }, transformed_vertices{ transformed_vertices_ }
	{}
	template <typename T>
	bool operator()(T const* color_weights, T* residuals) const {
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> color_pca_basis_full(bfm.color_pca_basis, 85764, 199);

        std::cout << "gandon 1" << std::endl;
		Matrix<T, 3, 3> face_color;
		Matrix<double, 3, 2> positions;

        std::cout << "gandon 2" << std::endl;
		for (int t = 0; t < 3; t++) {
			int vertex_id = bfm.triangles[triangle_id + t * 56572];

			face_color(t, 0) = T(bfm.color_mean[vertex_id * 3]);
			face_color(t, 1) = T(bfm.color_mean[vertex_id * 3 + 1]);
			face_color(t, 2) = T(bfm.color_mean[vertex_id * 3 + 2]);

			for (int i = 0; i < COLOR_COUNT; i++) {
				T value = T(sqrt(bfm.color_pca_var[i])) * color_weights[i];
				face_color(t, 0) += T(color_pca_basis_full(vertex_id * 3, i)) * value;
				face_color(t, 1) += T(color_pca_basis_full(vertex_id * 3 + 1, i)) * value;
				face_color(t, 2) += T(color_pca_basis_full(vertex_id * 3 + 2, i)) * value;
			}

			double w = (*transformed_vertices)(vertex_id, 3);
			positions(t, 0) = ((*transformed_vertices)(vertex_id, 0) / w + 1) / 2.0 * image->cols;
			positions(t, 1) = image->rows - (((*transformed_vertices)(vertex_id, 1) / w + 1) / 2.0 * image->rows);
		}

        std::cout << "gandon 3" << std::endl;
		double x1 = positions(0, 0);
		double y1 = positions(0, 1);

		double x2 = positions(1, 0);
		double y2 = positions(1, 1);

		double x3 = positions(2, 0);
		double y3 = positions(2, 1);

		//std::cout << "(" << x << "," << y << "): " << x1 << " " << y1 << " - " << x2 << " " << y2 << " - " << x3 << " " << y3 << " triangle: " << triangle_id << "\n";

		double alpha1 = ((y2 - y3) * (x + 0.5 - x3) + (x3 - x2) * (y + 0.5 - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
		double alpha2 = ((y3 - y1) * (x + 0.5 - x3) + (x1 - x3) * (y + 0.5 - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
		double alpha3 = 1.0 - alpha1 - alpha2;

		T barycentricCoordinates[] = {
			T(alpha1),
			T(alpha2),
			T(alpha3)
		};

		auto result = image->at<cv::Vec3b>(y, x);

		residuals[0] = ((face_color(0, 0) * 255.0) * barycentricCoordinates[0] + (face_color(1, 0) * 255.0) * barycentricCoordinates[1] + (face_color(2, 0) * 255.0) * barycentricCoordinates[2]) - T((int)result[2]);
		residuals[1] = ((face_color(0, 1) * 255.0) * barycentricCoordinates[0] + (face_color(1, 1) * 255.0) * barycentricCoordinates[1] + (face_color(2, 1) * 255.0) * barycentricCoordinates[2]) - T((int)result[1]);
		residuals[2] = ((face_color(0, 2) * 255.0) * barycentricCoordinates[0] + (face_color(1, 2) * 255.0) * barycentricCoordinates[1] + (face_color(2, 2) * 255.0) * barycentricCoordinates[2]) - T((int)result[0]);


		return true;
	}
private:
	const BFM bfm;
	const cv::Mat* image;
	const int triangle_id;
	const MatrixXd* transformed_vertices;
	const int x, y;
};

class DenseOptimizer {
public:
	void optimize(cv::Mat image, std::vector<dlib::full_object_detection> detected_landmarks);
    //void optimize_rgbd(RGBD_Image* rgbd, std::vector<dlib::full_object_detection> detected_landmarks);
private: 
	//void render(cv::Mat image, BFM bfm, Parameters params, Eigen::Vector3d translation, double* rotation);
    void render(cv::Mat image, BFM bfm, Parameters params, Eigen::Vector3d translation, double* rotation, double fov, bool include_alternative = false);
	int render_number=0;

    cv::Mat albedo_render;
	cv::Mat triangle_render;
	double* alternative_colors;
};