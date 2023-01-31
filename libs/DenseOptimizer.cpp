#include "DenseOptimizer.h"
#include "RegularizationTerm.h"

#define SHAPE_REGULARIZATION_WEIGHT 5
#define EXPRESSION_REGULARIZATION_WEIGHT 5
#define COLOR_REGULARIZATION_WEIGHT 400

void DenseOptimizer::optimize(cv::Mat image, std::vector<dlib::full_object_detection> detected_landmarks)
{
	BFM bfm = bfm_setup();

	//Variables that we are going to optimize for
	double* rotation = new double[4];
	rotation[0] = 1;
	rotation[1] = 0;
	rotation[2] = 0;
	rotation[3] = 0;
	Eigen::Vector3d translation = { 0, 0, -400 };
	Parameters params = bfm_mean_params();
    std::cout << "aaa" << std::endl;
	//Learn position+rotation (using landmarks)
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 4;
		options.minimizer_progress_to_stdout = true;

        std::cout << "cols: " << image.cols << " rows: " << image.rows << std::endl;

		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };
            std::cout << "x: " << detected_landmarks[0].part(j).x() << " y: " << detected_landmarks[0].part(j).y() << std::endl;

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], image.cols, image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);
		for (int j = 0; j < 199; j++) {
			sparse_problem.SetParameterUpperBound(params.shape_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.shape_weights.data(), j, -1);
		}

		for (int j = 0; j < 100; j++) {
			sparse_problem.SetParameterUpperBound(params.exp_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.exp_weights.data(), j, -1);
		}

		// keep the shape and expression constant
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());

        std::cout << "aaa1" << std::endl;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

        std::cout << "aaa2" << std::endl;
	    //bfm_create_obj(bfm, params);
		//render(image, bfm, params, translation, rotation);
	}

    std::cout << "bbb" << std::endl;
	//Learn position + rotation + shape weights + expression weights
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 4;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, 199, 199>(
				new ShapeCostFunction(bfm)
				);
		sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, 100, 100>(
				new ExpressionCostFunction(bfm)
				);
		sparse_problem.AddResidualBlock(expression_cost, NULL, params.exp_weights.data());



		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], image.cols, image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);
		for (int j = 0; j < 199; j++) {
			sparse_problem.SetParameterUpperBound(params.shape_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.shape_weights.data(), j, -1);
		}

		for (int j = 0; j < 100; j++) {
			sparse_problem.SetParameterUpperBound(params.exp_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.exp_weights.data(), j, -1);
		}
		

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		//render(image, bfm, params, translation, rotation);
	}

    //std::cout << "Writing bfm..." << std::endl;
	bfm_create_obj(bfm, params);
    std::cout << "ccc" << std::endl;
	//Learn color weights only
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		//options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
		options.num_threads = 1;
		options.minimizer_progress_to_stdout = true;
        std::cout << "ccc1" << std::endl;

		ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, 199, 199>(
				new ColorCostFunction(bfm)
				);
		sparse_problem.AddResidualBlock(color_cost, NULL,params.col_weights.data());
        std::cout << "ccc2" << std::endl;


		//for (int j = 0; j < 28588; j++) {
		for (int j = 0; j < 200; j++) {
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBCost, 3, 4, 3, 199, 100, 199>(
				new DenseRGBCost(bfm, &image, j)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data(), params.col_weights.data());
		}
        std::cout << "ccc3" << std::endl;
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());
		sparse_problem.SetParameterBlockConstant(rotation);
		sparse_problem.SetParameterBlockConstant(translation.data());
        std::cout << "ccc4" << std::endl;
		for (int j = 0; j < 199; j++) {
			sparse_problem.SetParameterUpperBound(params.col_weights.data(), j, 3);
			sparse_problem.SetParameterLowerBound(params.col_weights.data(), j, -3);
		}
        std::cout << "ccc5" << std::endl;

		ceres::Solver::Summary summary;
        std::cout << "ccc6" << std::endl;
		ceres::Solve(options, &sparse_problem, &summary);
        std::cout << "ccc7" << std::endl;
		std::cout << summary.BriefReport() << std::endl;

        std::cout << "ddd" << std::endl;
		//render(image, bfm, params, translation, rotation);
	}

    std::cout << "eee" << std::endl;
	bfm_create_obj(bfm, params);
}
/**
void DenseOptimizer::optimize_rgbd(RGBD_Image* rgbd, std::vector<dlib::full_object_detection> detected_landmarks)
{
    BFM bfm = bfm_setup();

	//Variables that we are going to optimize for
	double* rotation = new double[4];
	rotation[0] = 1;
	rotation[1] = 0;
	rotation[2] = 0;
	rotation[3] = 0;
	Eigen::Vector3d translation = { 0, 0, -400 };
	Parameters params = bfm_mean_params();

	//Learn position+rotation (using landmarks)
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 8;
		options.minimizer_progress_to_stdout = true;

		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], rgbd->image.cols, rgbd->image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);

		// keep the shape and expression constant
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		//render(rgbd->image, bfm, params, translation, rotation);
	}

	std::cout << translation << "\n";

	//Learn position + rotation + shape weights + expression weights
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 8;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, 199, 199>(
			new ShapeCostFunction(bfm, SHAPE_REGULARIZATION_WEIGHT)
			);
		sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, 100, 100>(
			new ExpressionCostFunction(bfm, EXPRESSION_REGULARIZATION_WEIGHT)
			);
		sparse_problem.AddResidualBlock(expression_cost, NULL, params.exp_weights.data());

		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], rgbd->image.cols, rgbd->image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);


		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		//render(rgbd->image, bfm, params, translation, rotation);
	}

    std::cout << "Bad shandau: " << std::endl;
	//Learn color + depth
	//{
	//	auto vertices = get_vertices(bfm, params);
	//	Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
	//	auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
	//	auto transformed_vertices = calculate_transformation_perspective(rgbd->image.cols, rgbd->image.rows, transformation_matrix, vertices);

	//	ceres::Problem sparse_problem;
	//	ceres::Solver::Options options;
	//	//options.linear_solver_type = ceres::DENSE_QR;
	//	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//	options.num_threads = 8;
	//	options.minimizer_progress_to_stdout = true;
	//	options.max_linear_solver_iterations = 1;

    //    std::cout << "S1" << std::endl;
	//	ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, 199, 199>(
	//		new ColorCostFunction(bfm, COLOR_REGULARIZATION_WEIGHT)
	//		);
	//	sparse_problem.AddResidualBlock(color_cost, NULL, params.col_weights.data());

    //    std::cout << "S2" << std::endl;
	//	ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, 199, 199>(
	//		new ShapeCostFunction(bfm, SHAPE_REGULARIZATION_WEIGHT)
	//		);
	//	sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

	//	ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, 100, 100>(
	//		new ExpressionCostFunction(bfm, EXPRESSION_REGULARIZATION_WEIGHT)
	//		);
	//	sparse_problem.AddResidualBlock(expression_cost, NULL, params.exp_weights.data());

    //    std::cout << "S3" << std::endl;
	//	//double model_depth_min = 9999, model_depth_max = 0;
	//	//double image_depth_min = 9999, image_depth_max = 0;

	//	//for (int j = 0; j < 68; j++) {
	//	//	double image_depth = rgbd->get_depth(detected_landmarks[0].part(j).x() / 2, detected_landmarks[0].part(j).y() / 2);
	//	//	double model_depth = transformed_vertices(bfm.landmarks[j], 2);

	//	//	if (model_depth < model_depth_min) {
	//	//		model_depth_min = model_depth;
	//	//	}
	//	//	else if (model_depth > model_depth_max) {
	//	//		model_depth_max = model_depth;
	//	//	}

	//	//	if (image_depth < image_depth_min) {
	//	//		image_depth_min = image_depth;
	//	//	}
	//	//	else if (image_depth > image_depth_max) {
	//	//		image_depth_max = image_depth;
	//	//	}
	//	//}

	//	std::cout << model_depth_min << " - " << model_depth_max << "\n";
	//	std::cout << image_depth_min << " - " << image_depth_max << "\n";

	//	//for (int j = 0; j < 56572; j++) {
	//	for (int j = 0; j < 200; j++) {
    //        
	//		//ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBDepthCost, 4, 4, 3, 199, 100, 199>(
	//		ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBDepthCost, 3, 4, 3, 199, 100, 199>(
	//			new DenseRGBDepthCost(bfm, rgbd, j, &transformed_vertices, model_depth_min, model_depth_max, image_depth_min, image_depth_max)
	//			);
	//		sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data(), params.col_weights.data());
	//	}

	//	sparse_problem.SetParameterBlockConstant(rotation);
	//	sparse_problem.SetParameterBlockConstant(translation.data());

	//	ceres::Solver::Summary summary;
    //    std::cout << "S5" << std::endl;
	//	ceres::Solve(options, &sparse_problem, &summary);
    //    std::cout << "S6" << std::endl;
	//	std::cout << summary.BriefReport() << std::endl;

	//	//render(rgbd->image, bfm, params, translation, rotation);
	//}
    //

    //render(rgbd->image, bfm, params, translation, rotation, 45.0);
    //{
	//	auto vertices = get_vertices(bfm, params);
	//	Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
	//	auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
	//	auto transformed_vertices = calculate_transformation_perspective(rgbd->image.cols, rgbd->image.rows, transformation_matrix, vertices);

	//	ceres::Problem sparse_problem;
	//	ceres::Solver::Options options;
	//	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	//	options.num_threads = 8;
	//	options.minimizer_progress_to_stdout = true;

	//	ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, COLOR_COUNT, COLOR_COUNT>(
	//			new ColorCostFunction(bfm, COLOR_REGULARIZATION_WEIGHT)
	//			);
	//	sparse_problem.AddResidualBlock(color_cost, NULL,params.col_weights.data());

	//	for (int i = 0; i < triangle_render.rows; i++) {
	//		for (int j = 0; j < triangle_render.cols; j++) {
	//			auto p = triangle_render.data + (i * triangle_render.cols + j) * 3;
	//			int triangle_id = (0 << 24) | ((int)p[2] << 16) | ((int)p[1] << 8) | ((int)p[0]);
	//			if (triangle_id < 56572) {
	//				ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseColorCost, 3, COLOR_COUNT>(
	//					new DenseColorCost(bfm, &rgbd->image, triangle_id, j, i, &transformed_vertices)
	//					);
	//				sparse_problem.AddResidualBlock(cost_function, NULL, params.col_weights.data());
	//			}
	//		}
	//	}
    //}
	bfm_create_obj(bfm, params);
}


void DenseOptimizer::render(cv::Mat image, BFM bfm, Parameters params, Eigen::Vector3d translation, double* rotation, double fov, bool include_alternative)
{
    std::cout << 1 << std::endl;
	auto context = init_rendering_context(image.cols, image.rows);
	auto vertices = get_vertices(bfm, params);
    std::cout << 1.1 << std::endl;
	auto colors = get_colors(bfm, params);
    std::cout << 2 << std::endl;
	if (include_alternative) {
		for (int i = 0; i < 85764; i++) {
			colors[i] += alternative_colors[i];
			colors[i] /= 2.0;
		}
	}
    std::cout << 3 << std::endl;

	Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
	auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
	auto transformed_vertices = calculate_transformation_perspective(image.cols, image.rows, transformation_matrix, vertices);

    std::cout << 4 << std::endl;
	albedo_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false);
	triangle_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false, true);

    std::cout << 5 << std::endl;
	cv::imwrite("img_"+ std::to_string(render_number) + ".png", albedo_render);
	cv::imwrite("img_"+ std::to_string(render_number) + "_triangles.png", triangle_render);
	terminate_rendering_context();

	render_number++;
}

//void DenseOptimizer::render(cv::Mat image, BFM bfm, Parameters params, Eigen::Vector3d translation, double* rotation){
//    auto context = init_rendering_context(image.cols, image.rows);
//    auto vertices = get_vertices(bfm, params);
//    auto colors = get_colors(bfm, params);
//    Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
//    auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
//    auto transformed_vertices = calculate_transformation_perspective(image.cols, image.rows, transformation_matrix, vertices);
//
//    albedo_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false);
//    triangle_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false, true);
//
//    cv::imwrite("img_"+ std::to_string(render_number) + ".png", albedo_render);
//    cv::imwrite("img_"+ std::to_string(render_number) + "_triangles.png", triangle_render);
//    terminate_rendering_context();
//
//    render_number++;
//}
**/
