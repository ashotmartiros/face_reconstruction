#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

//void drawPolyline(cv::Mat &image, dlib::full_object_detection landmarks, int start, int end, bool isClosed=false){
//    std::vector<cv::Point> points;
//    for(int i=start; i<=end; i++){
//        points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
//    }
//    cv::polylines(image, points, isClosed, cv::Scalar(0, 255, 255), 2, 16);
//}
//
//void drawPolylines(cv::Mat &image, dlib::full_object_detection landmarks){
//    drawPolyline(image, landmarks, 0, 16);              //jaw line
//    drawPolyline(image, landmarks, 17, 21);             //left eyebrow
//    drawPolyline(image, landmarks, 22, 26);             //right eyebrow
//    drawPolyline(image, landmarks, 27, 30);             //Nose bridge
//    drawPolyline(image, landmarks, 30, 35, true);       //lower nose
//    drawPolyline(image, landmarks, 36, 41, true);       //left eye
//    drawPolyline(image, landmarks, 42, 47, true);       //right eye
//    drawPolyline(image, landmarks, 48, 59, true);       //outer lip
//    drawPolyline(image, landmarks, 60, 67, true);       //inner lip
//}

void DetectLandmarks(cv::Mat& img, std::vector<dlib::full_object_detection>& landmarks) 
{
    try
    {
        // Load face detection and landmark detection models.
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor predictor;
        dlib::deserialize("/Users/ashot/workspace/tum/3d/face-reconstruction/src/shape_predictor_68_face_landmarks.dat") >> predictor;

        dlib::cv_image<dlib::bgr_pixel> cimg(img);
        std::vector<dlib::rectangle> faces = detector(cimg);
        // Find landmarks of each face.
        for (unsigned long i = 0; i < faces.size(); ++i) {
            dlib::full_object_detection landmark = predictor(cimg, faces[i]);
            //drawPolylines(img, landmark);
            landmarks.push_back(landmark);
        }

        //cv::imwrite("landmark_output.jpg", img);

        //for (unsigned long i = 0; i < faces.size(); ++i)
        //    for (unsigned long j = 0; j < pose_model(cimg, faces[i]).num_parts(); ++j)
        //        std::cout << pose_model(cimg, faces[i]).part(j).x() << std::endl;

        //    if (presentLandmarks) {
        //        // Display it all on the screen
        //        win.clear_overlay();
        //        win.set_image(cimg);
        //        win.add_overlay(dlib::render_face_detections(shapes));

        //        if (holdImg) {
        //            int key = cv::waitKey(10000);
        //            if (key == 27) {
        //                break;
        //            }
        //        }
        //    }
    }
    catch (dlib::serialization_error& e)
    {
        std::cout << "You need dlib's default face landmarking model file to run this example.\n";
        std::cout << "You can get it from the following URL: \n";
        std::cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n";
        std::cout << "\n" << e.what() << "\n";
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << "\n";
    }
}
