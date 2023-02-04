#ifndef __SETTINGS__H__
#define __SETTINGS__H__

#include "../logsrc/loguru.hpp"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp> 
#include <opencv4/opencv2/core/persistence.hpp> 
#include <opencv2/core/types.hpp>

class Settings{
	public:
		// Default is an Error Input
		Settings():goodInput(false){}
		void write(cv::FileStorage& fs) const {
			fs << "{";
			fs << "modelTxt" << modelTxt;
			fs << "modelBin" << modelBin;
			fs << "imageFile" << imageFile;
			fs << "dataset" << dataset;

			fs << "W_in" << W_in;
			fs << "H_in" << H_in;

			fs << "thresh" << thresh;
			fs << "scale" << scale;

			fs << "logPath" << logPath;
			fs << "device" << device;
			fs << "}";
		}
		void read(const cv::FileNode& node){
			// Calibration
			node["modelTxt"] >> modelTxt;
			node["modelBin"] >> modelBin;
			node["imageFile"] >> imageFile;
			node["dataset"] >> dataset;

			node["W_in"] >> W_in;
			node["H_in"] >> H_in;

			node["thresh"] >> thresh;
			node["scale"] >> scale;

			node["logPath"] >> logPath;

			node["device"] >> device;

			validate();
		}

		void validate(){
			goodInput = true;
			if(logPath.empty()){
				LOG_F(INFO, "No Log Path Specified, Log to stderr Only");
			}else{
	      loguru::add_file(logPath.c_str(), loguru::Truncate, loguru::Verbosity_MAX);
			  LOG_F(INFO, "Log to File '%s'",logPath.c_str());
			}
			if(modelTxt.empty() || modelBin.empty() || dataset.empty()){
				LOG_F(ERROR, "Model Configuration Crashed");
				goodInput = false;
			}else{
				LOG_F(INFO, "model type: %s",dataset.c_str());
			}
			if(device!="CPU" && device != "GPU"){
        LOG_F(ERROR, "Device '%s' Not Supported",device.c_str());
				goodInput = false;
			}
			/* Parameters Reference: 
			 * https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp 
			 */
			if(dataset=="COCO"){
				nPoints = 18;
				keypointsMapping = {
					"Nose", "Neck",
					"R-Sho", "R-Elb", "R-Wr",
					"L-Sho", "L-Elb", "L-Wr",

					"R-Hip", "R-Knee", "R-Ank",
					"L-Hip", "L-Knee", "L-Ank",
					"R-Eye", "L-Eye", "R-Ear", "L-Ear"
				};
				mapIdx= {
					{31,32}, {39,40}, {33,34}, {35,36}, {41,42}, {43,44},
					{19,20}, {21,22}, {23,24}, {25,26}, {27,28}, {29,30},
					{47,48}, {49,50}, {53,54}, {51,52}, {55,56}, {37,38},
					{45,46}
				};
				posePairs = {
					{1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
					{1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
					{1,0}, {0,14}, {14,16}, {0,15}, {15,17}, {2,17},
					{5,16}
				};
			}else if(dataset=="BODY_25"){
				nPoints = 25;
				keypointsMapping = {
					"Nose", "Neck", 
					"RShoulder", "RElbow","RWrist",
					"LShoulder", "LElbow", "LWrist",
					"MidHip", 
					"RHip", "RKnee", "RAnkle",
					"LHip", "LKnee", "LAnkle",
					"REye", "LEye",
					"REar", "LEar",
					"LBigToe", "LSmallToe",
					"LHeel", "RBigToe",
					"RSmallToe", "RHeel",
					"Background" // 第 index=25 却不适用
				};
				mapIdx = {
					{26, 27}, {40, 41}, {48, 49}, {42, 43}, {44, 45},
					{50, 51}, {52, 53}, {32, 33}, {28, 29}, {30, 31},
					{34, 35}, {36, 37}, {38, 39}, {56, 57}, {58, 59},
					{62, 63}, {60, 61}, {64, 65},
					//{46, 47}, {54, 55},
					{66, 67}, {68, 69}, {70, 71}, {72, 73}, {74, 75},
					{76, 77},
				};
				posePairs = {
					{1,8}, 	{1,2}, 	{1,5}, 	{2,3}, 	{3,4},
					{5,6}, 	{6,7}, 	{8,9}, 	{9,10}, {10,11},
					{8,12}, {12,13},{13,14},{1,0}, 	{0,15}, 
					{15,17},{0,16}, {16,18},
					//{2,17}, {5,18},
					{14,19},{19,20},{14,21},{11,22},{22,23}, 
					{11,24},
				};
			}else{
        LOG_F(ERROR, "Model Type '%s' Not Supported",dataset.c_str());
				goodInput = false;
			}

			if(imageFile=="0"){
        LOG_F(INFO, "Use Camera");
				isCamera = true;
			}else{
        LOG_F(INFO, "Use Image");
				isCamera = false;
			}
		}

	public:
		std::string modelTxt;    // model configuration, e.g. hand/pose.prototxt 
		std::string modelBin;    // model weights, e.g. hand/pose_iter_102000.caffemodel 

		std::string imageFile;   // path to image file (containing a single person, or hand) 

		std::string device; 	 	// CPU or GPU
		std::string dataset;     // specify what kind of model was trained. It could be (COCO, MPI, HAND) depends on dataset.

		int W_in;           // Preprocess input image by resizing to a specific width. 
		int H_in;           // Preprocess input image by resizing to a specific height. 

		float thresh;       // threshold or confidence value for the heatmap
		float scale;        // scale for blob 

		bool goodInput;     // true if all inputs are valid
		bool isCamera;

		std::string logPath;  // Log Output Path (loguru)

		std::vector<std::pair<int,int>> mapIdx; 
		std::vector<std::pair<int,int>> posePairs;
		std::vector<std::string> keypointsMapping;
		int nPoints;
};

static inline void read(const cv::FileNode& node, Settings& s, const Settings& default_value = Settings()){
	if(node.empty()){
		s = default_value;
	} else {
		s.read(node);
	}
	return ;
}
#endif
