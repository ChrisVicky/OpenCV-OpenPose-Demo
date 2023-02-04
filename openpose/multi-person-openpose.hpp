#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

#include<iostream>
#include<chrono>
#include<random>
#include<set>
#include<cmath>

#include "../include/settings.hpp"
#include "../logsrc/loguru.hpp"


////////////////////////////////
struct KeyPoint{
	KeyPoint(cv::Point point,float probability){
		this->id = -1;
		this->point = point;
		this->probability = probability;
	}

	int id;
	cv::Point point;
	float probability;
};

struct ValidPair{
	ValidPair(int aId,int bId,float score){
		this->aId = aId;
		this->bId = bId;
		this->score = score;
	}

	int aId;
	int bId;
	float score;
};

/**
 * @brief 跑一次网络，输出含有标记的图片
 * @param input cv::Mat
 * @param s     Settings
 * @return  	cv::Mat
 */
cv::Mat forwardNet(cv::Mat input, Settings s);

/**
 * @brief  通过设置初始化网络
 * @param s
 * @return cv::dnn::Net
 */
cv::dnn::Net initNet(Settings s);
