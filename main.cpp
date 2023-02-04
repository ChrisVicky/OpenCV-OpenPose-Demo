/* Copyright (C) 
 * 2023 - Christopher Liu
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * @file main.cpp
 * @brief 
 * 	A DEMO that demonstrates the usage of OpenCV cv::dnn::Net that 
 * 	implement OpenPOSE to detect Human Pose
 * 	- APIs are encapsulated in ./openpose/ dir
 * 	- Configurations are controlled by ./default.xml file
 * 	- Logs are supported via ./logsrc/ reference: https://github.com/emilk/loguru
 * @author Christopher Liu
 * @version 1.0
 * @date 2023-02-04
 */

#include<opencv2/core.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/persistence.hpp>
#include<opencv2/videoio.hpp>

#include "./logsrc/loguru.hpp"
#include "./openpose/multi-person-openpose.hpp"
#include "./include/settings.hpp"

#include<iostream>

Settings s;

int main(int argc, char *argv[]){

	std::string Keys = 
		"{ h help         | false       | print this help message }"
		"{@settings s     | default.xml | input setting files}"
		;
	cv::CommandLineParser parser(argc, argv, Keys);

	if (parser.get<bool>("help")){
		std::cout << "A sample app to demonstrate human or hand pose detection with a pretrained OpenPose dnn." << std::endl;
		parser.printMessage();
		return 0;
	}

	/* Read Settings */
	Settings s;
	const std::string settings_file = parser.get<std::string>(0);
	std::cout << "Setting File: " << settings_file << std::endl;

	cv::FileStorage fs(settings_file, cv::FileStorage::READ);
	if(!fs.isOpened()){
		std::cout << "ERROR! Could not open configuration " << settings_file << std::endl;
		parser.printMessage();
		return -1;
	}

	fs["Settings"] >> s;
	fs.release(); 
	if(!s.goodInput){
		std::cout << "ERROR! Invalid input detected. Application Stopping." << std::endl;
		parser.printMessage();
		return -1;
	}

	LOG_F(INFO, "Program Start");

	cv::dnn::Net net = initNet(s);

	cv::Mat input;
	cv::Mat show;
	if(s.isCamera){
		cv::VideoCapture cap(0);
		bool LOOP = true;
		while(LOOP){
			cap >> input;
			show = forwardNet(input, s);
			cv::putText(show, "Press 'q' to Exit", cv::Point(50,50), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(255,255,255), 2);
			imshow("Results", show);
			char key = cv::waitKey(1);
			if(key == 'q'){
				LOOP = false;
			}
		}
	}else{
		input = cv::imread(s.imageFile, cv::IMREAD_COLOR);
		show = forwardNet(input,s);
		imshow("Results", show);
    imwrite("Result.png", show);
		cv::waitKey();
	}
	return 0;
}
