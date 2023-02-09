#include "multi-person-openpose.hpp"
#include <opencv4/opencv2/highgui.hpp>
////////////////////////////////
std::ostream& operator << (std::ostream& os, const KeyPoint& kp)
{
	os << "Id:" << kp.id << ", Point:" << kp.point << ", Prob:" << kp.probability << std::endl;
	return os;
}

std::ostream& operator << (std::ostream& os, const ValidPair& vp)
{
	os << "A:" << vp.aId << ", B:" << vp.bId << ", score:" << vp.score << std::endl;
	return os;
}

template < class T > std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
	os << "[";
	bool first = true;
	for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
	{
		if(!first) os << ",";
		os << " " << *ii;
	}
	os << "]";
	return os;
}

template < class T > std::ostream& operator << (std::ostream& os, const std::set<T>& v)
{
	os << "[";
	bool first = true;
	for (typename std::set<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
	{
		if(!first) os << ",";
		os << " " << *ii;
	}
	os << "]";
	return os;
}
//////////////////////////////

int nPoints;
std::vector<std::string> keypointsMapping;
std::vector<std::pair<int,int>> mapIdx;
std::vector<std::pair<int,int>> posePairs;

std::vector<cv::Scalar> colors;

/**
 * @brief 对于每个 body part 的 heatMap 找到其中可能的 ketPoints
 * @param probMap 	-> 某个 body part 的 heatMap
 * @param threshold 	-> 大于它就认为是
 * @param keyPoints 	-> Return 值
 */
void getKeyPoints(cv::Mat& probMap,double threshold,std::vector<KeyPoint>& keyPoints){
	cv::Mat smoothProbMap;
	cv::GaussianBlur( probMap, smoothProbMap, cv::Size( 3, 3 ), 0, 0 );

	cv::Mat maskedProbMap;
	cv::threshold(smoothProbMap,maskedProbMap,threshold,255,cv::THRESH_BINARY);

	maskedProbMap.convertTo(maskedProbMap,CV_8U,1);

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(maskedProbMap,contours,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);

	for(int i = 0; i < contours.size();++i){
		cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows,smoothProbMap.cols,smoothProbMap.type());

		cv::fillConvexPoly(blobMask,contours[i],cv::Scalar(1));

		double maxVal;
		cv::Point maxLoc;

		cv::minMaxLoc(smoothProbMap.mul(blobMask),0,&maxVal,0,&maxLoc);

		keyPoints.push_back(KeyPoint(maxLoc, probMap.at<float>(maxLoc.y,maxLoc.x)));
	}
}

/**
 * @brief 生成 nColors 个 color 每个 color 之间的距离是一定的
 * @param colors 	-> 返回值，生成的颜色序列
 * @param nColors 	-> nColors 个不同的颜色
 */
void populateColorPalette(std::vector<cv::Scalar>& colors,int nColors){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis1(64, 200);
	std::uniform_int_distribution<> dis2(100, 255);
	std::uniform_int_distribution<> dis3(100, 255);

	for(int i = 0; i < nColors;++i){
		colors.push_back(cv::Scalar(dis1(gen),dis2(gen),dis3(gen)));
	}
}

/**
 * @brief 将网络输出分成 nParts 个图
 * 	前 nBody 个是 heatMap 表示每个 body part 在图中的可能位置
 * 	后 nParts - nBody 个是 PAF 图 表示关节的可能方向
 * @param netOutputBlob 	-> Network Output
 * @param targetSize 		-> Size(hxw) 输入图片的 hxw
 * @param netOutputParts 	-> Vector<Mat> (Return) -> heatMap
 */
void splitNetOutputBlobToParts(cv::Mat& netOutputBlob,const cv::Size& targetSize,std::vector<cv::Mat>& netOutputParts){
	int nParts = netOutputBlob.size[1];
	int h = netOutputBlob.size[2];
	int w = netOutputBlob.size[3];

	for(int i = 0; i< nParts;++i){
		cv::Mat part(h, w, CV_32F, netOutputBlob.ptr(0,i));

		cv::imshow(cv::format("HeatMap %d", i), part);
		cv::waitKey();
		cv::Mat resizedPart;

		cv::resize(part,resizedPart,targetSize);

		netOutputParts.push_back(resizedPart);
	}
}

void populateInterpPoints(const cv::Point& a,const cv::Point& b,int numPoints,std::vector<cv::Point>& interpCoords){
	float xStep = ((float)(b.x - a.x))/(float)(numPoints-1);
	float yStep = ((float)(b.y - a.y))/(float)(numPoints-1);

	interpCoords.push_back(a);

	for(int i = 1; i< numPoints-1;++i){
		interpCoords.push_back(cv::Point(a.x + xStep*i,a.y + yStep*i));
	}

	interpCoords.push_back(b);
}


/**
 * @brief 分析 body keypoints 之间的关系以及 PAF 得到可能的点对
 * @param netOutputParts 	-> 提供 PAF 
 * @param detectedKeypoints 	-> 每个 body part 的识别到的点
 * @param validPairs 		-> 可能的点对
 * @param invalidPairs 		-> 失败的点对的序号
 */
void getValidPairs(const std::vector<cv::Mat>& netOutputParts,
		const std::vector<std::vector<KeyPoint>>& detectedKeypoints,
		std::vector<std::vector<ValidPair>>& validPairs,
		std::set<int>& invalidPairs) {

	int nInterpSamples = 10;
	float pafScoreTh = 0.1;
	float confTh = 0.7;

	for(int k = 0; k < mapIdx.size();++k ){

		//A->B constitute a limb
		cv::Mat pafA = netOutputParts[mapIdx[k].first];
		cv::Mat pafB = netOutputParts[mapIdx[k].second];

		//Find the keypoints for the first and second limb
		const std::vector<KeyPoint>& candA = detectedKeypoints[posePairs[k].first];
		const std::vector<KeyPoint>& candB = detectedKeypoints[posePairs[k].second];

		int nA = candA.size();
		int nB = candB.size();

		/*
		 * If keypoints for the joint-pair is detected
		 * check every joint in candA with every joint in candB
		 * Calculate the distance vector between the two joints
		 * Find the PAF values at a set of interpolated points between the joints
		 * Use the above formula to compute a score to mark the connection valid
		 */

		if(nA != 0 && nB != 0){
			std::vector<ValidPair> localValidPairs;

			for(int i = 0; i< nA;++i){
				int maxJ = -1;
				float maxScore = -1;
				bool found = false;

				for(int j = 0; j < nB;++j){
					std::pair<float,float> distance(candB[j].point.x - candA[i].point.x,candB[j].point.y - candA[i].point.y);

					float norm = std::sqrt(distance.first*distance.first + distance.second*distance.second);

					if(!norm){
						continue;
					}

					distance.first /= norm;
					distance.second /= norm;

					//Find p(u)
					std::vector<cv::Point> interpCoords;
					populateInterpPoints(candA[i].point,candB[j].point,nInterpSamples,interpCoords);
					//Find L(p(u))
					std::vector<std::pair<float,float>> pafInterp;
					for(int l = 0; l < interpCoords.size();++l){
						pafInterp.push_back(
								std::pair<float,float>(
									pafA.at<float>(interpCoords[l].y,interpCoords[l].x),
									pafB.at<float>(interpCoords[l].y,interpCoords[l].x)
									));
					}

					std::vector<float> pafScores;
					float sumOfPafScores = 0;
					int numOverTh = 0;
					for(int l = 0; l< pafInterp.size();++l){
						float score = pafInterp[l].first*distance.first + pafInterp[l].second*distance.second;
						sumOfPafScores += score;
						if(score > pafScoreTh){
							++numOverTh;
						}

						pafScores.push_back(score);
					}

					float avgPafScore = sumOfPafScores/((float)pafInterp.size());

					if(((float)numOverTh)/((float)nInterpSamples) > confTh){
						if(avgPafScore > maxScore){
							maxJ = j;
							maxScore = avgPafScore;
							found = true;
						}
					}

				}/* j */

				if(found){
					localValidPairs.push_back(ValidPair(candA[i].id,candB[maxJ].id,maxScore));
				}

			}/* i */

			validPairs.push_back(localValidPairs);

		} else {
			invalidPairs.insert(k);
			validPairs.push_back(std::vector<ValidPair>());
		}
	}/* k */
}

/**
 * @brief 把每个人的骨架分解出来
 * 	（通过识别出点对的连接）
 * @param validPairs 		-> 成功识别
 * @param invalidPairs 		-> 失败的点对序号
 * @param personwiseKeypoints 	-> 输出:每个人,成功识别出的骨架,的编号
 */
void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>>& validPairs,
		const std::set<int>& invalidPairs,
		std::vector<std::vector<int>>& personwiseKeypoints) {
	for(int k = 0; k < mapIdx.size();++k){
		if(invalidPairs.find(k) != invalidPairs.end()){
			continue;
		}

		const std::vector<ValidPair>& localValidPairs(validPairs[k]);

		int indexA(posePairs[k].first);
		int indexB(posePairs[k].second);

		for(int i = 0; i< localValidPairs.size();++i){
			bool found = false;
			int personIdx = -1;

			for(int j = 0; !found && j < personwiseKeypoints.size();++j){
				if(indexA < personwiseKeypoints[j].size() &&
						personwiseKeypoints[j][indexA] == localValidPairs[i].aId){
					personIdx = j;
					found = true;
				}
			}/* j */

			if(found){
				personwiseKeypoints[personIdx].at(indexB) = localValidPairs[i].bId;
			} else if(k< (nPoints-1)){
				std::vector<int> lpkp(std::vector<int>(nPoints,-1));

				lpkp.at(indexA) = localValidPairs[i].aId;
				lpkp.at(indexB) = localValidPairs[i].bId;

				personwiseKeypoints.push_back(lpkp);
			}

		}/* i */
	}/* k */
}
#define STARTTIME(x) std::chrono::time_point<std::chrono::system_clock>x=std::chrono::system_clock::now()
#define ENDTIME(content,x) do{\
	std::chrono::time_point<std::chrono::system_clock> endTP = std::chrono::system_clock::now();\
	LOG_F(INFO, "Time %s: %ld",content,std::chrono::duration_cast<std::chrono::milliseconds>(endTP - x).count());\
}while(0)

cv::dnn::Net net;
/**
 * @brief  通过设置初始化网络
 * @param s
 * @return cv::dnn::Net
 */
cv::dnn::Net initNet(Settings s){
	nPoints = s.nPoints;
	keypointsMapping = s.keypointsMapping;
	mapIdx = s.mapIdx;
	posePairs = s.posePairs;

	net = cv::dnn::readNetFromCaffe(s.modelTxt, s.modelBin);

	if(s.device=="CPU"){
		LOG_F(INFO, "Using CPU Device");
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}else{
		LOG_F(INFO, "Using GPU device ('CUDA')");
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}

	populateColorPalette(colors,nPoints);

	LOG_F(INFO, "Init Net Complete");

	return net;
}

/**
 * @brief 跑一次网络，输出含有标记的图片
 * @param input cv::Mat
 * @return  	cv::Mat
 */
cv::Mat forwardNet(cv::Mat input, Settings s){
	LOG_F(INFO, ">>>>>>>>>>>>>>>>>>>> Network START");

	cv::Mat inputBlob = cv::dnn::blobFromImage(input, s.scale, cv::Size((int)((double)s.W_in*(double)input.cols/(double)input.rows), s.H_in), cv::Scalar(0, 0, 0), false, false);

	LOG_F(INFO, "%d x %d",input.cols, input.rows);

	net.setInput(inputBlob);
	LOG_F(1, "Input Prepared");

	cv::Mat netOutputBlob = net.forward();
	LOG_F(1, "Forward Completed");

	std::vector<cv::Mat> netOutputParts;
	splitNetOutputBlobToParts(netOutputBlob,cv::Size(input.cols,input.rows),netOutputParts);
	LOG_F(1, "Output Split Completed");
	LOG_F(INFO, "OUTPUT SIZE: %ld",netOutputParts.size());

	int keyPointId = 0;
	std::vector<std::vector<KeyPoint>> detectedKeypoints;
	std::vector<KeyPoint> keyPointsList;

	for(int i = 0; i < nPoints;++i){
		std::vector<KeyPoint> keyPoints;

		getKeyPoints(netOutputParts[i],0.1,keyPoints);

		// std::cout << "Keypoints - " << keypointsMapping[i] << " : " << keyPoints << std::endl;

		for(int i = 0; i< keyPoints.size();++i,++keyPointId){
			keyPoints[i].id = keyPointId;
		}

		detectedKeypoints.push_back(keyPoints);
		keyPointsList.insert(keyPointsList.end(),keyPoints.begin(),keyPoints.end());
	}
	LOG_F(1, "Key Points Extracted");


	cv::Mat outputFrame = input.clone();

	/* 将识别到的 Points 在图上标出来 */
	for(int i = 0; i < nPoints;++i){
		for(int j = 0; j < detectedKeypoints[i].size();++j){
			cv::circle(outputFrame,detectedKeypoints[i][j].point,5,colors[i],-1,cv::LINE_AA);
		}
	}

	std::vector<std::vector<ValidPair>> validPairs;
	std::set<int> invalidPairs;
	getValidPairs(netOutputParts,detectedKeypoints,validPairs,invalidPairs);
	LOG_F(1, "Points Paired");

	std::vector<std::vector<int>> personwiseKeypoints;
	getPersonwiseKeypoints(validPairs,invalidPairs,personwiseKeypoints);
	LOG_F(1, "Person Points Detected");

	/* 绘图 */
	for(int i = 0; i< nPoints-1;++i){
		for(int n  = 0; n < personwiseKeypoints.size();++n){
			const std::pair<int,int>& posePair = posePairs[i];
			int indexA = personwiseKeypoints[n][posePair.first];
			int indexB = personwiseKeypoints[n][posePair.second];

			if(indexA == -1 || indexB == -1){
				continue;
			}

			const KeyPoint& kpA = keyPointsList[indexA];
			const KeyPoint& kpB = keyPointsList[indexB];

			cv::line(outputFrame,kpA.point,kpB.point,colors[i],3,cv::LINE_AA);

		}
	}
	LOG_F(1, "Output Frame Drawn");
	LOG_F(INFO, "<<<<<<<<<<<<<<<<<<<< Network Finished");

	return outputFrame;
}

