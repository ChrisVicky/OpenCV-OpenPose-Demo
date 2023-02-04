# OpenPose-OpenCV Demo

* An encapsulation that demonstrates the usage of OpenCV **`cv::dnn`** on **OpenPOSE** to detect **Multiple** Human Pose.
## File Structure
```
.
├── CMakeLists.txt      -> CMake Settings
├── default.xml         -> Program Configuration
├── getModels.sh        -> Download Models from `OPENPOSE_URL`[1]
├── group.jpg           -> Example Image[2]
├── include
│   └── settings.hpp    -> Read from Configuration Files
├── LICENSE
├── logsrc              -> Log Helper[3]
├── main.cpp            -> Main Files that Utilize APIs
├── openpose            -> Encapsulated APIs[4] 
├── pose                -> OpenPOSE Models
│   ├── body_25
│   └── coco
└── README.md

7 directories, 18 files
```
> * \[1\]: `OPENPOSE_URL`=`http://posefs1.perception.cs.cmu.edu/OpenPose/models/`
> * \[2\]: Reference: `https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/group.jpg`
> * \[3\]: Reference: `https://github.com/emilk/loguru`
> * \[4\]: Reference: `https://github.com/spmallick/learnopencv/tree/master/OpenPose-Multi-Person`

## Requirement
* OpenCV
* pthread (loguru requirement, other choices could be found in the [manual](https://github.com/emilk/loguru))
* wget and Access to the Internet (To download models)

## Usage
```shell
# Download Models
./getModels.sh

mkdir build && cd build
cmake ..
make
./run -h # Check Program Usage
./run
```

