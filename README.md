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

## Configurations
* File `default.xml`

```
<?xml version="1.0"?>
<opencv_storage>
	<Settings>

		<!-- specify what kind of model was trained. It could be (COCO, BODY_25) depends on dataset. -->
		<dataset>BODY_25</dataset>
		<!-- model configuration, e.g. hand/pose.prototxt -->
		<modelTxt>./pose/body_25/pose_deploy.prototxt</modelTxt>
		<!-- model weights, e.g. hand/pose_iter_102000.caffemodel -->
		<modelBin>./pose/body_25/pose_iter_584000.caffemodel</modelBin>

		<!-- Preprocess input image by resizing to a specific widh. -->
		<W_in>368</W_in>
		<!-- Preprocess input image by resizing to a specific height. -->
		<H_in>368</H_in>

		<!-- threshold or confidence value for the heatmap -->
		<thresh>0.07</thresh>
		<!-- scale for blob -->
		<scale>0.003922</scale>

		<logPath>log.log</logPath>

		<!-- Could be (CPU, GPU) depends on devices and OpenCV Versions -->
		<device>CPU</device>

		<!-- "0" = Use Camera -->
		<!-- <imageFile>"0"</imageFile> -->
		<!-- "*.png/jpg .etc" = Use Images -->
		<imageFile>./group.jpg</imageFile>

	</Settings>
</opencv_storage>
```

