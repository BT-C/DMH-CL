# DMH-CL: Dynamic Model Hardness based Curriculum Learning for Complex Pose Estimation
# Introduction
This is an official pytorch implementation of "DMH-CL: Dynamic Model Hardness based Curriculum Learning for Complex Pose Estimation". In this work, we systematically identify the key deficiencies of existing pose datasets that prevent the power of well-designed models from being fully exploited, and propose corresponding solutions. Specifically, we propose a novel training strategy for robust complex pose estimation, termed **DMH-CL**, which is brought from the idea of curriculum learning (CL), i.e., mainly addresses easy examples in the early training stage and hard ones in the later stage. Different from typical CL methods, we distinguish easy examples from hard ones via mining both the dataset-specific statistical difficulty and the cross-model evaluated difficulty. After that, we adopt an annealing arrangement strategy to construct learning courses from easy to hard. Furthermore, we introduce **Dynamic Model Hardness (DMH)** i.e., the moving average of a model's instantaneous hardness (e.g., a loss or losses fluctuation) over the training history, for courses scheduling. Since a low DMH indicts that a model retains enough knowledge about current examples, we can judge whether the model has learned simple courses well. If so, the model start learning difficult ones. In such a manner, DMH-CL explicitly emphasizes the exploration of hard poses and utilizes the knowledge learned from easy poses to better handle complex scenes. Without any manual model architecture design or use of external data, our DMH-CL achieves significant performance improvements on two challenging benchmarks and is flexible to enhance various methods for pose estimation, including two-stage frameworks (i.e., top-down and bottom-up) and single-stage models. Notably, it respectively achieves substantial performance gains of **2.6%/4.6%** for hard poses compared to the strong single-stage model PETR on CrowdPose and COCO datasets, showing its effectiveness and robustness for complex scenes. 

## Visualization on COCO val2017 dataset
![](https://github.com/vikki-dai/Full-DG/blob/Full-DG/visualization/confused_vis.png)

# Environment
The code is developed based on the [MMPose Project](https://github.com/open-mmlab/mmpose). NVIDIA GPUs are needed. The code is developed and tested using 8 NVIDIA RTX GPU cards. Other platforms or GPU cards are not fully tested.
# Installation
1. You can follow the instruction at https://github.com/open-mmlab/mmpose 
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install requirments：
```python
  pip install -r requirements.txt
```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
```python
  # COCOAPI=/path/to/clone/cocoapi
  git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
  cd $COCOAPI/PythonAPI
  # Install into global site-packages
  make install
  # Alternatively, if you do not have permissions or prefer
  # not to install the COCO API into global site-packages
  python3 setup.py install --user 
```
5. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose)
```python
  Install CrowdPoseAPI exactly the same as COCOAPI.
  Reverse the bug stated in https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
```
# Data Preparation
* For **COCO data**, please download from [COCO download](https://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download and extract them under {POSE_ROOT}/data.  
* For **CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training and validation. Please download and extract them under {POSE_ROOT}/data.