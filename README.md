# ECCV2022-Paper-Code-Interpretation
ECCV2022 论文/代码/解读合集，极市团队整理


# ECCV2022 最新论文分类

检索链接：https://arxiv.org/search/?query=ECCV2022&searchtype=all&source=header<br>
更新时间：2022年7月22日<br>


相关报道：[ECCV 2022放榜了！1629篇论文中选，录用率不到20%](https://mp.weixin.qq.com/s/1UrOvyvZDd11-Ortx4IUFw)

### 1.[ECCV2022 接受论文/代码分方向整理(持续更新)](#eccv)
### 2.[ECCV2022 oral](#oral)
### 3.[ECCV2022 论文解读汇总](#3)


>update: <br>
>2022/7/20 [更新 54 篇](https://www.cvmart.net/community/detail/6592)

<br><br>

<a name="eccv"/>

### 1. ECCV2022 接受论文/代码分方向整理(持续更新)

# 目录
### [1. 检测](#detection)
* [2D目标检测(2D Object Detection)](#IOD)
* [视频目标检测(Video Object Detection)](#VOD)
* [3D目标检测(3D Object Detection)](#3DOD)
* [人物交互检测(HOI Detection)](#HOI)
* [伪装目标检测(Camouflaged Object Detection)](#COD)
* [旋转目标检测(Rotation Object Detection)](#ROD)
* [显著性目标检测(Saliency Object Detection)](#SOD)
* [关键点检测(Keypoint Detection)](#KeypointDetection)
* [车道线检测(Lane Detection)](#LaneDetection)
* [边缘检测(Edge Detection)](#EdgeDetection)
* [消失点检测(Vanishing Point Detection)](#VPD)
* [异常检测(Anomaly Detection)](#AnomalyDetection)

### [2. 分割(Segmentation)](#Segmentation)
* [图像分割(Image Segmentation)](#ImageSegmentation)
* [全景分割(Panoptic Segmentation)](#PanopticSegmentation)
* [语义分割(Semantic Segmentation)](#SemanticSegmentation)
* [实例分割(Instance Segmentation)](#InstanceSegmentation)
* [超像素(Superpixel)](#Superpixel)
* [视频目标分割(Video Object Segmentation)](#VOS)
* [抠图(Matting)](#Matting)
* [密集预测(Dense Prediction)](#DensePrediction)

### [3. 图像处理(Image Processing)](#ImageProcessing)
* [超分辨率(Super Resolution)](#SuperResolution)
* [图像复原/图像增强/图像重建(Image Restoration/Image Reconstruction)](#ImageRestoration)
* [图像去阴影/去反射(Image Shadow Removal/Image Reflection Removal)](#ISR)
* [图像去噪/去模糊/去雨去雾(Image Denoising/Deblurring/Dehazing)](#ImageDenoising)
* [图像编辑/图像修复(Image Edit/Image Inpainting)](#ImageEdit)
* [图像翻译(Image Translation)](#ImageTranslation)
* [图像质量评估(Image Quality Assessment)](#IQA)
* [风格迁移(Style Transfer)](#StyleTransfer)

### [4. 视频处理(Video Processing)](#VideoProcessing)
* [视频编辑(Video Editing)](#VideoEditing)
* [视频修复(Video Inpainting)](#VideoInpainting)
* [视频生成/视频合成(Video Generation/Video Synthesis)](#VideoGeneration)
* [视频超分(Video Super-Resolution)](#VideoSR)

### [5. 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)](#ImageRetrieval)
* [行为识别/行为识别/动作识别/检测/分割(Action/Activity Recognition)](#ActionRecognition)
* [行人重识别/检测(Re-Identification/Detection)](#Re-Identification)
* [图像/视频字幕(Image/Video Caption)](#VideoCaption)
* [视频理解(Video Understanding)](#VideoUnderstanding)
* [图像/视频检索(Image/Video Retrieval)](#VideoRetrieval)

### [6. 估计(Estimation)](#Estimation)
* [光流/运动估计(Flow/Motion Estimation)](#Flow/Pose/MotionEstimation)
* [深度估计(Depth Estimation)](#DepthEstimation)
* [人体解析/人体姿态估计(Human Parsing/Human Pose Estimation)](#HumanPoseEstimation)
* [手势估计(Gesture Estimation)](#GestureEstimation)

### [7. 人脸(Face)](#Face)
* [人脸识别/检测(Facial Recognition/Detection)](#FacialRecognition)
* [人脸生成/合成/重建/编辑(Face Generation/Face Synthesis/Face Reconstruction/Face Editing)](#FaceSynthesis)
* [人脸伪造/反欺骗(Face Forgery/Face Anti-Spoofing)](#FaceAnti-Spoofing)

### [8. 三维视觉(3D Vision)](#3DVision)
* [点云(Point Cloud)](#3DPC)
* [三维重建(3D Reconstruction)](#3DReconstruction)
* [场景重建/视图合成/新视角合成(Novel View Synthesis)](#NeRF)

### [9. 目标跟踪(Object Tracking)](#ObjectTracking)

### [10. 医学影像(Medical Imaging)](#MedicalImaging)

### [11. 文本检测/识别/理解(Text Detection/Recognition/Understanding)](#TDR)

### [12. 遥感图像(Remote Sensing Image)](#RSI)

### [13. GAN/生成式/对抗式(GAN/Generative/Adversarial)](#GAN)

### [14. 图像生成/图像合成(Image Generation/Image Synthesis)](#IGIS)

### [15. 场景图(Scene Graph](#SG)
* [场景图生成(Scene Graph Generation)](#SGG)
* [场景图预测(Scene Graph Prediction)](#SGP)
* [场景图理解(Scene Graph Understanding)](#SGU)

### [16. 视觉推理/视觉问答(Visual Reasoning/VQA)](#VisualReasoning)

### [17. 视觉预测(Vision-based Prediction)](#Vision-basedPrediction)

### [18. 神经网络结构设计(Neural Network Structure Design)](#NNS)
* [CNN](#CNN)
* [Transformer](#Transformer)
* [图神经网络(GNN)](#GNN)
* [神经网络架构搜索(NAS)](#NAS)
* [MLP](#MLP)

### [19. 神经网络可解释性(Neural Network Interpretability)](#interpretability)

### [20. 数据集(Dataset)](#Dataset)

### [21. 数据处理(Data Processing)](#DataProcessing)
* [数据增广(Data Augmentation)](#DataAugmentation)
* [归一化/正则化(Batch Normalization)](#BatchNormalization)
* [图像聚类(Image Clustering)](#ImageClustering)
* [图像压缩(Image Compression)](#ImageCompression)

### [22. 图像特征提取与匹配(Image feature extraction and matching)](#matching)

### [23. 视觉表征学习(Visual Representation Learning)](#VisualRL)

### [24. 模型训练/泛化(Model Training/Generalization)](#ModelTraining)
* [噪声标签(Noisy Label)](#NoisyLabel)
* [长尾分布(Long-Tailed Distribution)](#Long-Tailed)

### [25. 模型压缩(Model Compression)](#ModelCompression)
* [知识蒸馏(Knowledge Distillation)](#KnowledgeDistillation)
* [剪枝(Pruning)](#Pruning)
* [量化(Quantization)](#Quantization)

### [26. 模型评估(Model Evaluation)](#ModelEvaluation)

### [27. 图像分类(Image Classification)](#ImageClassification)

### [28. 图像计数(Image Counting)](#CrowdCounting)

### [29. 机器人(Robotic)](#Robotic)

### [30. 半监督学习/弱监督学习/无监督学习/自监督学习(Self-supervised Learning/Semi-supervised Learning)](#self-supervisedlearning)

### [31. 多模态学习(Multi-Modal Learning)](#MMLearning)
* [视听学习(Audio-visual Learning)](#Audio-VisualLearning)
* [视觉-语言（Vision-language）](#VLRL)

### [32. 主动学习(Active Learning)](#ActiveLearning)

### [33. 小样本学习/零样本学习(Few-shot/Zero-shot Learning)](#Few-shotLearning)

### [34. 持续学习(Continual Learning/Life-long Learning)](#ContinualLearning)

### [35. 迁移学习/domain/自适应(Transfer Learning/Domain Adaptation)](#domain)

### [36. 度量学习(Metric Learning)](#MetricLearning)

### [37. 对比学习(Contrastive Learning)](#ContrastiveLearning)

### [38. 增量学习(Incremental Learning)](#IncrementalLearning)

### [39. 强化学习(Reinforcement Learning)](#RL)

### [40. 元学习(Meta Learning)](#MetaLearning)

### [41. 联邦学习(Federated Learning)](#FederatedLearning)

### [42. 模仿学习(Imitation Learning)](#ImitationLearning)

<br><br>


<br>
<a name="detection"/> 

## 检测


<br>
<a name="IOD"/> 

### 2D目标检测(2D Object Detection)
[4] PromptDet: Towards Open-vocabulary Detection using Uncurated Images<br>
[paper](https://arxiv.org/abs/2203.16513) |[code](https://github.com/fcjian/PromptDet)<br><br>

[3] Point-to-Box Network for Accurate Object Detection via Single Point Supervision<br>
[paper](https://arxiv.org/abs/2207.06827) | [code](https://github.com/ucas-vg/p2bnet)<br><br>

[2] You Should Look at All Objects<br>
[paper](https://arxiv.org/abs/2207.07889) | [code](https://github.com/charlespikachu/yslao)<br><br>

[1] Adversarially-Aware Robust Object Detector (**Oral**))<br>
[paper](https://arxiv.org/abs/2207.06202) | [code](https://github.com/7eu7d7/robustdet)<br><br>


<br>
<a name="3DOD"/> 

### 3D目标检测(3D Object Detection)
[2] Densely Constrained Depth Estimator for Monocular 3D Object Detection<br>
[paper](https://arxiv.org/abs/2207.10047) | [code](https://github.com/bravegroup/dcd)<br><br>

[1] Rethinking IoU-based Optimization for Single-stage 3D Object Detection<br>
[paper](https://arxiv.org/abs/2207.09332)<br><br>


<br>
<a name="VOD"/>

### 视频目标检测(Video Object Detection)<br><br>


<br>
<a name="HOI"/> 

### 人物交互检测(HOI Detection)
[1] Towards Hard-Positive Query Mining for DETR-based Human-Object Interaction Detection<br>
[paper](https://arxiv.org/abs/2207.05293) | [code](https://github.com/muchhair/hqm)<br><br>


<br>
<a name="SOD"/> 

### 显著性目标检测(Saliency Object Detection)<br><br>


<br>
<a name="COD"/> 

### 伪装目标检测(Camouflaged Object Detection)<br><br>


<br>
<a name="ADI"/> 

### 图像异常检测/表面缺陷检测(Anomally Detection in Image)
[1] DICE: Leveraging Sparsification for Out-of-Distribution Detection<br>
[paper](https://arxiv.org/abs/2111.09805) | [code](https://github.com/deeplearning-wisc/dice)<br><br>


<br>
<a name="EdgeDetection"/> 

### 边缘检测(Edge Detection)



<br><br>


<br>
<a name="Segmentation"/> 

## 分割(Segmentation)<br><br>


<br>
<a name="ImageSegmentation"/> 

### 图像分割(Image Segmentation)


<br>
<a name="InstanceSegmentation"/> 

### 实例分割(Instance Segmentation)
[2] Box-supervised Instance Segmentation with Level Set Evolution<br>
[paper](https://arxiv.org/abs/2207.09055)<br><br>

[1] OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers<br>
[paper](https://arxiv.org/abs/2207.02255) | [code](https://github.com/pjlallen/osformer)<br><br>


<br>
<a name="SemanticSegmentation"/> 

### 语义分割(Semantic Segmentation)
[1] 2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds<br>
[paper](https://arxiv.org/abs/2207.04397) | [code](https://github.com/yanx27/2dpass)<br><br>
 

<br>
<a name="VOS"/> 

### 视频目标分割(Video Object Segmentation)
[1] Learning Quality-aware Dynamic Memory for Video Object Segmentation<br>
[paper](https://arxiv.org/abs/2207.07922) | [code](https://github.com/workforai/qdmn)<br><br>


<br>
<a name="RIS"/> 

### 参考图像分割(Referring Image Segmentation)


<br>
<a name="DensePrediction"/> 

### 密集预测(Dense Prediction)


<br><br>


<br>
<a name="ImageProcessing"/> 

## 图像处理(Image Processing)


<br>
<a name="SuperResolution"/> 

### 超分辨率(Super Resolution)
[2] Efficient Meta-Tuning for Content-aware Neural Video Delivery<br>
[paper](https://arxiv.org/abs/2207.09691) | [code](https://github.com/neural-video-delivery/emt-pytorch-eccv2022)<br><br>

[1] Dynamic Dual Trainable Bounds for Ultra-low Precision Super-Resolution Networks<br>
[paper](https://arxiv.org/abs/2203.03844) | [code](https://github.com/zysxmu/ddtb)<br><br>


<br>
<a name="ImageDenoising"/> 

### 图像去噪(Image Denoising/Deblurring/Dehazing)
[3] Perceiving and Modeling Density is All You Need for Image Dehazing (感知和建模密度是图像去雾所需的全部) (**Oral**)<br>
[paper](https://arxiv.org/abs/2111.09733) |[code](https://github.com/Owen718/Perceiving-and-Modeling-Density-is-All-You-Need-for-Image-Dehazing)<br><br>

[2] Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance<br>
[paper](https://arxiv.org/abs/2207.10123) | [code](https://github.com/zzh-tech/Animation-from-Blur)<br><br>

[1] Deep Semantic Statistics Matching (D2SM) Denoising Network<br>
[paper](https://arxiv.org/abs/2207.09302)<br><br>


<br>
<a name="ImageRestoration"/> 

### 图像复原/图像增强/图像重建(Image Restoration/Image Reconstruction)
[8] Bringing Rolling Shutter Images Alive with Dual Reversed Distortion(通过双重反转失真使滚动快门图像重现) (**Oral**)<br>
[paper](https://arxiv.org/abs/2203.06451) | [code](https://github.com/zzh-tech/dual-reversed-rs)<br><br>

[7] Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression<br>
[paper](https://arxiv.org/abs/2207.10564) | [code](https://github.com/jinyeying/night-enhancement)<br><br>

[6] Semantic-Sparse Colorization Network for Deep Exemplar-based Colorization<br>
[paper](https://arxiv.org/abs/2112.01335) <br><br>

[5] Geometry-aware Single-image Full-body Human Relighting<br>
[paper](https://arxiv.org/abs/2207.04750) <br><br>
  
[4] Multi-Modal Masked Pre-Training for Monocular Panoramic Depth Completion<br>
[paper](https://arxiv.org/abs/2203.09855) <br><br>
  
[3] PanoFormer: Panorama Transformer for Indoor 360 Depth Estimation<br>
[paper](https://arxiv.org/abs/2203.09283) <br><br>
  
[2] SESS: Saliency Enhancing with Scaling and Sliding<br>
[paper](https://arxiv.org/abs/2207.01769) <br><br>

[1] RigNet: Repetitive Image Guided Network for Depth Completion<br>
[paper](https://arxiv.org/abs/2107.13802) <br><br>


<br>
<a name="ImageOutpainting"/>   

### 图像外推(Image Outpainting)
[1] Outpainting by Queries<br>
[paper](https://arxiv.org/abs/2207.05312) | [code](https://github.com/kaiseem/queryotr)<br><br>


<br>
<a name="StyleTransfer"/> 

### 风格迁移(Style Transfer)
[1] CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer (**Oral**)<br>
[paper](https://arxiv.org/abs/2207.04808) | [code](https://github.com/JarrentWu1031/CCPL)<br><br>

 
<br><br>


<br>
<a name="VideoProcessing"/> 

## 视频处理(Video Processing)


<br>
<a name="VideoEditing"/> 

### 视频编辑(Video Editing)
[2] Improving the Perceptual Quality of 2D Animation Interpolation<br>
[paper](https://arxiv.org/abs/2111.12792) | [code](https://github.com/shuhongchen/eisai-anime-interpolator)<br><br>

[1] Real-Time Intermediate Flow Estimation for Video Frame Interpolation<br>
[paper](https://arxiv.org/abs/2011.06294)  | [code](https://github.com/MegEngine/arXiv2020-RIFE)<br><br>

<br>
<a name="VideoInpainting"/> 

### 视频修复(Video Inpainting)
[1] Error Compensation Framework for Flow-Guided Video Inpainting<br>
[paper](https://arxiv.org/abs/2207.10391)<br><br>

<br><br>


<br>
<a name="ImageRetrieval"/> 

## 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)


<br>
<a name="ActionRecognition"/> 

### 行为识别/行为识别/动作识别/检测/分割(Action/Activity Recognition)
[4] GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality<br>
[paper](https://arxiv.org/abs/2203.03972) | [code](https://github.com/shiqiyu/opengait)<br><br>

[3] Collaborating Domain-shared and Target-specific Feature Clustering for Cross-domain 3D Action Recognition<br>
[paper](https://arxiv.org/abs/2207.09767) | [code](https://github.com/canbaoburen/CoDT)<br><br>

[2] ReAct: Temporal Action Detection with Relational Queries<br>
[paper](https://arxiv.org/abs/2207.07097) | [code](https://github.com/sssste/react)<br><br>

[1] Hunting Group Clues with Transformers for Social Group Activity Recognition<br>
[paper](https://arxiv.org/abs/2207.05254) <br><br>
  
  
<br>
<a name="Re-Identification"/> 

### 行人重识别/检测(Re-Identification/Detection)
[1] PASS: Part-Aware Self-Supervised Pre-Training for Person Re-Identification<br>
[paper](https://arxiv.org/abs/2203.03931) | [code](https://github.com/casia-iva-lab/pass-reid)<br><br>

<br>
<a name="VideoCaption"/> 

### 图像/视频字幕(Image/Video Caption)


<br>
<a name="VideoUnderstanding"/> 

### 视频理解(Video Understanding)
[1] GraphVid: It Only Takes a Few Nodes to Understand a Video (**Oral**)<br>
[paper](https://arxiv.org/abs/2207.01375) <br><br>


<br>
<a name="VideoRetrieval"/> 

### 图像/视频检索(Image/Video Retrieval)
[5] Feature Representation Learning for Unsupervised Cross-domain Image Retrieval
[paper](https://arxiv.org/abs/2207.09721) | [code](https://github.com/conghuihu/ucdir)

[4] LocVTP: Video-Text Pre-training for Temporal Localization<br>
[paper](https://arxiv.org/abs/2207.10362) | [code](https://github.com/mengcaopku/locvtp)<br><br>

[3] Deep Hash Distillation for Image Retrieval<br>
[paper](https://arxiv.org/abs/2112.08816) | [code](https://github.com/youngkyunjang/deep-hash-distillation)<br><br>

[2] TS2-Net: Token Shift and Selection Transformer for Text-Video Retrieval<br>
[paper](https://arxiv.org/abs/2207.07852) | [code](https://github.com/yuqi657/ts2_net)<br><br>

[1] Lightweight Attentional Feature Fusion: A New Baseline for Text-to-Video Retrieval<br>
[paper](https://arxiv.org/abs/2112.01832)<br><br>


<br><br>


<br>
<a name="Estimation"/> 

## 估计(Estimation)

<br>
<a name="VisualLocalization"/> 

### 视觉定位/位姿估计(Visual Localization/Pose Estimation)
[2] Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration<br>
[paper] (https://arxiv.org/abs/2207.10447) | [code](https://github.com/164140757/scm)<br><br>

[1] Category-Level 6D Object Pose and Size Estimation using Self-Supervised Deep Prior Deformation Networks<br>
[paper](https://arxiv.org/abs/2207.05444)  | [code](https://github.com/jiehonglin/self-dpdn)<br><br>


<br>
<a name="DepthEstimation"/> 

### 深度估计(Depth Estimation)
[1] Physical Attack on Monocular Depth Estimation with Optimal Adversarial Patches<br>
[paper](https://arxiv.org/abs/2207.04718) <br><br>


<br><br>


<br>
<a name="Face"/> 

## 7. 人脸(Face)


<br><br>


<br>
<a name="3DVision"/> 

## 8. 三维视觉(3D Vision)


<br>
<a name="3DReconstruction"/> 

### 三维重建(3D Reconstruction)
[1] DiffuStereo: High Quality Human Reconstruction via Diffusion-based Stereo Using Sparse Cameras<br>
[paper](https://arxiv.org/abs/2207.08000)<br><br>


<br>
<a name="NeRF"/> 

### 场景重建/视图合成/新视角合成(Novel View Synthesis)
[1] Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields<br>
[paper](https://arxiv.org/abs/2203.10821) | [code](https://github.com/donydchen/sem2nerf)<br><br>


<br><br>


<br>
<a name="ObjectTracking"/> 

## 9. 目标跟踪(Object Tracking)
[1] Towards Grand Unification of Object Tracking (**Oral**)<br>
[paper](https://arxiv.org/abs/2207.07078) | [code](https://github.com/masterbin-iiau/unicorn)<br><br>


<br><br>


<br>
<a name="MedicalImaging"/> 

## 10. 医学影像(Medical Imaging)


<br><br>


<br>
<a name="TDR"/> 

## 11. 文本检测/识别/理解(Text Detection/Recognition/Understanding)
[2] End-to-End Video Text Spotting with Transformer (**Oral**)<br>
[paper](https://arxiv.org/abs/2203.10539) | [code](https://github.com/weijiawu/transdetr)<br><br>

[1] Dynamic Low-Resolution Distillation for Cost-Efficient End-to-End Text Spotting<br>
[paper](https://arxiv.org/abs/2207.06694) | [code](https://github.com/hikopensource/davar-lab-ocr)<br><br>


<br><br>


<br>
<a name="RSI"/> 

## 12. 遥感图像(Remote Sensing Image)


<br><br>


<br>
<a name="GAN"/> 

## 13. GAN/生成式/对抗式(GAN/Generative/Adversarial)
[6] Adaptive Image Transformations for Transfer-based Adversarial Attack<br>
[paper](https://arxiv.org/abs/2111.13844)<br><br>

[5] Generative Multiplane Images: Making a 2D GAN 3D-Aware<br>
[paper](https://arxiv.org/abs/2207.10642) | [code](https://github.com/apple/ml-gmpi)<br><br>

[4] Eliminating Gradient Conflict in Reference-based Line-Art Colorization<br>
[paper](https://arxiv.org/abs/2207.06095) | [code](https://github.com/kunkun0w0/sga)<br><br>

[3] WaveGAN: Frequency-aware GAN for High-Fidelity Few-shot Image Generation<br>
[paper](https://arxiv.org/abs/2207.07288) | [code](https://github.com/kobeshegu/eccv2022_wavegan) <br><br>

[2] FakeCLR: Exploring Contrastive Learning for Solving Latent Discontinuity in Data-Efficient GANs<br>
[paper](https://arxiv.org/abs/2207.08630) | [code](https://github.com/iceli1007/fakeclr) <br><br>

[1] UniCR: Universally Approximated Certified Robustness via Randomized Smoothing<br>
[paper](https://arxiv.org/abs/2207.02152) <br><br>


<br><br>


<br>
<a name="IGIS"/> 

## 14. 图像生成/图像合成(Image Generation/Image Synthesis)


<br><br>


<br>
<a name="SG"/> 

## 15. 场景图(Scene Graph)


<br><br>


<br>
<a name="VisualReasoning"/> 

## 16. 视觉推理/视觉问答(Visual Reasoning/VQA)


<br><br>


<br>
<a name="Vision-basedPrediction"/> 

## 17. 视觉预测(Vision-based Prediction)
[1] D2-TPred: Discontinuous Dependency for Trajectory Prediction under Traffic Lights<br>
[paper](https://arxiv.org/abs/2207.10398) | [code](https://github.com/vtp-tl/d2-tpred)<br><br>

<br><br>


<br>
<a name="NNS"/>  

## 18. 神经网络结构设计(Neural Network Structure Design)


<br>
<a name="CNN"/> 

### CNN


<br>
<a name="Transformer"/> 

### Transformer
[2] ScalableViT: Rethinking the Context-oriented Generalization of Vision Transformer<br>
[paper](https://arxiv.org/abs/2203.10790) | [code](https://github.com/yangr116/scalablevit)<br><br>

[1] Visual Prompt Tuning<br>
[paper](https://arxiv.org/abs/2203.12119) | [code](https://github.com/KMnP/vpt)<br><br>


<br>
<a name="GNN"/> 

### 图神经网络(GNN)


<br>
<a name="NAS"/> 

### 神经网络架构搜索(NAS)
[3] ScaleNet: Searching for the Model to Scale<br>
[paper](https://arxiv.org/abs/2207.07267) | [code](https://github.com/luminolx/scalenet)<br><br>

[2] Ensemble Knowledge Guided Sub-network Search and Fine-tuning for Filter Pruning<br>
[paper](https://arxiv.org/abs/2203.02651)  | [code](https://github.com/sseung0703/ekg)<br><br>

[1] EAGAN: Efficient Two-stage Evolutionary Architecture Search for GANs<br>
[paper](https://arxiv.org/abs/2111.15097)  | [code](https://github.com/marsggbo/EAGAN)<br><br>


<br>
<a name=" MLP"/> 

### MLP


<br><br>


<br>
<a name="interpretability"/> 

## 19. 神经网络可解释性(Neural Network Interpretability)


<br><br>


<br>
<a name="Dataset"/> 

## 20. 数据集(Dataset)


<br>
<a name="DataProcessing"/> 

## 21. 数据处理(Data Processing)


<br>
<a name="DataAugmentation"/> 

### 数据增广(Data Augmentation)


<br>
<a name="BatchNormalization"/> 

### 归一化/正则化(Batch Normalization)
[1] Fine-grained Data Distribution Alignment for Post-Training Quantization (**Oral**)<br>
[paper](https://arxiv.org/abs/2109.04186) | [code](https://github.com/zysxmu/fdda)<br><br>


<br>
<a name="ImageClustering"/>   

### 图像聚类(Image Clustering)


<br>
<a name="ImageCompression"/> 

### 图像压缩(Image Compression)


<br><br>


<br>
<a name="matching"/> 

## 22. 图像特征提取与匹配(Image feature extraction and matching)
[1] Unsupervised Deep Multi-Shape Matching<br>
[paper](https://arxiv.org/abs/2207.09610)<br><br>



<br><br>


<br>
<a name="VisualRL"/>

## 23. 视觉表征学习(Visual Representation Learning)
[1] Object-Compositional Neural Implicit Surfaces<br>
[paper](https://arxiv.org/abs/2207.09686) | [code](https://github.com/qianyiwu/objsdf)<br><br>


<br><br>


<br>
<a name="ModelTraining"/> 

## 24. 模型训练/泛化(Model Training/Generalization)


<br><br>

<br>
<a name="NoisyLabel"/> 

### 噪声标签(Noisy Label)
[1] Learning with Noisy Labels by Efficient Transition Matrix Estimation to Combat Label Miscorrection<br>
[paper](https://arxiv.org/abs/2111.14932) <br><br>
  

<br>
<a name="Long-Tailed"/> 

### 长尾分布(Long-Tailed Distribution)


<br><br>


<br>
<a name="ModelCompression"/> 

## 25. 模型压缩(Model Compression)


<br>
<a name="KnowledgeDistillation"/> 

### 知识蒸馏(Knowledge Distillation)
[1] Knowledge Condensation Distillation<br>
[paper](https://arxiv.org/abs/2207.05409) | [code](https://github.com/dzy3/kcd)<br><br>
 
 
<br> 
<a name="Pruning"/> 

### 剪枝(Pruning)

<br>
<a name="Quantization"/> 

### 量化(Quantization)


<br><br>


<br>
<a name="ModelEvaluation"/> 

## 26. 模型评估(Model Evaluation)

[1] Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting<br>
[paper](https://arxiv.org/abs/2207.04624) | [code](https://github.com/d1024choi/hlstrajforecast)<br><br>


<br><br>


<br>
<a name="ImageClassification"/> 

## 27. 图像分类(Image Classification)


<br><br>


<br>
<a name="CrowdCounting"/> 

## 28. 图像计数(Image Counting)


<br><br>


<br>

<a name="Robotic"/> 

## 29. 机器人(Robotic)


<br><br>


<br>
<a name="self-supervisedlearning"/> 

## 30. 半监督学习/弱监督学习/无监督学习/自监督学习(Self-supervised Learning/Semi-supervised Learning)
[6] CA-SSL: Class-Agnostic Semi-Supervised Learning for Detection and Segmentation<br>
[paper](https://arxiv.org/abs/2112.04966) | [code](https://github.com/dvlab-research/Entity)<br><br>

[5] FedX: Unsupervised Federated Learning with Cross Knowledge Distillation<br>
[paper](https://arxiv.org/abs/2207.09158)<br><br>

[4] Synergistic Self-supervised and Quantization Learning<br>
[paper](https://arxiv.org/abs/2207.05432) | [code](https://github.com/megvii-research/ssql-eccv2022)<br><br>

[3] Contrastive Deep Supervision<br>
[paper](https://arxiv.org/abs/2207.05306) | [code](https://github.com/archiplab-linfengzhang/contrastive-deep-supervision)<br><br>

[2] Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection<br>
[paper](https://arxiv.org/abs/2207.02541)<br><br>

[1] Image Coding for Machines with Omnipotent Feature Learning<br>
[paper](https://arxiv.org/abs/2207.01932) <br><br>


<br><br>


<br>
<a name="MMLearning"/> 

## 31. 多模态学习/跨模态(Multi-Modal Learning/Cross-Modal Learning)


<br><br>

<br>
<a name="Audio-VisualLearning"/> 

###  视听学习(Audio-visual Learning)<br>


<br>
<a name="VLRL"/> 

### 视觉-语言（Vision-language）
[1] Contrastive Vision-Language Pre-training with Limited Resources<br>
[paper](https://arxiv.org/abs/2112.09331) | [code](https://github.com/zerovl/zerovl)<br><br>


<br>
<a name="CML"/> 

### 跨模态（cross-modal)
[1] Cross-modal Prototype Driven Network for Radiology Report Generation<br>
[paper](https://arxiv.org/abs/)  | [code](https://github.com/markin-wang/xpronet)<br><br>


<br><br>


<br>
<a name="ActiveLearning"/> 

## 32. 主动学习(Active Learning)


<br>
<a name="Few-shotLearning"/> 

## 33. 小样本学习/零样本学习(Few-shot/Zero-shot Learning)
[1] Learning Instance and Task-Aware Dynamic Kernels for Few Shot Learning<br>
[paper](https://arxiv.org/abs/2112.03494) <br><br>


<br>
<a name="ContinualLearning"/> 

## 34. 持续学习(Continual Learning/Life-long Learning)


<br><br>


<br>
<a name="domain"/> 

## 35. 迁移学习/domain/自适应(Transfer Learning/Domain Adaptation)
[2] Factorizing Knowledge in Neural Networks<br>
[paper](https://arxiv.org/abs/2207.03337)  | [code](https://github.com/adamdad/knowledgefactor)<br><br>


[1] CycDA: Unsupervised Cycle Domain Adaptation from Image to Video<br>
[paper](https://arxiv.org/abs/2203.16244) <br><br>


<br><br>


<br>
<a name="MetricLearning"/> 

## 36. 度量学习(Metric Learning)


<br>
<a name="ContrastiveLearning"/> 

## 37. 对比学习(Contrastive Learning)


<br>
<a name="IncrementalLearning"/> 

## 38. 增量学习(Incremental Learning)


<br>
<a name="RL"/> 

## 39. 强化学习(Reinforcement Learning)
[1] Target-absent Human Attention<br>
[paper](https://arxiv.org/abs/2207.01166) | [code](https://github.com/neouyghur/sess)<br><br>


<br>
<a name="MetaLearning"/> 

## 40. 元学习(Meta Learning)


<br>
<a name="FederatedLearning"/> 

## 41. 联邦学习(Federated Learning)


<br>
<a name="ImitationLearning"/> 

## 42. 模仿学习(Imitation Learning)
[1] Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction<br>
[paper](https://arxiv.org/abs/2207.09705)<br><br>




<br>
<a name="oral"/> 

### ECCV2022 oral 

[8] Perceiving and Modeling Density is All You Need for Image Dehazing (感知和建模密度是图像去雾所需的全部) (**Oral**)<br>
[paper](https://arxiv.org/abs/2111.09733) |[code](https://github.com/Owen718/Perceiving-and-Modeling-Density-is-All-You-Need-for-Image-Dehazing)<br><br>

[7] Bringing Rolling Shutter Images Alive with Dual Reversed Distortion(通过双重反转失真使滚动快门图像重现) (**Oral**)<br>
[paper](https://arxiv.org/abs/2203.06451) | [code](https://github.com/zzh-tech/dual-reversed-rs)<br><br>

[6] End-to-End Video Text Spotting with Transformer(使用 Transformer 的端到端视频文本定位) (**Oral**)<br>
[paper](https://arxiv.org/abs/2203.10539) | [code](https://github.com/weijiawu/transdetr)<br><br>

[5] GraphVid: It Only Takes a Few Nodes to Understand a Video(GraphVid：只需几个节点即可理解视频) (**Oral**)<br>
[paper](https://arxiv.org/abs/2207.01375) <br><br>

[4] CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer(CCPL：用于通用风格迁移的对比相干性保留损失) (**Oral**)<br>
[paper](https://arxiv.org/abs/2207.04808) | [code](https://github.com/JarrentWu1031/CCPL)<br><br>

[3] Fine-grained Data Distribution Alignment for Post-Training Quantization(训练后量化的细粒度数据分布对齐) (**Oral**)<br>
[paper](https://arxiv.org/abs/2109.04186) | [code](https://github.com/zysxmu/fdda)<br><br>

[2] Adversarially-Aware Robust Object Detector(对抗性感知鲁棒目标检测器) (**Oral**))<br>
[paper](https://arxiv.org/abs/2207.06202) | [code](https://github.com/7eu7d7/robustdet)<br><br>

[1] Towards Grand Unification of Object Tracking(迈向目标跟踪的大统一) (**Oral**)<br>
[paper](https://arxiv.org/abs/2207.07078) | [code](https://github.com/masterbin-iiau/unicorn)<br><br>

<br>

<a name="3"/> 

# 3. ECCV2022 论文解读汇总

【1】[ECCV 2022 Oral | Unicorn：迈向目标跟踪的大统一](https://www.cvmart.net/community/detail/6586)

