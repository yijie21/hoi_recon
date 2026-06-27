# EgoAERO: Learning Dexterous Manipulation from a Single Egocentric Video without Object Assets

Yichen Niu $ ^{1,2} $, Haoran Lv $ ^{1} $, Xinrui Zhang $ ^{1} $, Xueyao Wan $ ^{1} $, Shiyu Gao $ ^{1} $

Ying Ai $ ^{1} $, Hui Xu $ ^{1} $, Yongqi Hu $ ^{1} $, Hengyi Zhang $ ^{3} $, Yang Xie $ ^{3} $

Zhaxizhuoma $ ^{4,5} $, Yue Zhao $ ^{1} $, Zhenshan Bing $ ^{6} $, Yan Ding $ ^{2,7,8} $, Jianxing Liu $ ^{1,*} $

 $ ^{1} $School of Astronautics, Harbin Institute of Technology

 $ ^{2} $Lumos Robotic

 $ ^{3} $Suzhou Research Institute, Harbin Institute of Technology

 $ ^{4} $Shanghai Jiao Tong University  $ ^{5} $Shanghai AI Lab

 $ ^{6} $Nanjing University  $ ^{7} $Xi'an Jiaotong-Liverpool University  $ ^{8} $Fudan University

 $ ^{*} $Corresponding author

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//14a7b624-c485-4ffd-8efd-8ba2751474d9/markdown_0/imgs/img_in_image_box_263_543_951_906.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-06-27T08%3A49%3A02Z%2F-1%2F%2F242be70dab49778e58ad54de3698646a46db88efee748e33c79a96971a8e2905" alt="Image" width="56%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 1: End-to-end overview of EgoAERO. Starting from a single egocentric RGB-D human demonstration, EgoAERO reconstructs contact-consistent hand-object trajectories without object assets, transfers them to a simulated dexterous hand through two-stage policy learning, and executes the learned manipulation behavior on a real-world robot.</div> </div>


Abstract: Egocentric RGB-D videos offer a natural source of human dexterous manipulation demonstrations, but existing data is difficult to use for robot learning because object pose, geometry, and contact information are often missing or require pre-scanned object assets. We present EgoAERO, the first framework that learns dexterous manipulation from a single egocentric RGB-D human demonstration without object assets. EgoAERO reconstructs contact-consistent hand-object trajectories through asset-free object tracking and reconstruction, ego motion compensation, and adaptive contact optimization, then converts them into robot policies using two-stage residual learning. We further introduce an online quality assessment mechanism and construct EgoDex-R, a large-scale egocentric dataset with 4.3M RGB-D frames for dexterous policy learning. Simulation and real-world experiments show that EgoAERO enables single-demonstration dexterous manipulation and achieves downstream performance close to CAD-based reconstructions on HOI4D.

Keywords: Dexterous manipulation, Egocentric RGB-D demonstration, Asset-free reconstruction

## 1 Introduction

Natural human demonstrations are becoming an important source for training dexterous manipulation policies. Compared with teleoperation, UMI-style interfaces [1, 2], or data gloves, egocentric RGB-D devices can collect natural hand motions, object interactions, and contact-rich manipulation in everyday environments with minimal intrusion. This makes ego demonstrations a promising data source for scalable dexterous robot learning.

However, existing ego data is still hard to use directly for policy learning. First, large-scale ego-centric datasets such as EPIC-KITCHENS [3], Ego4D [4], and EgoDex [5] provide rich videos or hand motion, but usually lack the manipulated object's 6-DoF pose, geometry, and contact state, which are needed for object-conditioned rewards, contact constraints, and replayable tasks. Second, datasets with 3D hand-object annotations, including H2O [6], HOI4D [7], HOT3D [8], HO-3D [9], DexYCB [10], OakInk [11], and ARCTIC [12], often rely on CAD models, scanned meshes, multi-view capture, or known object assets. This limits their scalability to arbitrary daily objects and natural collection settings.

We propose EgoAERO, the first framework that converts a single egocentric RGB-D human demonstration into an executable dexterous manipulation policy without object assets (Fig. 1). EgoAERO first reconstructs structured hand-object trajectories from raw ego RGB-D data by combining lightweight MLLM semantic initialization, asset-free object tracking and reconstruction under hand occlusion, ego motion compensation, and adaptive contact optimization. It then trains a two-stage residual policy: a hand-tracking policy learns to follow the reconstructed human hand motion, and a residual policy uses object pose and contact feedback to produce executable dexterous manipulation.

We also design an online ego data quality assessment mechanism and build EgoDex-R. Simulation and real-world experiments show that EgoAERO learns executable manipulation from natural ego demonstrations, and HOI4D comparisons show that its asset-free reconstructions achieve downstream policy learning performance close to CAD-based reconstructions. Our main contributions are:

- EgoAERO: the first framework for learning dexterous manipulation from a single ego RGB-D video without object assets.

An asset-free ego hand-object reconstruction pipeline with robust object tracking, geometry reconstruction, ego motion compensation, and adaptive contact optimization.

- An online ego quality assessment mechanism and EgoDex-R, a large-scale dataset with 4.3M RGB-D frames for dexterous policy learning.

## 2 EgoAERO Method

### 2.1 Asset-free Egocentric Hand-Object Reconstruction

Fig. 2 illustrates the overall pipeline of EgoAERO's asset-free egocentric hand-object reconstruction module.

#### 2.1.1 MLLM-Guided Semantic Preprocessing for Dexterous Manipulation

Natural human demonstrations often involve multiple objects and complex hand-object interactions, where the task goal, the manipulated object, and its relations to other objects in the scene are not explicitly specified. Therefore, task-level semantic parsing is required before hand-object state reconstruction. To this end, EgoAERO first employs an MLLM to perform lightweight semantic preprocessing on the raw RGB-D video. The system samples a small number of keyframes from the input video and feeds them, together with the task description, into the MLLM to identify the target manipulated object and its potentially related supporting objects, containers, or static objects. Based on the semantic parsing results, the MLLM generates text prompts for SAM3 [13, 14] segmentation

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//14a7b624-c485-4ffd-8efd-8ba2751474d9/markdown_2/imgs/img_in_image_box_222_145_1002_689.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-06-27T08%3A49%3A03Z%2F-1%2F%2Faf8186e3df9a6a2d368015a5be93cd8255c07b0395d8faf42750c04baba59e03" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 2: Overview of asset-free egocentric hand-object reconstruction. Given a single ego RGB-D video, EgoAERO first performs MLLM-guided semantic preprocessing to identify the manipulated object and obtain segmentation prompts. It then reconstructs the unknown object without CAD assets through coarse pose initialization, keyframe memory-pool pose optimization, and neural-field-guided coarse-to-fine mesh reconstruction. The reconstructed object state is combined with RGB-D corrected hand pose, ego motion compensation, and adaptive contact optimization to produce contact-consistent hand-object trajectories for downstream robot learning.</div> </div>


and selects a less-occluded keyframe as the initialization seed frame. SAM3 then uses these prompts to obtain frame-wise masks of the target object and related objects, which serve as inputs to the subsequent asset-free object tracking and reconstruction module. It is important to note that the MLLM is only used for semantic-level initialization and pipeline configuration; it does not directly estimate the object 6-DoF pose, object geometry, or hand pose.

#### 2.1.2 Asset-free Object Tracking and Reconstruction under Egocentric Hand Occlusion

In egocentric manipulation scenarios without relying on object CAD models or pre-scanned meshes, object tracking and reconstruction must handle frequent hand occlusions, rapidly changing visible regions, and tracking drift caused by low-texture surfaces. Inspired by recent unknown-object RGB-D tracking and reconstruction systems [15, 16, 17], EgoAERO relies only on RGB-D observations and object masks, and recovers the 6-DoF pose trajectory and geometric representation of unknown objects through coarse pose initialization, keyframe memory-pool optimization, and coarse-to-fine mesh reconstruction. This provides stable object states for subsequent hand-object joint modeling.

Coarse Object Pose Initialization. To provide a reliable initialization for the subsequent keyframe-based pose optimization, EgoAERO first estimates a coarse object pose  $ \tilde{T}_t \in SE(3) $ for the current frame  $ F_t $. Given the RGB-D input and object mask  $ M_t^O $ at frame  $ t $, the system back-projects the visible object region into a local point cloud and initializes a canonical object frame  $ \mathcal{O} $ from the first reliable observation. EgoAERO then establishes local RGB-D correspondences between the visible object regions in the current frame  $ F_t $ and the previous frame  $ F_{t-1} $, and lifts the 2D matches to 3D correspondences using depth. To reduce the influence of outliers caused by hand occlusion, low-texture surfaces, and segmentation errors, the system adopts RANSAC [18] for ro

bust rigid pose estimation and selects the hypothesis with the highest inlier consistency as the coarse object pose  $ \tilde{T}_{t} $. This coarse pose is used only as the initialization for the subsequent memory-pool pose optimization, rather than as the final object pose.

Keyframe Memory-pool Pose Optimization. The coarse pose  $ T_t $ can still drift under hand occlusion, partial visibility, and low-texture observations. To stabilize tracking, EgoAERO maintains a keyframe memory pool  $ \mathcal{P} $ containing historical frames with reliable object observations and complementary viewpoints. For the current frame  $ F_t $, EgoAERO selects a small subset of relevant keyframes  $ \mathcal{K}_t \subset \mathcal{P} $ and constructs a local pose graph  $ \mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t) $, where  $ \mathcal{V}_t = \{t\} \cup \mathcal{K}_t $. The current node is initialized by  $ \tilde{T}_t $, while memory nodes are initialized by their previously optimized poses. The object poses in this local graph are then jointly optimized by combining cross-frame correspondence, geometric, surface, silhouette, and pose-prior constraints:

 $$ \min_{\{T_{i}\}_{i\in\mathcal{V}_{t}}}\sum_{(i,j)\in\mathcal{E}_{t}}\left[\lambda_{f}E_{\mathrm{feat}}(i,j)+\lambda_{g}E_{\mathrm{geo}}(i,j)\right]+\sum_{i\in\mathcal{V}_{t}}\left[\lambda_{s}E_{\mathrm{sdf}}(i)+\lambda_{m}E_{\mathrm{mask}}(i)+\lambda_{p}E_{\mathrm{pose}}(i)\right] $$ 

Here,  $ E_{feat} $ and  $ E_{geo} $ provide multi-view RGB-D alignment,  $ E_{sdf} $ couples pose tracking with the online object geometry,  $ E_{mask} $ constrains silhouette consistency, and  $ E_{pose} $ prevents excessive deviation from the initial poses. After optimization, the current pose  $ T_t $ is used as the final tracking result and is written back to the memory pool when the frame provides reliable new view coverage. This local memory-pool optimization converts single-frame coarse tracking into a temporally stable 6-DoF object trajectory. Details of memory scoring, keyframe selection, graph construction, and residual definitions are provided in Appendix A.

Neural Object Field Guided Coarse-to-Fine Mesh Reconstruction. Object observations in egocentric dexterous manipulation videos are usually incomplete. Directly reconstructing a high-fidelity mesh from local RGB-D observations can easily introduce noise and coordinate inconsistency, while relying only on an online neural field often provides temporally consistent but relatively coarse geometry. Therefore, EgoAERO adopts a coarse-to-fine object reconstruction strategy: it first uses an online neural object field to obtain an object-centric and temporally consistent coarse geometry, and then combines SAM3D [19] with the original RGB-D observations to recover finer surface details. Specifically, it maintains an online neural object field  $ \Omega_{\Theta} $ in the object canonical frame  $ \mathcal{O} $ and fuses keyframe observations aligned by the optimized poses. The field is trained with occlusion-aware ray supervision:

 $$ \mathcal{L}_{\mathrm{o b j}}=\lambda_{\mathrm{s u r f}}\mathcal{L}_{\mathrm{s u r f}}+\lambda_{\mathrm{f r e e}}\mathcal{L}_{\mathrm{f r e e}}+\lambda_{\mathrm{o c c}}\mathcal{L}_{\mathrm{o c c}}+\lambda_{\mathrm{r g b}}\mathcal{L}_{\mathrm{r g b}}+\lambda_{\mathrm{e i k}}\mathcal{L}_{\mathrm{e i k}} $$ 

This produces a temporally consistent coarse mesh  $ \mathcal{M}_O^{\mathrm{coarse}} $ from the zero level set of  $ \Omega_\Theta $. Second, EgoAERO uses the original RGB-D observations, object masks, and point maps as inputs to SAM3D to recover a more detailed mesh  $ \mathcal{M}_O^{\mathrm{sam}} $. Since  $ \mathcal{M}_O^{\mathrm{sam}} $ is not guaranteed to lie in the object canonical frame, EgoAERO aligns it to  $ \mathcal{M}_O^{\mathrm{coarse}} $ with rigid and scale alignment, yielding the final mesh  $ \mathcal{M}_O $. Thus, the final mesh inherits the coordinate consistency of the neural field while incorporating finer surface details from SAM3D. Details of ray supervision, loss terms, and mesh alignment are provided in Appendix B.

#### 2.1.3 Egocentric Hand Pose Estimation

EgoAERO uses the camera-frame MANO [20] results from HaWoR [21] as the hand initialization, and further applies a lightweight correction to the global hand translation using RGB-D depth information. Specifically, given an RGB sequence, the hand estimator  $ \mathcal{H}_{\psi} $ outputs the MANO articulation, shape parameters, and hand root pose for each frame:

 $$ \begin{pmatrix}\theta_{t},\beta_{t},^{C_{t}}T_{H_{t}}\end{pmatrix}_{t=1}^{T}=\mathcal{H}_{\psi}(I_{1:T}), $$ 

where $\theta_t$ denotes the MANO articulation, $\beta_t$ denotes the shape parameter, and $C_t T_{H_t} \in SE(3)$ represents the rigid transformation from the hand root to the current camera frame $C_t$. To maintain shape consistency across the sequence, EgoAERO uses the sequence-level average shape $\bar{\beta}$ and obtains the hand mesh vertices and joints in the camera frame through MANO forward kinematics.

Since monocular hand pose estimation may suffer from global depth bias, EgoAERO further uses the RGB-D depth map to correct the global translation of the whole hand. Specifically, the system projects MANO vertices onto the RGB-D image, queries RGB-aligned depth values in local neighborhoods, and estimates a translation correction  $ \Delta p_t^C $ from the robust residuals between the visible 3D hand surface observations and the predicted vertices. This correction is applied only to the global hand translation. After this step, the hand mesh is more stably aligned with the RGB-D geometry, providing a reliable hand state for subsequent hand-object contact recovery.

#### 2.1.4 Ego Motion Compensation

Since the egocentric camera is mounted on the head, head motion is mixed into the hand-object trajectories expressed in the camera frame: static objects may exhibit spurious drift, and the hand wrist trajectory may contain camera motion unrelated to manipulation. To address this issue, EgoAERO estimates the camera trajectory with an RGB-D SLAM backend [22] and transforms all frames into a fixed table frame T. In this way, the hand-object states estimated in each camera frame are represented in a unified and stable coordinate frame. To reduce the influence of dynamic hand regions on SLAM, EgoAERO uses hand masks to down-weight hand pixels during camera trajectory estimation, allowing the background and tabletop regions to dominate ego motion recovery. After the transformation, EgoAERO applies only lightweight temporal smoothing to the object trajectory and hand root translation. It does not force the object to remain on the table or impose vertical constraints on the object's local axes, avoiding incorrectly flattening real object rotations during grasping and manipulation.

#### 2.1.5 Adaptive Contact Optimization

Due to egocentric occlusions and hand pose estimation errors, fingertip floating, missing contacts, or local penetrations may still occur during grasping. EgoAERO formulates this problem as a conservative geometry-level contact correction: it keeps the object pose, object mesh, and MANO articulation unchanged, and only applies bounded corrections to the global hand translation and vertices in local contact regions. Specifically, EgoAERO first selects an active manipulation window based on the operation prior and hand-object dis-

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//14a7b624-c485-4ffd-8efd-8ba2751474d9/markdown_4/imgs/img_in_chart_box_564_727_1001_942.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-06-27T08%3A49%3A04Z%2F-1%2F%2Fd805c8206d6c7dc160f939bca3dfe6ef25a9b4051f57946ed23b766f14fc5b10" alt="Image" width="35%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 3: Adaptive contact optimization. Before and after visualization of local hand-object contact correction.</div> </div>


tance, so that corrections are applied only around grasping, moving, and placing stages, avoiding erroneous attraction in non-manipulation periods. For each active frame, EgoAERO constructs three types of contact regions: the thumb pulp, a dynamically selected opposing non-thumb fingertip region, and the thenar region that helps stabilize grasping. The optimization first estimates a constrained whole-hand translation to correct the global hand-object misalignment, and then applies small local geometric offsets to the thumb and opposing fingers to enhance fingertip contact while keeping the palm and wrist stable. Finally, EgoAERO temporally smooths the correction trajectory, applies boundary tapering, and uses penetration push-back to suppress severe local penetrations. This module does not re-solve MANO parameters or modify the object trajectory and geometry, thereby improving the contact consistency and physical usability of the reconstructed data while preserving the original demonstration motion as much as possible. Detailed definitions of contact region selection, whole-hand translation, local finger correction, and penetration push-back are provided in Appendix C.

### 2.2 Two-stage Residual Policy Learning

EgoAERO uses the reconstructed hand-object trajectories for policy learning, converting a single egocentric RGB-D demonstration into an executable dexterous robot manipulation policy. Following human-video-based dexterous imitation and two-stage residual learning ideas [23, 24], the first stage learns stable hand trajectory tracking, while the second stage learns small residual corrections under object and contact constraints. This decomposition avoids directly exploring high-dimensional dexterous hand actions from sparse task rewards.

Stage I: Hand Trajectory Tracking. The first-stage policy does not directly model object dynamics, but instead learns stable tracking of human hand motion with a dexterous robot hand. EgoAERO first reconstructs the human wrist pose and finger keypoint trajectories from a single ego demonstration, and uses them as the reference motion for robot hand tracking. Since the human hand and the dexterous robot hand have different morphologies, EgoAERO first obtains an initial robot hand trajectory through kinematic retargeting to warm-start policy training. However, this initialization only provides a reachable initial action sequence, while the supervision target remains the reconstructed human hand reference trajectory. Based on this, EgoAERO trains a hand tracking policy  $ \pi_I(a_t^I \mid s_t^I) $, so that the robot wrist and finger keypoints follow the human hand motion in the task frame while maintaining smooth actions. This stage does not require the object to be successfully manipulated; instead, it provides a stable hand control prior for the second-stage object-contact residual policy. Details of the reference trajectory definition and reward are provided in Appendix D.

Stage II: Object-Contact Residual Policy Learning. Hand tracking alone is usually insufficient for stable object manipulation. Therefore, in the second stage, EgoAERO introduces object states and contact feedback on top of the hand tracking policy, and learns a residual policy  $ \pi_R $ to make small corrections to the base action  $ a_t^I $ produced by the hand tracking policy:

 $$ a_{t}=a_{t}^{I}+\Delta a_{t}^{R},\qquad a_{t}^{I}\sim\pi_{I}(\cdot\mid s_{t}^{I}),\qquad\Delta a_{t}^{R}\sim\pi_{R}(\cdot\mid s_{t}^{R},a_{t}^{I}) $$ 

Here,  $ \Delta a_t^R $ denotes the residual correction. In addition to the hand tracking state,  $ s_t^R $ includes the object reference trajectory, the current object pose and velocity, the object geometry encoding, the hand-object distance, and simulated contact forces. The reward in the residual stage retains the hand imitation term and further incorporates object trajectory tracking and contact constraints, so that the robot can both stay close to the hand motion reconstructed by EgoAERO and drive the object to follow the reference trajectory with stable contact. In this way, EgoAERO converts the reconstructed hand-object trajectories into an executable closed-loop robot policy. Detailed reward definitions are provided in Appendix D.

## 3 Egocentric Demonstration Data Collection and Quality Assessment

EgoAERO enables a closed-loop data collection pipeline for egocentric dexterous manipulation. During collection, the system reconstructs a provisional hand-object trajectory online, evaluates whether it is physically usable, and immediately decides whether to keep, repair, or recapture the demonstration.

Online Ego Data Quality Assessment.

The online assessment is based on bounded recoverability: a sequence is considered useful if stable hand-object contact can be recovered through small, local, and interpretable

<div style="text-align: center;"><div style="text-align: center;">Table 1: Ego dataset comparison.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Dataset</td><td style='text-align: center; word-wrap: break-word;'>Scale</td><td style='text-align: center; word-wrap: break-word;'>Obj. state</td><td style='text-align: center; word-wrap: break-word;'>Asset-free</td><td style='text-align: center; word-wrap: break-word;'>Depth</td><td style='text-align: center; word-wrap: break-word;'>SLAM</td><td style='text-align: center; word-wrap: break-word;'>Contact eval.</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Ego4D</td><td style='text-align: center; word-wrap: break-word;'>Large</td><td style='text-align: center; word-wrap: break-word;'>✗</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✗</td><td style='text-align: center; word-wrap: break-word;'>✗</td><td style='text-align: center; word-wrap: break-word;'>✗</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>EgoDex</td><td style='text-align: center; word-wrap: break-word;'>Large</td><td style='text-align: center; word-wrap: break-word;'>✗</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✗</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✗</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>HOI4D</td><td style='text-align: center; word-wrap: break-word;'>Medium</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✗</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✗</td><td style='text-align: center; word-wrap: break-word;'>Partial</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>EgoDex-R</td><td style='text-align: center; word-wrap: break-word;'>Large</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✓</td><td style='text-align: center; word-wrap: break-word;'>✓</td></tr></table>

corrections, without modifying the object trajectory or re-solving hand articulation. EgoAERO evaluates tracking stability, contact consistency, residual penetration, and temporal jitter from the reconstructed hand state, object pose, and object geometry. It outputs three collection decisions: accept, repairable accept, or recapture. Complete scoring functions, thresholds, and per-finger diagnostics are provided in Appendix E.

<div style="text-align: center;"><div style="text-align: center;">Table 2: Dexterous manipulation results. The first four metrics are averaged over successful rollouts.</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Dataset</td><td style='text-align: center; word-wrap: break-word;'>Method</td><td style='text-align: center; word-wrap: break-word;'>$ E_{r}\downarrow $</td><td style='text-align: center; word-wrap: break-word;'>$ E_{t}\downarrow $</td><td style='text-align: center; word-wrap: break-word;'>$ E_{j}\downarrow $</td><td style='text-align: center; word-wrap: break-word;'>$ E_{ft}\downarrow $</td><td style='text-align: center; word-wrap: break-word;'>SR(%) \uparrow</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>EgoDex-R</td><td style='text-align: center; word-wrap: break-word;'>Only Hand Pose</td><td style='text-align: center; word-wrap: break-word;'>28.6</td><td style='text-align: center; word-wrap: break-word;'>4.72</td><td style='text-align: center; word-wrap: break-word;'>3.35</td><td style='text-align: center; word-wrap: break-word;'>2.48</td><td style='text-align: center; word-wrap: break-word;'>9.8</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>EgoDex-R</td><td style='text-align: center; word-wrap: break-word;'>w/o Adaptive Contact Optimization</td><td style='text-align: center; word-wrap: break-word;'>15.4</td><td style='text-align: center; word-wrap: break-word;'>1.36</td><td style='text-align: center; word-wrap: break-word;'>2.93</td><td style='text-align: center; word-wrap: break-word;'>2.18</td><td style='text-align: center; word-wrap: break-word;'>36.2</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>EgoDex-R</td><td style='text-align: center; word-wrap: break-word;'>EgoAERO</td><td style='text-align: center; word-wrap: break-word;'>9.7</td><td style='text-align: center; word-wrap: break-word;'>0.82</td><td style='text-align: center; word-wrap: break-word;'>2.48</td><td style='text-align: center; word-wrap: break-word;'>1.65</td><td style='text-align: center; word-wrap: break-word;'>49.5</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>HOI4D</td><td style='text-align: center; word-wrap: break-word;'>Raw Data (with Object CAD)</td><td style='text-align: center; word-wrap: break-word;'>10.4</td><td style='text-align: center; word-wrap: break-word;'>0.73</td><td style='text-align: center; word-wrap: break-word;'>2.32</td><td style='text-align: center; word-wrap: break-word;'>1.69</td><td style='text-align: center; word-wrap: break-word;'>43.3</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>HOI4D</td><td style='text-align: center; word-wrap: break-word;'>EgoAERO</td><td style='text-align: center; word-wrap: break-word;'>10.9</td><td style='text-align: center; word-wrap: break-word;'>0.68</td><td style='text-align: center; word-wrap: break-word;'>2.44</td><td style='text-align: center; word-wrap: break-word;'>1.58</td><td style='text-align: center; word-wrap: break-word;'>44.7</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">EgoDex-R: An Egocentric Dataset for Dexterous Manipulation Learning. Using this collection loop, we construct EgoDex-R, an egocentric dexterous manipulation dataset collected with FastUMI Ego and without object CAD assets. EgoDex-R contains approximately 4.3M RGB-D frames, 5,600 manipulation sequences, and over 1,000 target object instances across more than 200 daily task categories. As shown in Table 1, EgoDex-R is distinguished by providing contact-consistent hand-object trajectories with object pose, object geometry, depth, SLAM, and contact quality annotations. These structured outputs make the dataset directly usable for robot retargeting, imitation learning, residual policy learning, and simulation replay. Detailed data fields and collection statistics are provided in Appendix F.</div> </div>


## 4 Experiments

We evaluate EgoAERO from two perspectives: (1) Can a single ego RGB-D demonstration reconstructed by EgoAERO drive a robot to accomplish dexterous manipulation tasks? We evaluate this through both simulation and real-world experiments. (2) Without using object asset priors, can the trajectories be reconstructed by EgoAERO achieve performance comparable to data with object priors?

### 4.1 Simulation Experiment Setting

All simulation experiments are conducted in Isaac Gym [25] with the two-stage residual policy learning pipeline in Sec. 2.2. For each ego demonstration, EgoAERO reconstructs an object mesh, an object reference trajectory, and a human hand motion trajectory, which are then converted into a dexterous manipulation task in simulation. We evaluate two settings. First, on EgoDex-R, we test whether a single ego demonstration can drive manipulation, and compare the full EgoAERO pipeline with two ablations: Only Hand Pose and w/o Adaptive Contact Optimization. Second, on HOI4D, we compare EgoAERO's asset-free reconstruction from raw RGB-D input with trajectories reconstructed using object CAD assets. We report object rotation error ( $ E_{r} $), object translation error ( $ E_{t} $), mean joint position error ( $ E_{j} $), mean fingertip position error ( $ E_{ft} $), and success rate (SR). Detailed protocols and metric definitions are provided in Appendix G and Appendix H.

### 4.2 Simulation Experiment Results

Table 2 reports the simulation results. On EgoDex-R, using only hand pose leads to low success rates, showing that hand motion alone is insufficient for dexterous object manipulation without object state and contact information. Introducing asset-free object reconstruction greatly reduces object tracking errors and improves task success, while adaptive contact optimization further improves hand tracking quality and contact stability.

On HOI4D, EgoAERO achieves performance close to the raw CAD-based annotations and performs better on some metrics. One possible reason is that the raw annotations are mainly designed for perception-level reconstruction, whereas EgoAERO further applies adaptive contact optimization before policy learning. This step improves the physical consistency of the hand-object demonstrations by reducing fingertip floating, missing contact, and local penetration. These results indicate that, even without object CAD assets, EgoAERO can reconstruct hand-object demonstrations with consistent object motion and physically plausible contact, providing effective supervision for downstream policy learning.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//57c89bfa-1b47-495b-988b-dc33532d0738/markdown_2/imgs/img_in_image_box_225_150_996_471.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-06-27T08%3A49%3A03Z%2F-1%2F%2F27e743ab91c288cb0cce72601198316e35776f1ba2153c0a3a3fe281fe71644f" alt="Image" width="62%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 4: Qualitative demonstration of EgoAERO. From a single egocentric human video, EgoAERO reconstructs the hand-object trajectory, transfers it to a simulated dexterous hand, and enables real-world robot execution.</div> </div>


### 4.3 Real-World Experiment

We further evaluate EgoAERO on a real robot platform composed of a Unitree G1 humanoid robot and an Inspire Hand, as shown in Fig. 4. For each task, EgoAERO first reconstructs the hand-object trajectory from a single egocentric demonstration, and then trains the two-stage residual policy described in Sec. 2.2. The learned policy generates the dexterous hand and wrist trajectory used for execution. On hardware, the G1 arm tracks the policy-generated wrist motion, while the Inspire Hand executes the policy-generated finger commands. During real-world execution, we do not enforce strict temporal synchronization with the original human demonstration, since the physical robot may move more slowly than the human hand. These experiments verify that EgoAERO can transform a natural ego demonstration into a contact-consistent robot trajectory that can be executed on physical dexterous hardware.

## 5 Conclusion

We presented EgoAERO, an asset-free framework that converts a single egocentric RGB-D human demonstration into structured hand-object trajectories and executable dexterous manipulation policies. EgoAERO combines MLLM-guided semantic initialization, asset-free object tracking and reconstruction, egocentric hand pose estimation, ego-motion compensation, and adaptive contact optimization to recover replayable hand-object demonstrations without requiring object CAD models or pre-scanned meshes. Based on these reconstructed trajectories, a two-stage residual policy learning pipeline transfers natural human demonstrations to dexterous robot execution. Experiments on EgoDex-R and HOI4D show that EgoAERO substantially improves over hand-only imitation, benefits from contact optimization, and achieves performance close to CAD-based trajectories while avoiding object asset priors. Real-world experiments further demonstrate that the reconstructed trajectories can be transferred to physical dexterous hardware.

## 6 Limitations

EgoAERO still has several limitations. Its current pipeline mainly targets single-hand manipulation, and performance can degrade under severe occlusion, reflective objects, fast motion, or inaccurate hand/segmentation estimates. In addition, policy learning is still task-specific and requires simulation training, leaving cross-task generalization and lower training cost as future directions.

## References

[1] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Proceedings of Robotics: Science and Systems (RSS), 2024.

[2] Z. Zhaxizhuoma, K. Liu, C. Guan, Z. Jia, Z. Wu, X. Liu, T. Wang, S. Liang, P. Chen, P. Zhang, H. Song, D. Qu, D. Wang, Z. Wang, N. Cao, Y. Ding, B. Zhao, and X. Li. Fastumi: A scalable and hardware-independent universal manipulation interface with dataset. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 3069–3093. PMLR, 2025.

[3] D. Damen, H. Doughty, G. M. Farinella, S. Fidler, A. Furnari, E. Kazakos, D. Moltisanti, J. Munro, T. Perrett, W. Price, and M. Wray. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European Conference on Computer Vision (ECCV), pages 720–736, 2018.

[4] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu, M. Martin, T. Nagarajan, I. Radosavovic, S. K. Ramakrishnan, F. Ryan, J. Sharma, M. Wray, M. Xu, E. Z. Xu, C. Zhao, S. Bansal, D. Batra, V. Cartillier, S. Crane, T. Do, M. Doulaty, A. Erapalli, C. Feichtenhofer, A. Fragomeni, Q. Fu, A. Gebreselasie, C. Gonzalez, J. Hillis, X. Huang, Y. Huang, W. Jia, W. Khoo, J. Kolar, S. Kottur, A. Kumar, F. Landini, C. Li, Y. Li, K. Mangalam, R. Modhugu, J. Munro, T. Murrell, T. Nishiyasu, W. Price, P. Puentes, M. Ramazanova, L. Sari, K. Somasundaram, A. Southerland, Y. Sugano, R. Tao, M. Vo, Y. Wang, X. Wu, T. Yagi, Y. Zhu, P. Arbelaez, D. Crandall, D. Damen, G. M. Farinella, C. Fuegen, B. Ghanem, V. K. Ithapu, H. Joo, K. Kitani, H. Li, R. Newcombe, A. Oliva, H. S. Park, J. M. Rehg, Y. Sato, J. Shi, M. Z. Shou, A. Torralba, L. Torresani, M. Yan, and J. Malik. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18995–19012, 2022.

[5] R. Hoque, P. Huang, D. J. Yoon, M. Sivapurapu, and J. Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025. URL https://arxiv.org/abs/2505.11709.

[6] T. Kwon, B. Tekin, J. Stühmer, F. Bogo, and M. Pollefeys.  $ H_{2}O $: Two hands manipulating objects for first person interaction recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10138–10148, 2021.

[7] Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21013–21022, 2022.

[8] P. Banerjee, S. Shkodrani, P. Moulon, S. Hampali, S. Han, F. Zhang, L. Zhang, J. Fountain, E. Miller, S. Basol, R. Newcombe, R. Wang, J. J. Engel, and T. Hodan. Hot3d: Hand and object tracking in 3d from egocentric multi-view videos. arXiv preprint arXiv:2411.19167, 2025. URL https://arxiv.org/abs/2411.19167.

[9] S. Hampali, M. Rad, M. Oberweger, and V. Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3196–3206, 2020.

[10] Y.-W. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, J. Tremblay, Y. S. Narang, K. Van Wyk, U. Iqbal, S. Birchfield, J. Kautz, and D. Fox. Dexycb: A benchmark for capturing hand grasping of objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9044–9053, 2021.

[11] L. Yang, K. Li, X. Zhan, J. Lv, W. Xu, J. Li, and C. Lu. Oakink: A large-scale knowledge repository for understanding hand-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20953–20962, 2022.

[12] Z. Fan, O. Taheri, D. Tzionas, M. Kocabas, M. Kaufmann, M. J. Black, and O. Hilliges. Arctic: A dataset for dexterous bimanual hand-object manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12943–12954, 2023.

[13] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4026, 2023.

[14] N. Carion et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025. URL https://arxiv.org/abs/2511.16719.

[15] B. Wen, C. Mitash, B. Ren, and K. E. Bekris. Bundletrack: 6d pose tracking for novel objects without instance or category-level 3d models. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8067–8074, 2021.

[16] B. Wen, J. Tremblay, V. Blukis, S. Tyree, T. Müller, A. Evans, D. Fox, J. Kautz, and S. Birchfield. Bundlesdf: Neural 6-dof tracking and 3d reconstruction of unknown objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 606–617, 2023.

[17] B. Wen, W. Yang, J. Kautz, and S. Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17868–17879, 2024.

[18] M. A. Fischler and R. C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981. doi:10.1145/358669.358692.

[19] SAM 3D Team, X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, A. Lin, J. Liu, Z. Ma, A. Sagar, B. Song, X. Wang, J. Yang, B. Zhang, P. Dollár, G. Gkioxari, M. Feiszli, and J. Malik. Sam 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624, 2025. URL https://arxiv.org/abs/2511.16624.

[20] J. Romero, D. Tzionas, and M. J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, 36(6):245:1–245:17, 2017. doi:10.1145/3130800.3130883.

[21] J. Zhang, J. Deng, C. Ma, and R. A. Potamias. Hawor: World-space hand motion reconstruction from egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1805–1815, 2025.

[22] C. Campos, R. Elvira, J. J. G. Rodríguez, J. M. M. Montiel, and J. D. Tardós. Orb-slam3: An accurate open-source library for visual, visual-inertial, and multimap slam. IEEE Transactions on Robotics, 37(6):1874–1890, 2021. doi:10.1109/TRO.2021.3075644.

[23] Y. Qin, Y.-H. Wu, S. Liu, H. Jiang, R. Yang, Y. Fu, and X. Wang. Dexmv: Imitation learning for dexterous manipulation from human videos. In Proceedings of the European Conference on Computer Vision (ECCV), pages 570–587, 2022.

[24] K. Li, P. Li, T. Liu, Y. Li, and S. Huang. Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6991–7003, 2025.

[25] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021. URL https://arxiv.org/abs/2108.10470.

### A Details of Keyframe Memory-pool Pose Optimization

Memory-frame representation and quality score. EgoAERO maintains a keyframe memory pool  $ \mathcal{P} $ to store informative historical observations for local pose graph optimization. Each memory frame is represented as

 $$ F_{k}=\{I_{k},D_{k},M_{k}^{O},M_{k}^{H},P_{k}^{C},T_{k},q_{k}\}, $$ 

where  $ I_k $ and  $ D_k $ denote the RGB image and depth map,  $ M_k^O $ and  $ M_k^H $ denote the object and hand masks,  $ P_k^C $ is the local object point cloud in the camera frame,  $ T_k $ is the optimized object pose, and  $ q_k $ is the observation quality score.

For a new frame  $ F_{t} $, EgoAERO computes its quality score as

 $$ q_{t}=\alpha_{v}A_{t}^{O}+\alpha_{d}R_{t}^{D}+\alpha_{\theta}C_{t}^{\theta}-\alpha_{h}H_{t}^{\mathrm{o c c}}, $$ 

where  $ A_{t}^{O} $ measures the visible object area,  $ R_{t}^{D} $ measures the valid depth ratio within the object region,  $ C_{t}^{\theta} $ measures the view complementarity with respect to existing memory frames, and  $ H_{t}^{\mathrm{occ}} $ measures the hand occlusion level. A frame is inserted into  $ \mathcal{P} $ only when  $ q_{t} $ is above a threshold and the frame provides additional view coverage.

Keyframe subset selection. Given the current frame  $ F_t $, EgoAERO selects a subset of memory frames  $ \mathcal{K}_t \subset \mathcal{P} $ for local pose graph optimization. For each candidate memory frame  $ F_k $, the selection score is defined as

 $$ s(k,t)=\beta_{o}\operatorname{O v e r l a p}(F_{k},F_{t})-\beta_{r}d_{R}(R_{k},\tilde{R}_{t})+\beta_{q}q_{k}, $$ 

where Overlap $ (F_k, F_t) $ measures the visible-region overlap between the candidate frame and the current frame,  $ d_R(R_k, \tilde{R}_t) $ is the geodesic distance between their object rotations, and  $ q_k $ is the quality score of the candidate memory frame. Here,  $ R_k $ and  $ \tilde{R}_t $ are the rotation components of the optimized pose  $ T_k $ and the coarse pose  $ \tilde{T}_t $, respectively. EgoAERO selects the top-K memory frames according to  $ s(k, t) $ and constructs a local pose graph

 $$ \mathcal{G}_{t}=(\mathcal{V}_{t},\mathcal{E}_{t}),\qquad\mathcal{V}_{t}=\{t\}\cup\mathcal{K}_{t}. $$ 

Pose graph objective. In the local pose graph, the current frame is initialized with the coarse pose  $ \tilde{T}_{t} $, while memory frames are initialized with their previously optimized poses. EgoAERO jointly optimizes the object poses of all nodes:

 $$ \min_{\left\{T_{i}\right\}_{i\in\mathcal{V}_{t}}}\sum_{(i,j)\in\mathcal{E}_{t}}\left[\lambda_{f}E_{\mathrm{f e a t}}(i,j)+\lambda_{g}E_{\mathrm{g e o}}(i,j)\right]+\sum_{i\in\mathcal{V}_{t}}\left[\lambda_{s}E_{\mathrm{s d f}}(i)+\lambda_{m}E_{\mathrm{m a s k}}(i)+\lambda_{p}E_{\mathrm{p o s e}}(i)\right]. $$ 

The feature term  $ E_{feat} $ measures the consistency of cross-frame RGB-D correspondences after transforming them into the object canonical frame. Given the correspondence set  $ C_{ij} $ between frame i and frame j, it is written as

 $$ E_{\mathrm{f e a t}}(i,j)=\sum_{(p_{i},p_{j})\in\mathcal{C}_{i j}}\rho\left(\left\|T_{i}^{-1}p_{i}-T_{j}^{-1}p_{j}\right\|_{2}\right), $$ 

where $p_i$ and $p_j$ are 3D correspondence points in their camera frames, and $\rho(\cdot)$ is a robust kernel.

The geometric term  $ E_{geo} $ enforces point-to-plane consistency between local depth observations. For a point p in frame i, EgoAERO transforms it into frame j using the current pose estimates, finds its projective association in the depth map of frame j, and penalizes the point-to-plane distance:

 $$ E_{\mathrm{g e o}}(i,j)=\sum_{p\in P_{i}^{C}}\rho\left(\left|n_{j}(\hat{p})^{\top}\left(\hat{p}-T_{j}^{-1}T_{i}p\right)\right|\right), $$ 

where  $ \hat{p} $ is the associated 3D point in frame j, and  $ n_{j}(\hat{p}) $ is the corresponding surface normal. The SDF term  $ E_{sdf} $ attracts aligned observations to the current implicit object surface:

 $$ E_{\mathrm{s d f}}(i)=\sum_{p\in P_{i}^{C}}\rho\left(\left|\Omega_{\Theta}(T_{i}^{-1}p)\right|\right), $$ 

where $\Omega_{\Theta}$ denotes the online neural object field defined in the object canonical frame. The mask term $E_{\mathrm{mask}}$ imposes weak silhouette consistency between the projected object model and the observed object mask, while the pose regularization term prevents the solution from drifting too far from the initialization:

 $$ E_{\mathrm{p o s e}}(i)=\left\|\log(\bar{T}_{i}^{-1}T_{i})\right\|_{2}^{2},\qquad\bar{T}_{i}=\left\{\begin{matrix}{\tilde{T}_{t},}&{i=t,}\\ {T_{i}^{\mathrm{o l d}},}&{i\in\mathcal{K}_{t}.}\\ \end{matrix}\right. $$ 

Here,  $ \bar{T}_{i} $ denotes the initialization pose of node i.

Pose update and memory update. The pose variables are updated in the Lie algebra:

 $$ T_{i}\leftarrow\operatorname{e x p}(\delta\xi_{i}^{\wedge})T_{i},\qquad\delta\xi_{i}\in\mathfrak{s e}(3). $$ 

After optimization, the optimized current pose  $ T_{t} $ is used as the final tracking result. If the current frame satisfies the memory update condition, it is inserted into  $ \mathcal{P} $ with its optimized pose and quality score. Selected historical memory frames can also be updated with their refined poses, which helps reduce early tracking errors and improves the consistency of subsequent geometry fusion.

### B Details of Coarse-to-Fine Object Mesh Reconstruction

This appendix provides the details of the object mesh reconstruction module. EgoAERO defines an online neural object field $\Omega_\Theta$ in the object canonical frame $\mathcal{O}$. Given optimized keyframe poses, RGB-D observations are transformed into $\mathcal{O}$ and used to sample points along depth rays. For a sampled point $x \in \mathbb{R}^3$, the geometry network predicts the SDF value $s_x = \Omega_\Theta(x)$.

### C Details of Adaptive Contact Optimization

This appendix provides the implementation details of the adaptive contact optimization described in the main paper. The inputs to the optimization are the hand vertices  $ V_t^{\mathcal{T}} $, hand joints  $ J_t^{\mathcal{T}} $, the object mesh  $ \mathcal{M}_{\mathcal{O}} $, and the object pose  $ \mathcal{T}T_{\mathcal{O},t} = (R_t, q_t) $ in the table frame  $ \mathcal{T} $. After sampling surface points and normals  $ \{(p_i, n_i)\}_{i=1}^{N_s} $ from the object mesh in its local frame, the object surface at frame  $ t $ can be written as

 $$ \begin{array}{r}{\hat{p}_{i,t}=R_{t}p_{i}+q_{t},\qquad\hat{n}_{i,t}=R_{t}n_{i}.}\end{array} $$ 

The optimization keeps the object pose, object mesh, MANO shape, and original articulation unchanged, and only updates the replay hand geometry:

 $$ V_{t}^{\mathcal{T}}\to V_{t}^{r\mathcal{T}},\qquad J_{t}^{\mathcal{T}}\to J_{t}^{r\mathcal{T}}. $$ 

Active window and contact regions. The optimization is applied only to the active frame set W. In practice, W is determined by hand validity, script-specified frame ranges, and an optional operation prior. If semantic stage priors are available, only the grasp, move, and place stages, together with their padded ranges, are retained. For each active frame, EgoAERO constructs three candidate contact regions: the thumb pulp  $ C_t^{thumb} $, the opposing non-thumb fingertip region  $ C_t^{opp} $, and the thenar region  $ C_t^{hukou} $. The opposing finger is not fixed, but dynamically selected frame by frame from the index, middle, ring, and little fingers according to the closest distance to the object:

 $$ f_{t}^{{o p p}}=\operatorname{a r g}\operatorname*{m i n}_{f\in\{\operatorname{i n d e x,m i d d l e,r i n g,l i t t l e}\}}D_{f}(t), $$ 

where  $ D_{f}(t) $ denotes the nearest-neighbor distance statistic between the fingertip vertices of finger f and the object surface. This strategy allows the optimization to adapt to different contact patterns in two-finger, three-finger, and multi-finger grasps.

Global hand translation. Let  $ \pi_t(x) $ denote the nearest point of a hand point  $ x $ on the current object surface, and let  $ n_{\pi_t(x)} $ be the corresponding surface normal. The signed distance is approximated as

 $$ s_{t}(x)=\left(x-\pi_{t}(x)\right)^{\top}n_{\pi_{t}(x)}. $$ 

If a contact point is still floating relative to the target contact gap  $ g_k $, it is pulled closer along the object normal. For a contact region  $ C_t^k $, the attraction direction is defined as

 $$ d_{t}^{k}=\frac{1}{\left|\mathcal{C}_{t}^{k}\right|}\sum_{x\in\mathcal{C}_{t}^{k}}-n_{\pi_{t}(x)}\operatorname{R e L U}\left(s_{t}(x)-g_{k}\right),\qquad k\in\{\operatorname{t h u m b},\operatorname{o p p},\operatorname{h u k o u}\}. $$ 

The three regions are aggregated with region weights to obtain the whole-hand translation, which is then clipped by the maximum correction magnitude:

 $$ \Delta_{t}^{r a w}=\frac{\sum_{k}w_{k}d_{t}^{k}}{\sum_{k}w_{k}},\qquad\Delta_{t}^{g l o b a l}=\operatorname{c l i p}\left(\Delta_{t}^{r a w},\Delta_{\operatorname*{m a x}}\right). $$ 

By default, EgoAERO does not apply whole-hand rotation and only uses whole-hand translation to correct the global relative misalignment between the hand and the object.

Temporal smoothing and local finger correction. To avoid jitter caused by frame-wise nearest-neighbor queries, the whole-hand translation is smoothed with a finite-window triangular kernel and multiplied by taper weights at the boundaries of active segments:

 $$ \tilde{\Delta}_{t}^{{g l o b a l}}=b_{t}\sum_{\tau\in\mathcal{N}(t)}K(t-\tau)\Delta_{\tau}^{{g l o b a l}}, $$ 

where K is the triangular kernel and $b_{t}$ is the boundary transition weight. After applying the whole-hand translation, EgoAERO recomputes the contact distances of the thumb and the opposing finger, and estimates small local geometric offsets $\delta_{t}^{thumb}$ and $\delta_{t}^{opp}$. For a vertex of finger $f$, the update is defined as

 $$ v_{i}^{\prime}=v_{i}+\tilde{\Delta}_{t}^{g l o b a l}+\alpha_{i}^{f}\delta_{t}^{f}, $$ 

where  $ \alpha_{i}^{f} $ is determined by the MANO finger chain. Vertices closer to the distal fingertip receive larger weights, while palm and wrist vertices have weights close to zero. The joints are updated in the same way using the corresponding finger-chain weights. This local correction does not re-solve the MANO pose or beta, but only enhances fingertip contact at the replay-geometry level.

Penetration rollback. After contact attraction, EgoAERO performs signed-distance checks on hand points near the object. If obvious deep penetration occurs, the penetrating point set is defined as

 $$ \mathcal{P}_{t}=\{x\mid s_{t}(x)<-\epsilon\},\qquad{d e p t h}(x)=\operatorname{R e L U}(-\epsilon-s_{t}(x)). $$ 

A global push-back vector is then estimated along the object normals:

 $$ r_{t}=\mathrm{c l i p}\left(\frac{1}{|\mathcal{P}_{t}|}\sum_{x\in\mathcal{P}_{t}}d e p t h(x)n_{\pi_{t}(x)},r_{\max}\right), $$ 

and applied to the hand vertices and joints. In implementation, EgoAERO allows light contact or small local penetration, but suppresses obvious deep penetration. The optimized hand geometry is written back to both the table frame and the camera frame for replay, retargeting package construction, and subsequent error analysis.

By default, EgoAERO uses a contact gap of 0.5 mm, a thenar gap of 1.8 mm, a maximum whole-hand translation of 34 mm, a maximum local finger displacement of 15 mm, and a maximum penetration push-back of 8 mm. The upper bound for whole-hand rotation is set to  $ 0^{\circ} $, meaning that whole-hand rotation is disabled by default. Both the whole-hand translation and local finger corrections use a temporal smoothing window of length 9, and a 6-frame boundary transition is applied at the beginning and end of each active segment.

### D Reward Definitions for Two-stage Policy Learning

The first-stage objective is to track the human hand reference trajectory  $ \tau_{h}^{H} $ reconstructed by EgoAERO. The retargeted robot hand trajectory  $ \tau_{h}^{init} $ is only used to initialize policy training, rather than serving as the direct tracking target in the reward. The hand tracking reward is defined as

 $$ r_{t}^{I}=w_{w}r_{t}^{\mathrm{w r i s t}}+w_{f}r_{t}^{\mathrm{f i n g e r}}+w_{s}r_{t}^{\mathrm{s m o o t h}}. $$ 

The wrist tracking reward uses the reconstructed human wrist trajectory as the reference and constrains position, orientation, and velocity:

 $$ r_{t}^{\mathrm{wrist}}=\exp\left(-\lambda_{p}\|p_{W,t}-p_{W,t}^{H}\|_{2}^{2}-\lambda_{R}d_{R}(R_{W,t},R_{W,t}^{H})^{2}-\lambda_{v}\|\dot{p}_{W,t}-\dot{p}_{W,t}^{H}\|_{2}^{2}\right). $$ 

The finger imitation reward encourages the robot fingertips and finger keypoints to track the corresponding human hand keypoints:

 $$ r_{t}^{\mathrm{f i n g e r}}=\frac{1}{|\mathcal{K}_{h}|}\sum_{k\in\mathcal{K}_{h}}\exp\left(-\lambda_{k}\|x_{k,t}-x_{k,t}^{H}\|_{2}^{2}\right), $$ 

where  $ \mathcal{K}_{h} $ denotes the set of hand keypoints with established correspondence after retargeting, including fingertips and finger keypoints. The smoothness reward suppresses high-frequency actions and unnecessary energy consumption, where  $ \tau_{t} $ denotes the actuator torque:

 $$ r_{t}^{\mathrm{s m o o t h}}=\exp\left(-\lambda_{a}\|a_{t}^{I}-a_{t-1}^{I}\|_{2}^{2}-\lambda_{\tau}\|\tau_{t}\odot\dot{q}_{t}\|_{1}\right). $$ 

The second stage trains a residual policy by adding object and contact constraints on top of the hand trajectory tracking reward. The reward is defined as

 $$ \begin{array}{r}{r_{t}^{R}=\eta_{I}r_{t}^{I}+\eta_{o}r_{t}^{\mathrm{o b j}}+\eta_{c}r_{t}^{\mathrm{c o n t a c t}}+\eta_{\Delta}r_{t}^{\mathrm{r e s}}.}\end{array} $$ 

Here, the second-stage  $ r_{t}^{I} $ is computed from the robot state after executing the residual action, so as to prevent the residual policy from breaking the original hand motion structure. The object tracking reward encourages the simulated object to follow the reference object trajectory reconstructed by EgoAERO:

 $$ r_{t}^{\mathrm{o b j}}=\exp\left(-\mu_{p}\|p_{O,t}-p_{O,t}^{\mathrm{r e f}}\|_{2}^{2}-\mu_{R}d_{R}(R_{O,t},R_{O,t}^{\mathrm{r e f}})^{2}-\mu_{v}\|\dot{p}_{O,t}-\dot{p}_{O,t}^{\mathrm{r e f}}\|_{2}^{2}\right). $$ 

The contact reward is applied only to fingers that are expected to be in contact according to the reference trajectory. Let  $ \mathcal{A}_{t}^{\mathrm{ref}} $ denote the reference active contact finger set at frame t,  $ d_{f,t} $ denote the distance from robot finger f to the object surface, and  $ F_{f,t} $ denote the corresponding contact force. The contact reward is

 $$ r_{t}^{\mathrm{c o n t a c t}}=\frac{1}{|\mathcal{A}_{t}^{\mathrm{r e f}}|}\sum_{f\in\mathcal{A}_{t}^{\mathrm{r e f}}}\exp(-\mu_{d}d_{f,t}^{2})\left(1-\exp(-\mu_{F}\|F_{f,t}\|_{2})\right). $$ 

When  $ \mathcal{A}_t^{\text{ref}} $ is empty, this contact term is skipped for the current frame. The residual regularization limits the second-stage policy from deviating excessively from the first-stage hand prior:

 $$ r_{t}^{\mathrm{r e s}}=\exp\left(-\mu_{\Delta}\|\Delta a_{t}^{R}\|_{2}^{2}\right). $$ 

During training, an episode is terminated early if the object pose error, hand tracking error, or noncontact penetration exceeds a predefined threshold, which improves the sampling efficiency of residual policy learning.

### E Details of Online Ego Data Quality Assessment

This appendix provides the detailed definition of the online quality assessment module. Given the coarse trajectory produced by EgoAERO in real time,

 $$ x_{0}=\{H_{t},O_{t},\mathcal{M}_{\mathcal{O}}\}_{t=1}^{T}, $$ 

where  $ H_{t} $ denotes the hand state,  $ O_{t} $ denotes the object pose, and  $ \mathcal{M}_{\mathcal{O}} $ denotes the reconstructed object geometry, the quality assessment module solves a constrained projection:

 $$ x^{\star}=\arg\min_{x}E_{\mathrm{c o n t a c t}}(x)+E_{\mathrm{p e n}}(x)+E_{\mathrm{t e m p}}(x)+\lambda\|x-x_{0}\|^{2}, $$ 

with bounded corrections:

 $$ \|\Delta H_{t}^{f}\|\leq\delta_{\operatorname*{m a x}},\qquad O_{t}=O_{t}^{0}. $$ 

This constraint allows the system to repair small errors around finger contact regions, but prevents it from producing high-quality results by significantly moving the wrist, modifying the object trajectory, or changing the hand articulation.

For each candidate finger $f$ in the active manipulation window $\mathcal{W}$, let $\mathcal{P}_{f}$ denote its fingertip pad vertex set. The module first computes the contact gap before and after the bounded projection:

 $$ g_{t}^{f,\mathrm{b e f o r e}}=\operatorname{m e d i a n}_{v\in\mathcal{P}_{f}}\:d(v_{t},\mathcal{M}_{\mathcal{O}}),\qquad g_{t}^{f,\mathrm{a f t e r}}=\operatorname{m e d i a n}_{v\in\mathcal{P}_{f}}\:d(v_{t}^{\star},\mathcal{M}_{\mathcal{O}}), $$ 

where  $ d(\cdot, \mathcal{M}_{\mathcal{O}}) $ denotes the distance to the object surface. The contact recoverability of finger f is defined as

 $$ {\cal Q}_{\mathrm{r e c}}^{f}=\frac{1}{|\mathcal{W}|}\sum_{t\in\mathcal{W}}\mathbf{1}\left[g_{t}^{f,\mathrm{a f t e r}}<\epsilon_{g}\land\left\|\Delta H_{t}^{f}\right\|<\epsilon_{\Delta}\right]. $$ 

This metric measures whether a finger can recover stable and plausible contact within a limited correction budget, rather than simply checking whether contact is achieved after optimization.

To prevent over-repair, the system records the repair budget usage

 $$ B_{\mathrm{r e p a i r}}=\frac{\mathrm{m e d i a n}_{t,f}\left\|\Delta H_{t}^{f}\right\|}{\delta_{\mathrm{m a x}}}, $$ 

as well as residual penetration, residual contact gaps, and the ratio of object motion without recoverable contact. The final quality score is

 $$ Q=\exp\left(-\alpha R_{\mathrm{a f t e r}}-\beta B_{\mathrm{r e p a i r}}-\gamma U_{\mathrm{u n r e s o l v e d}}\right), $$ 

where  $ R_{after} $ denotes the remaining contact and penetration residuals after repair, and  $ U_{unresolved} $ denotes failure modes that cannot be explained by local correction, such as severe object tracking failure or erroneous hand articulation. In practice, the system also outputs per-finger contact states, failure attribution, repair budget usage, and visualization reports, which support three collection decisions: accept, repairable accept, and recapture.

### F EgoDex-R Dataset Details

Each EgoDex-R sequence contains synchronized raw observations, reconstructed hand-object states, and task-level metadata. The raw observations include the egocentric RGB video, aligned depth maps, camera intrinsics, and timestamps. EgoAERO further provides SLAM camera poses in the table frame, MANO hand pose and shape parameters, hand mesh vertices and joints, target object 6-DoF pose trajectories, reconstructed object meshes, object masks, contact windows, and per-frame quality diagnostics.

In addition to geometric annotations, each sequence contains a task description generated or verified from the collection protocol, including the manipulated object, the intended action, and relevant relational objects when present. We also assign a difficulty score from 1 to 5 using an MLLM-based evaluator. The score considers interaction complexity, hand-object occlusion level, object motion difficulty, contact richness, and the expected difficulty of policy learning. These metadata fields allow EgoDex-R to support filtering, curriculum construction, failure analysis, and task-conditioned policy training.

### G Simulation Experiment Protocol

Datasets and comparisons. For EgoDex-R, we randomly select 100 task sequences to evaluate whether a single egocentric RGB-D demonstration can drive dexterous manipulation. We compare the full EgoAERO pipeline with two ablations: Only Hand Pose, which uses hand motion without reconstructed object geometry and object pose supervision, and w/o Adaptive Contact Optimization, which disables the contact refinement module before policy learning. For HOI4D, we randomly select 100 sequences and compare two sources of demonstration trajectories: Raw Data (with Object CAD), which uses the available object assets and annotations, and EgoAERO, which reconstructs hand-object trajectories directly from raw RGB-D videos without object assets.

Policy learning and evaluation. All methods are evaluated in the same Isaac Gym environment with the same dexterous hand model, reward terms, policy architecture, and training budget. Each reconstructed demonstration is converted into a simulation task consisting of an object mesh, an object reference trajectory, and a hand reference trajectory. We train policies using the two-stage procedure in Sec. 2.2. At test time, each task is evaluated with multiple rollout seeds; success rate is computed over all evaluated rollouts. The object and hand tracking errors in Table 2 are averaged over successful rollouts, while failed rollouts are reflected by the success rate.

### H Definitions of Evaluation Metrics

This appendix provides the definitions of the evaluation metrics used in the simulation experiments. Following ManipTrans [24], we evaluate policy performance from three aspects: object trajectory tracking, hand motion imitation, and task success.

Object rotation error. The average object rotation error  $ E_{r} $ measures the mean rotational deviation between the simulated object pose and the reference object pose:

 $$ \mathrm{E_{r}}=\frac{1}{T}\sum_{t=1}^{T}d_{R}\left(R_{O,t},R_{O,t}^{\mathrm{ref}}\right), $$ 

where  $ R_{O,t} $ is the object rotation in simulation at frame  $ t $,  $ R_{O,t}^{\text{ref}} $ is the reference object rotation reconstructed by EgoAERO, and  $ d_R(\cdot, \cdot) $ denotes the geodesic distance between two rotations.  $ E_r $ is reported in degrees.

Object translation error. The average object translation error  $ E_{t} $ measures the mean positional deviation between the simulated object and the reference object:

 $$ \mathrm{E_{t}}=\frac{1}{T}\sum_{t=1}^{T}\left\|p_{O,t}-p_{O,t}^{\mathrm{ref}}\right\|_{2}, $$ 

where  $ p_{O,t} $ is the simulated object position and  $ p_{O,t}^{\text{ref}} $ is the reference object position.  $ E_t $ is reported in centimeters.

Mean joint position error. The mean joint position error  $ E_{j} $ measures the average joint-level tracking error between the robot hand and the reconstructed human hand:

 $$ \mathrm{E}_{\mathrm{j}}=\frac{1}{T|\mathcal{J}|}\sum_{t=1}^{T}\sum_{j\in\mathcal{J}}\left\|x_{j,t}^{R}-x_{j,t}^{H}\right\|_{2}, $$ 

where $\mathcal{J}$ is the set of evaluated hand joints, $x_{j,t}^{R}$ denotes the position of the $j$-th robot hand joint at frame $t$, and $x_{j,t}^{H}$ denotes the corresponding reconstructed human hand joint position. $\mathrm{E}_{\mathrm{j}}$ is reported in centimeters.

Mean fingertip position error. The mean fingertip position error  $ E_{ft} $ evaluates the fingertip-level imitation quality:

 $$ \mathrm{E}_{\mathrm{f t}}=\frac{1}{T|\mathcal{F}|}\sum_{t=1}^{T}\sum_{f\in\mathcal{F}}\left\|x_{f,t}^{R}-x_{f,t}^{H}\right\|_{2}, $$ 

where  $ \mathcal{F} $ denotes the fingertip set, and  $ x_{f,t}^{R} $ and  $ x_{f,t}^{H} $ are the positions of the f-th robot fingertip and reconstructed human fingertip at frame t, respectively. In our evaluation,  $ |\mathcal{F}| = 5 $.  $ \mathrm{E}_{\mathrm{ft}} $ is reported in centimeters.

Success rate. The success rate SR measures the proportion of rollouts that satisfy the predefined thresholds on both object tracking and hand imitation errors. A rollout is considered successful if

 $$ \mathrm{E_{r}}<\tau_{r},\qquad\mathrm{E_{t}}<\tau_{t},\qquad\mathrm{E_{j}}<\tau_{j},\qquad\mathrm{E_{ft}}<\tau_{ft}. $$ 

By default, we set  $ \tau_r = 30^\circ $,  $ \tau_t = 3 $ cm,  $ \tau_j = 8 $ cm, and  $ \tau_{ft} = 6 $ cm. The success rate is computed as

 $$ \mathrm{SR}=\frac{N_{\mathrm{s u c c e s s}}}{N_{\mathrm{r o l l o u t}}}, $$ 

where  $ N_{success} $ is the number of successful rollouts and  $ N_{rollout} $ is the total number of evaluated rollouts.

