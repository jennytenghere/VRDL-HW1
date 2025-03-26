# vrdl HW1
StudentID: 313553027

# Introduction
This repository contains the codes for VRDL Homework 1, where a 100-class species classification model is built using ResNet as the backbone. The model is designed to classify images into 100 different species categories, utilizing the power of ResNet’s deep learning architecture for feature extraction. The experiments conducted focus on evaluating different modifications and techniques to improve classification performance.  
ResNeXt_50_CBAM.py: ResNeXt50(32x4d) with CBAM  
ResNeXt_50_CBAM_Res2Net.py: ResNeXt50 with CBAM + Res2Net (4 splits + group=8)  

# How to install
GPU: GeForce RTX 4090  
Training time: 100min  
Importance package version:  
 - python=3.11.7
 - numpy=2.1.2
 - pandas=2.2.3
 - torch=2.6.0+cu124 

# Performance snapshot
![image](https://github.com/jennytenghere/VRDL-HW1/blob/main/v11_score.png)
