# Low-Rank-Adaptation-of-TrOCR

This repository contains code and resources to finetune an existing transformer model for the task of recognizing car license plates using an Image-to-Text Encoder-Decoder architecture based on the Transformer model. This project uses the model presented in the paper TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.

https://arxiv.org/abs/2109.10282

The metric used is Character Error Rate (CER) which is a metric used to evaluate the performance of optical character recognition (OCR). It is a measure of the accuracy of these systems in converting input data (such as images or spoken language) into text. CER is calculated by comparing the output of the system to a reference or ground truth text and counting the number of characters that are incorrectly recognized or substituted.

## Objective
The primary goal of this project is to adapt a pre-existing transformer model for license plate recognition, leveraging its capability to understand the spatial relationships in images and generate corresponding textual representations.

## Model Architecture
The model architecture is based on an Encoder-Decoder based Transformer. Encoder is similar to that of ViT (Vision Transformer). <br />
![Model Architecture](model_architecture.png?raw=true) <br />

## Dataset
The model will be finetuned on car license plate dataset provided by  <br />
https://universe.roboflow.com/yashwanthworkspace/numbers-identification/dataset/2  <br />

Number of training examples: 1843  <br />
Number of validation examples: 527  <br />
Number of test examples: 263

# Methods/Results

## Benchmark Performances
TODO. Add YOLO.

## 1. TrOCR-base Without Finetuning
Validation CER (Character Error Rate) : 0.21923769507803118

## 2: TrOCR-base With Finetuning
Setup: 10 Epochs. lr=5e-5 using AdamW Optimizer. Batch size = 4 (due to memory constraint) <br />
Validation CER (Character Error Rate) : 0.1397005772005773 <br />

Model seems to overfit after first epoch, training loss decreases while validation error increases. 

## 3: LoRa

Work in Progress

# Limitations
* License plates can vary significantly in format, structure, and character composition across different countries and regions. The model's training data primarily consists of North American license plates, and its performance may degrade when faced with license plates from other parts of the world.
* The model is trained to recognize license plates containing English and numeral characters only. It may not perform accurately on license plates with characters from other languages or character sets.
* TrOCR-BASE model has 334M. One should consider using distilled version of TrOCR-BASE model for production. 


