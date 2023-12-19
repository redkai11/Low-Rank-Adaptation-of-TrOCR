# Low-Rank-Adaptation-of-TrOCR

This repository contains code and resources to finetune an existing transformer model for the task of recognizing car license plates using an Image-to-Text Encoder-Decoder architecture based on the Transformer model. This project uses the model presented in the paper TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.

https://arxiv.org/abs/2109.10282

The metric used is Character Error Rate (CER) which is a metric used to evaluate the performance of optical character recognition (OCR). It is a measure of the accuracy of these systems in converting input data (such as images or spoken language) into text. CER is calculated by comparing the output of the system to a reference or ground truth text and counting the number of characters that are incorrectly recognized or substituted.

## Objective
The primary goal of this project is to adapt a pre-existing transformer model for license plate recognition, leveraging its capability to understand the spatial relationships in images and generate corresponding textual representations.

## Model Architecture
The model architecture is based on an Encoder-Decoder based Transformer. Encoder is similar to that of ViT (Vision Transformer).

## Dataset
The model will be finetuned on car license plate dataset provided by 

https://universe.roboflow.com/yashwanthworkspace/numbers-identification/dataset/2

Number of training examples: 1843 \\
Number of validation examples: 527 \\
Number of test examples: 263

# Methods/Results

## 1: Direct Finetuning
Setup: 10 Epochs. lr=5e-5.\\
After 3rd epoch, Validation CER increaes so it seems like the model is overfitting or we are overshooting the gradie
Training Speed: 2 iterations  per second, trained on 2 x GPU T4
Validation CER (Character Error Rate) : 0.28778011204481796

## 2: LoRa

Work in Progress

# Limitations
Due to memory/computing constraint, I have used trOCR-small model. 
https://huggingface.co/microsoft/trocr-small-printed
