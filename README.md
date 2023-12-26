# Low-Rank-Adaptation-of-TrOCR

This repository contains code and resources to finetune an existing transformer model for the task of recognizing car license plates using an Image-to-Text Encoder-Decoder architecture based on the Transformer model. This project uses the model presented in the paper [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282).

The metric used is Character Error Rate (CER) which is a metric used to evaluate the performance of optical character recognition (OCR). It is a measure of the accuracy of these systems in converting input data (such as images or spoken language) into text. CER is calculated by comparing the output of the system to a reference or ground truth text and counting the number of characters that are incorrectly recognized or substituted.

## Objective
The primary goal of this project is to adapt a pre-existing transformer model for license plate recognition, leveraging its capability to understand the spatial relationships in images and generate corresponding textual representations.

## Model Architecture
The model architecture is based on an Encoder-Decoder based Transformer. 
* TrOCR uses pretrained Image transformer and text transformer.
* Encoder was intialized with pre-trained ViT-style model (Vision Transformer). <br />
* Decoder was initalized with pre-trained RoBERTa and MiniLM. 
![Model Architecture](model_architecture.png?raw=true) <br />

## Dataset
The model will be finetuned on [car license plate dataset](https://universe.roboflow.com/yashwanthworkspace/numbers-identification/dataset/2) provided by Roboflow

Number of training examples: 1843  <br />
Number of validation examples: 527  <br />
Number of test examples: 263

## Results

![Results](results.png?raw=true) <br />

## Conclusion

The original author of the dataset/project, accessible [here](https://universe.roboflow.com/yashwanthworkspace/numbers-identification), appears to have employed a YOLO-based model, pretrained on the [COCO dataset](https://cocodataset.org/#home), boasting impressive metrics of mAP 99.5%, Precision 99.8%, and Recall 99.8%. While these numbers initially seem impressive, I verified the claims by downloading and evaluating the model. Upon further analysis, the author's model exhibited a Test CER of 0.0158.

Fine-tuning Transformer models on a small dataset often results in training instability, a known issue. To fix this problem, Microsoft propsed LoRA: Low-Rank-Adaptation. Implementing targeted LoRA finetuning, however, led to an enhancement in performance. The refined model achieved an outstanding Validation CER of 0.0119 and Test CER of 0.0122, after training for only 16 epochs. The finetuned and loraFinetuned small models falls short in contrast to the YOLO-based model with a Validation CER of 0.07 and 0.0292 each respectively. 

The fine-tuning process using LoRA has yielded promising reuslts. However, in the case for the base model, the relatively brief training period of 16 epochs raises questions about whether the model has fully converged to its optimal state. I believe that the model is capable of absorbing more information, learning further and reducing the CER. This project not only showcased the effectiveness of fine-tuning a pre-trained large language model but also emphasized the unique contribution of LoRA in the context of a dataset with limited size. The ability to harness the inherent knowledge within a pre-trained model and tailor it for specific tasks, showcasing the flexibility and the advantages of this approach.

## Limitations
* Due to computational limits, I did not calculate Test CER for method 1 and 2.
* License plates can vary significantly in format, structure, and character composition across different countries and regions. The model's training data primarily consists of North American license plates, and its performance may degrade when faced with license plates from other parts of the world.
* The model is trained to recognize license plates containing English and numeral characters only. It may not perform accurately on license plates with characters from other languages or character sets.
* TrOCR-BASE model has 334M which is huge. One should consider using distilled version of TrOCR-BASE LoRA finetuned model for production. 

## References
@misc{ numbers-identification_dataset,
    title = { numbers-identification Dataset },
    type = { Open Source Dataset },
    author = { yashwanthworkspace },
    howpublished = { \url{ https://universe.roboflow.com/yashwanthworkspace/numbers-identification } },
    url = { https://universe.roboflow.com/yashwanthworkspace/numbers-identification },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { nov },
    note = { visited on 2023-12-23 },
}

@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
