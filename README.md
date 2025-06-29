# EchoLens
A lens that returns (echoes) a description

![captioned_captioned_IMG-20250618-WA0025](https://github.com/user-attachments/assets/9d6acec0-9122-45de-ab51-09ca5f5826a0)
![captioned_captioned_IMG-20250618-WA0035](https://github.com/user-attachments/assets/7b18d5c1-b690-416d-879e-2ae292882bc9)
![captioned_captioned_WhatsApp Image 2025-06-20 at 19 26 39_1153bd50](https://github.com/user-attachments/assets/b6f7fd47-8dbb-4db3-868b-37870fb44ad9)
![captioned_IMG-20250618-WA0049](https://github.com/user-attachments/assets/db8d14d1-bd11-4b5e-a5bf-c629cc8c694e)

https://github.com/user-attachments/assets/16b8e626-cbbd-464a-961e-a9be14d48778



## Idea
Capture real-time images to generate accurate captions and convert text to speech to guide visually impaired people.

## Major Areas:
- Architecture
- Vision Encoder
- Text Decoder
- Text to Speech
- Additional features and BOT

## Some research paper reference:
1. https://ieeexplore.ieee.org/abstract/document/10890285
2. https://arxiv.org/abs/1411.4555 -- 2015
3. https://arxiv.org/abs/1502.03044 -- 2016
4. https://aclanthology.org/P18-1238/ -- 2018
5. https://arxiv.org/abs/2201.12086 -- 2022

## Solution

- Model -> Transformer Encoder Decoder with EfficientNetB0 as backbone model for image feature extraction.
- Modular image captioning framework using EfficientNetB0 and Transformer-based encoder-decoder.
- On-device visual preprocessing (scene parsing, face/object detection) with real-time audio feedback via text-to-speech.
- Make sure the pipeline is optimized for offline operation, minimal latency, and low power consumption

