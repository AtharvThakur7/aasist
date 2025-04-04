# Problem Statement: Real-Time Detection of AI-Generated Human Speech in Real-World Conversations

The goal of this project is to detect AI-generated human speech and explore deepfake detection approaches that are capable of operating in real-time or near real-time on real conversations.



#  Model Selection Goal: 

The goal is to identify the top 3 audio deepfake detection models that best meet Momenta’s core requirements:

 - High Accuracy in Detecting AI-Generated Speech : 
Must handle various deepfake types (TTS, VC) with strong robustness.

- Real-Time or Near Real-Time Capability : 
Fast inference and low computational overhead for live use.

- Robustness in Real Conversations
Effective in noisy, real-world environments across diverse audio sources.


# Research & Model Selection: Top 3 Forgery Detection Models for Momenta

To select the most suitable models for detecting AI-generated speech in real-world conversations, we evaluated top-performing deepfake detection methods from the Audio Deepfake Detection Repository.

## Key Selection Criteria:

- Equal Error Rate (EER %) :  Lower is better → Indicates model accuracy
- t-DCF – Lower is better → Measures cost-effectiveness of detection in real-world use
- Feature Extraction Method – Spectrogram-based vs. raw waveform-based
- Generalization Capability – Performance on unseen attacks and noisy data
- Real-Time Feasibility – Speed and deployability in live systems


## 1. AASIST – Audio Anti-Spoofing using Integrated Spectro-Temporal Features

### Key Technical Innovation:

- End-to-End Feature Extraction: Uses Sinc Filter & RawNet-based architecture for direct processing without handcrafted features
-  AASIST is  Best for generalization & real conversations .
- Graph Attention Layers: Capture long-range audio dependencies for improved robustness


### Reported Performance Metrics:

- EER( Equal Error Rate) : 0.83%.
- Tandem Detection Cost Function (t-DCF) : 0.028.
- Real-Time Feasibility: Moderate.

 ### Relevance to Momenta's Use Case::
 
- Very high accuracy across multiple spoof types (TTS, VC, Replay).
- Strong generalization to noisy and device-variable real-world conversations.
- Strikes a good balance between performance and compute.

### Limitations:
- Moderate inference speed due to complex fusion architecture
- Needs GPU for real-time performance in edge deployments

## 2. RawNet2   - End-to-End Deepfake Detection from Raw Audio Waveforms

### Key Technical Innovation: 
- Processes raw audio waveforms directly using sinc-convolutional layers, eliminating the need for handcrafted feature extraction and capturing more nuanced speech characteristics.​
- Real-Time Detection → No need for manual feature extraction (faster inference than handcrafted methods).
-  Real Conversation Analysis → Learns both speaker identity & fake speech artifacts, improving its robustness.

## Reported Performance Metrics: 
- Demonstrates an EER( Equal Error Rate) of 1.12% and a minimum t-DCF (Tandem Detection Cost Function)  of 0.033. ​


### Relevance to Momenta's Use Case: 
- The direct processing of raw audio allows for real-time or near real-time detection, aligning with the need for analyzing live conversations.​
- Extremely fast and light → great for real-time fraud detection (Slower then ASSIT)

### Potential Limitations: 
- While effective, the model's performance is slightly less optimal compared to AASIST, and it may still require significant computational resources.
- Less generalization on longer conversations or complex attacks.
- Lower robustness in noisy environments


## 3. Dual-Branch Network 

## Key Technical Innovation:
 - Fusion-Based Architecture: Uses both utterance-level (global) and segment-level (local) features
 - Combines multiple feature extraction methods, such as LFCC and CQT, within a dual-branch architecture to enhance detection capabilities.
 - Multi-Task Learning: Learns both classification of real vs. fake and fake speech type recognition.


### Reported Performance Metrics:
- Achieves an EER( Equal Error Rate) of 0.80% and a minimum t-DCF (Tandem Detection Cost Function)of 0.021, indicating robust performance.​
- Real-Time Feasibility:  Moderate

### Relevance to Momenta's Use Case: 
-The fusion of diverse features can improve the detection of various types of AI-generated speech, which is beneficial for analyzing real conversations.
- Best generalization across unseen attacks and speaker variations.

### Limitations:
- Highest compute demand among the three Methods.
- Slower inference speed → may need hardware acceleration.
- More complex training process, harder to iterate quickly.



