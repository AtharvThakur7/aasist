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
- End-to-End Feature Extraction: Uses Sinc Filter & RawNet-based architecture for direct processing without handcrafted features.
-  AASIST is  Best for generalization & real conversations .
- Graph Attention Layers: Capture long-range audio dependencies for improved robustness.
  
### Reported Performance Metrics:
- EER( Equal Error Rate) : 0.83%.
- Tandem Detection Cost Function (t-DCF) : 0.028.
- Real-Time Feasibility: Moderate.

 ### Relevance to Momenta's Use Case: 
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

### Reported Performance Metrics: 
- Demonstrates an EER( Equal Error Rate) of 1.12% and a minimum t-DCF (Tandem Detection Cost Function)  of 0.033. ​
  
### Relevance to Momenta's Use Case: 
- The direct processing of raw audio allows for real-time or near real-time detection, aligning with the need for analyzing live conversations.​
- Extremely fast and light → great for real-time fraud detection (Slower then ASSIT)

### Potential Limitations: 
- While effective, the model's performance is slightly less optimal compared to AASIST, and it may still require significant computational resources.
- Less generalization on longer conversations or complex attacks.
- Lower robustness in noisy environments


## 3. Dual-Branch Network 

### Key Technical Innovation:
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



##  Final Model Selection: AASIST — The Ideal Fit for Audio Deepfake Detection

 AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph) is selected as the best fit for our specific goals and real-world application needs.

 ### Why AASIST Stands Out ?

-  Accurate Detection of AI-Generated Human Speech :
AASIST delivers exceptional performance in detecting synthetic or manipulated speech. With an Equal Error Rate (EER) of just 0.83%, it ranks among the most accurate models in the field.

- Real-Time or Near Real-Time Feasibility :
AASIST strikes an excellent balance between speed and accuracy. Its efficient architecture makes it suitable for real-time or near real-time applications.

-  Robust in Real-World Conversations:
AASIST is highly resilient to background noise, microphone variability, and long-form speech, making it ideal for analyzing natural, unscripted conversations .

- Strong Generalization Across Attack Types:
Using spectro-temporal graph attention network, AASIST learns both local and global dependencies in speech  enables it to effectively detect a broad spectrum of deepfake techniques and unseen attacks.

 - No Handcrafted Feature Engineering Required:
Replaces the  traditional manual feature extraction and  allows for greater scalability and adaptability to diverse datasets without redesigning the input pipeline.



## 📊 Model Comparison Table

| Model               | Key Innovation                                      | Deepfake Detection Accuracy           | Real-Time Feasibility         | Handles Real Conversations (Noise, Variations, Long Speech) | Best Use Case                     |
|--------------------|-----------------------------------------------------|---------------------------------------|-------------------------------|--------------------------------------------------------------|----------------------------------|
|  **AASIST**        | Spectro-Temporal Graph Attention Networks           | ✅✅✅ **EER = 0.83%**                   | ✅✅ Moderate                  | ✅✅✅ **Best** (Handles noise, mic variation & long speech)   | **Best Overall for Momenta**     |
| **RawNet3**       | Raw waveform processing (no feature extraction)      | ✅✅ **EER = 1.12%**                    | ✅✅✅ **Fastest**              | ✅ Good (less effective on longer conversations)             | **Best for Real-Time Processing**|
|  **Dual-Branch Net**| Multi-task learning (Utterance + Segment-Level)     | ✅✅✅ **EER = 0.80% (Most Accurate)**   | ⏳ Slower (computational cost) | ✅✅✅ Great (Adapts well to new & varied AI speech)          | **Best for Learning New Attacks**|


## Implementation & Setup

This project implements the AASIST model — a graph attention-based architecture designed to detect spoofed audio in the ASVspoof2019-LA dataset. The codebase supports training, evaluation, logging, model checkpointing, and reproducibility-focused research workflows.


Clone the Repository 
```bash
  python manage.py runserver
```

Installing dependencies
```bash
pip install -r requirements.txt
```

Data preparation
We train/validate/evaluate AASIST using the ASVspoof 2019 logical access dataset .
```bash
python ./download_dataset.py
```

Ensure the structure:
```bash
ASVspoof2019/
└── LA/
    ├── ASVspoof2019_LA_train/
    ├── ASVspoof2019_LA_dev/
    ├── ASVspoof2019_LA_eval/
    └── ASVspoof2019_LA_cm_protocols/
```
Training
The main.py includes train/validation/evaluation.

To train AASIST :
```bash
python main.py --config ./config/AASIST.conf
```

To evaluate AASIST :

```bash
python main.py --eval --config ./config/AASIST.conf
```

Your Final Output : 

```bash
It shows EER: 0.83%, min t-DCF: 0.0275
```
## 🧩 Challenges Encountered & Solutions

| **Challenge**             | **Description**                                                                 | **How It Was Addressed**                                                                 |
|---------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Slow Training**         | Full training epochs were too time-consuming for testing/tuning.               | Reduced `num_epochs` to 4 and enabled `torch.backends.cudnn.benchmark` for faster training. |
| **Unclear Defaults**      | Flags like `eval_all_best` and `freq_aug` were missing in some configs.        | Added fallback logic with safe defaults (`True`, `False`) for robustness.                |
| **GPU Dependency**        | The original script assumed CUDA availability.                                 | Introduced device check with fallback to CPU, including warning messages and safe exit.  |
| **Evaluation Delay**      | Evaluation was slow and only useful after meaningful training.                 | Delayed evaluation to run only after best-dev checkpoints were found.                    |
| **Reproducibility**       | Randomness in data loading made it difficult to compare experiment runs.       | Used `torch.Generator`, `seed_worker`, and fixed random seeds for consistency.           |
| **Model Save Management** | Needed to separate best and final model checkpoints.                           | Saved models per epoch, `best-dev`, SWA-averaged, and `best-eval` with clear tags.       |
| **Metric Tracking**       | No persistent way to track loss or EER across epochs.                          | Used TensorBoard and flat file logs (`metric_log.txt`, `t-DCF_EER.txt`).                 |




## Assumptions Made

- Epoch count was kept low (e.g., 3) to reduce runtime under constraints
- Dataset is assumed to be pre-downloaded and correctly structured
- Pretrained weights not used; trained from scratch
- SWA is used selectively when updates are made
- Real-time system not required for this phase


