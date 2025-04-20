# ThinkInk-Graduation-Project

# ThinkInk
## This project aims to develop a system that can decode brain activity , specifically EEG signals, into text. By harnessing the power of deep learning and NLP, we seek to bridge the gap between thought and its expression in meaningful words , enabling silent communication and empowering individuals with speech impairments.

## Contents
0. [Project Overview](#Project-Overview)
0. [Unity Simulation](#Unity-Simulation) 
0. [Introduction](#Introduction)
0. [Model](#model)
0. [Data Collection](#data-collection)
0. [Normal Reading (NR) vs. Task-Specific Reading (TSR)](#normal-reading-nr-vs-task-specific-reading-tsr)
0. [Dataset](#dataset)
0. [Task Formulation](#Task-Formulation)

 
## **Project Overview**

The **Electroencephalography-to-Text (EEG-to-Text) generation** project lies at the intersection of neuroscience, artificial intelligence, and human-computer interaction. This groundbreaking technology focuses on translating brain activity, captured through EEG signals, directly into natural text. It represents a pivotal innovation in **Brain-Computer Interfaces (BCIs)**, enabling novel applications that enhance communication, accessibility, and productivity.

### Our Brain Translator Model 

![Alt text](https://github.com/MohamedTharwat21/ThinkInk/blob/main/BrainTranslator%20.jpg)

**Figure 0** : This model, BrainTranslator, is designed to convert EEG brain signals into natural language sentences using a sequence-to-sequence (Seq2Seq) architecture with a transformer-based encoder and decoder.


### **Why EEG-to-Text Technology Matters**
- **Purpose**: Provides life-changing solutions for individuals who cannot speak or write due to conditions like ALS, paralysis, or severe motor impairments.  
- **Impact**: By decoding their thoughts into text, the technology restores independence, enhances communication, and improves the quality of life.



![image](https://github.com/user-attachments/assets/bcebd498-c048-4ea4-807c-58151f577f07)

**Figure 1** : Decoding brain signals of a disabled woman.

**Figure src** : https://spectrum.ieee.org/brain-computer-interface-speech


## Unity Simulation






## Introduction

### Motivation

Neurological conditions, such as spinal cord injuries and neuromuscular disorders, can cause individuals to lose their ability to communicate despite retaining intact language and cognitive abilities. This inability to express oneself can drastically diminish their quality of life. Brain-Computer Interfaces (BCIs) offer a potential solution by decoding neural activity into natural language, referred to as **Brain-to-Text**. This approach has the potential to restore communication and significantly improve the lives of affected individuals.

| **Challenges** | **Details** |
|-----------------|-------------|
| **Subject-Dependent EEG Representation** | EEG signals tend to cluster based on individual subjects rather than sentence stimuli. This leads to similar cognitive patterns for different sentences within the same subject. |
| **Semantic-Dependent Text Representation** | Different subjects exhibit varied responses to the same sentence stimulus, making it challenging to generalize EEG signals across subjects. |
| **Many-to-One Generation Problem** | Multiple EEG signals often correspond to the same sentence, creating challenges in training sequence-to-sequence models due to data inconsistency. |
| **Limited Cross-Subject Generalizability** | Subject-dependent EEG signals are difficult to transfer to unseen subjects, significantly degrading model performance when exposed to new data. |


### Proposed Solution
To address these challenges,we aim to re-calibrate subject-dependent EEG representations into semantic-dependent EEG representations, making them more suitable for EEG-to-Text generation tasks.



## **Model**

The EEG-to-Text model is designed to transform **word-level EEG features** into coherent, natural-language sentences. These features are derived after the **Feature Extraction Step**, where raw EEG signals are preprocessed and converted into usable data.  

The model consists of **three key components**, each critical to the processing pipeline:

 

### **1. Word-Level EEG Feature Construction**

This stage creates unified word-level features by concatenating EEG features from different frequency bands corresponding to a single word.  

- **Input Features**: Each frequency band contributes EEG features of size **105**.  
- **Output**: Concatenated word-level features provide a comprehensive representation for each word.

| **Step**                  | **Details**                                |
|---------------------------|--------------------------------------------|
| **EEG Feature Size (Band)** | 105                                        |
| **Output Feature Size**   | Combined EEG features for one word         |

 

### **2. Encoder**

The encoder transforms EEG feature space into the embedding space required by the trained Seq2Seq model.  

#### **Components of the Encoder**
1. **Non-Linear Transformation**:  
   - Maps concatenated EEG features to a higher-dimensional space (**size 840**).  
2. **Transformer Encoder**:  
   - Processes the features further to capture sequential dependencies and richer representations.  

| **Step**                        | **Details**                                          |
|---------------------------------|-----------------------------------------------------|
| **Input EEG Feature Size**      | Word-level features (concatenated EEG features)     |
| **Transformed Feature Size**    | 840                                                |
| **Processing**                  | Sequential dependencies captured via Transformer    |
| **Purpose**                     | Bridges the gap between EEG signals and language generation |

 

### **3. Seq2Seq Model**

This component is responsible for generating the output sentence based on the processed EEG features.  

#### **Components of the Seq2Seq Model**
1. **Encoder**:  
   - Encodes EEG-derived embeddings into meaningful sequences.  
2. **Decoder**:  
   - Decodes the encoded representation to generate natural language sentences.  

- **Feature Size**: Both encoder and decoder operate with features of size **1024**.  
- **Goal**: Ensure high-quality semantic representation and natural language decoding.  

| **Component**          | **Function**                | **Feature Size** |
|-------------------------|----------------------------|------------------|
| **Encoder** | Encodes EEG-derived embeddings         | 1024             |
| **Decoder** | Generates natural language sentences   | 1024             |

 

## **Summary of Model Pipeline**

| **Stage**                 | **Input**                       | **Output**                                  | **Key Processing**                                     |
|---------------------------|----------------------------------|---------------------------------------------|-------------------------------------------------------|
| **Word-Level Construction**| EEG features (size 105 per band) | Unified word-level EEG feature              | Concatenation of frequency band features               |
| **Encoder**                | Word-level EEG feature           | Embedded feature (size 840)                 | Non-linear transformation and sequential encoding     |
| **Seq2Seq Model**          | Embedded feature (size 840)      | Natural language sentence                   | Encoder-decoder sequence generation       |

This modular pipeline ensures that raw EEG signals are effectively translated into meaningful text, enabling practical applications for individuals with communication impairments.




## Data Collection

### **ZuCo 2.0 Corpus: Data Collection and Processing**
This section describes the contents, participants, experimental design, and preprocessing techniques used in the ZuCo 2.0 dataset.

![image](https://github.com/user-attachments/assets/942f50da-4890-410b-96e2-7f4cfe8c73b1)

**Figure 5**: EEG Brain Signals.  

### **1. Participants**


#### Overview
- Total Participants: **19** (data from 1 participant discarded due to technical issues).  
- Final Dataset: **18 participants**.  
- Demographics:  
  - **Mean Age**: 34 years (SD = 8.3).  
  - **Gender**: 10 females.  
  - **Native Language**: English (participants from Australia, Canada, UK, USA, or South Africa).  


#### **Participant Demographics**


| **ID**   | **Age** | **Gender** |
|----------|---------|------------|
| YAC      | 32      | Female     |
| YAG      | 47      | Female     |
| YAK      | 31      | Female     |
| YDG      | 51      | Male       |
| YDR      | 25      | Male       |
| YFR      | 27      | Male       |
| YFS      | 39      | Male       |
| YHS      | 31      | Male       |
| YIS      | 52      | Male       |
| YLS      | 34      | Female     |
| YMD      | 31      | Female     |
| YMS      | 36      | Female     |
| YRH      | 28      | Female     |
| YRP      | 23      | Female     |
| YRK      | 34      | Male       |
| YSD      | 34      | Male       |
| YSL      | 32      | Female     |
| YTL*     | 36      | Male       |


| **Mean Age** | **Gender Distribution** |
|--------------|-------------------------|
| 34           | 44% Male                |


 

#### **Reading Material Statistics**


| **Metric**          | **NR**  | **TSR** |
|---------------------|---------|---------|
| **Sentences**       | 349     | 390     |
| **Sentence Length** | Mean = 19.6 (SD = 8.8), Range = 5-53 | Mean = 21.3 (SD = 9.5), Range = 5-53 |
| **Total Words**     | 6828    | 8310    |
| **Word Types**      | 2412    | 2437    |
| **Word Length**     | Mean = 4.9 (SD = 2.7), Range = 1-29  | Mean = 4.9 (SD = 2.7), Range = 1-21 |
| **Flesch Score**    | 55.38   | 50.76   |



### **2. EEG Data Acquisition**

#### Equipment and Setup
- **EEG System**: 128-channel Geodesic Hydrocel system (Electrical Geodesics).  
- **Sampling Rate**: 500 Hz.  
- **Bandpass Filter**: 0.1 to 100 Hz.  
- **Reference Electrode**: Set at electrode **Cz**.  
- **Electrode Setup**:  
  - Head circumference measured to select the appropriate EEG net size.  
  - Impedance of each electrode kept below **40 kOhm** to ensure proper contact.  
  - Impedance levels checked every 30 minutes after every 50 sentences.  


| **Specification**      | **Details**                   |
|-------------------------|-------------------------------|
| **System**             | Geodesic Hydrocel (128-channel) |
| **Sampling Rate**       | 500 Hz                       |
| **Bandpass**            | 0.1–100 Hz                   |
| **Reference Electrode** | Cz                           |
| **Electrode Impedance** | ≤ 40 kOhm                    |



### **3. Data Preprocessing and Extraction**

#### Raw and Preprocessed Data
- **Tools Used**:  
  - Preprocessing performed using **Automagic (version 1.4.6)** for automatic EEG data cleaning and validation.  
  - **MARA (Multiple Artifact Rejection Algorithm)**: A supervised machine learning algorithm used for automatic artifact rejection.  



![image](https://github.com/user-attachments/assets/9edfb463-0e92-45d5-8ec3-7bc9d8874098)
               **A**


![image](https://github.com/user-attachments/assets/f2290890-278b-40aa-bc63-5054167ccfb6)
               **B**


**Figure 3 and 4** : Visualization EEG data for a single sentence.
**(A)** Raw EEG data during a single sentence. **(B)** Same data as in **(A)** after preprocessing.



#### EEG Channel Selection
- **Channels Used**:  
  - **105 EEG channels** from the scalp recordings.  
  - **9 EOG channels** for artifact removal.  
  - **14 channels** (mainly on the neck and face) were discarded.  

| **Channels**             | **Count**           | **Purpose**              |
|--------------------------|---------------------|--------------------------|
| **EEG Channels**         | 105                 | Scalp recordings         |
| **EOG Channels**         | 9                   | Artifact removal         |
| **Discarded Channels**   | 14                  | Neck and face channels   |

#### Artifact Rejection
- **Artifacts Addressed**: Eye movements and muscle noise.  
- **Artifact Rejection Criterion**: Trials with transient noise above **90 µV** were excluded.


### **4. Synchronization and Feature Extraction**

#### Synchronization
- EEG signals synchronized with eye-tracking data to enable analyses time-locked to fixation onsets.  

#### Oscillatory Power Measures
- **Frequency Bands**:  
  - Theta1 (4–6 Hz), Theta2 (6.5–8 Hz).  
  - Alpha1 (8.5–10 Hz), Alpha2 (10.5–13 Hz).  
  - Beta1 (13.5–18 Hz), Beta2 (18.5–30 Hz).  
  - Gamma1 (30.5–40 Hz), Gamma2 (40–49.5 Hz).  
- **Processing Steps**:  
  - Bandpass filtering for each frequency band.  
  - **Hilbert Transformation** applied to preserve temporal amplitude information.  

| **Frequency Band** | **Range (Hz)** |
|--------------------|----------------|
| **Theta1**         | 4–6            |
| **Theta2**         | 6.5–8          |
| **Alpha1**         | 8.5–10         |
| **Alpha2**         | 10.5–13        |
| **Beta1**          | 13.5–18        |
| **Beta2**          | 18.5–30        |
| **Gamma1**         | 30.5–40        |
| **Gamma2**         | 40–49.5        |

#### Sentence-Level EEG Features
- **Power Calculations**:  
  - Power computed for each frequency band.  
  - Difference in power spectra between frontal left and right homologue electrodes.  
- **Eye-Tracking Features**: Corresponding EEG features computed for each fixation.  

| **Feature**                 | **Description**                                |
|-----------------------------|-----------------------------------------------|
| **Frequency Band Power**    | Power calculated for each band.               |
| **Electrode Differences**   | Power difference between left/right pairs.    |
| **Artifact Rejection**      | Channels excluded for noise > 90 µV.          |


### **5. Summary**
- The data collection process ensures high-quality EEG recordings synchronized with eye-tracking data.  
- Advanced preprocessing techniques, such as MARA and Hilbert transformations, enhance the usability of the dataset for downstream analyses.  
- The dataset captures meaningful EEG signals corresponding to natural reading tasks, enabling in-depth exploration of brain activity and behavior.


## Dataset

**Dataset: ZuCo Benchmark**
The dataset used for this project is derived from the **ZuCo Benchmark**, which combines data from two EEG datasets: **ZuCo [Hollenstein et al., 2018]** and **ZuCo 2.0 [Hollenstein et al., 2020]**. This benchmark provides a rich corpus of EEG signals and eye-tracking data collected during natural reading activities, making it highly suitable for EEG-to-Text research.

## This A New Dataset: ZuCo 2.0 for Studying Natural Reading and Annotation Processes
This A new dataset, ZuCo 2.0, was recorded and preprocessed to investigate the neural correlates of natural reading and annotation tasks using simultaneous eye-tracking and electroencephalography (EEG). This corpus provides gaze and brain activity data for 739 English sentences. 

* **349 sentences** were presented in a **normal reading paradigm**, where participants read naturally without any specific instructions beyond comprehension.
* **390 sentences** were presented in a **task-specific paradigm**, where participants actively searched for a specific semantic relation type within the sentences, acting as a linguistic annotation task.

ZuCo 2.0 complements the existing ZuCo 1.0 dataset by offering data specifically designed to analyze the cognitive processing differences between natural reading and annotation tasks. The data is freely available for download here: https://osf.io/2urht/.


## Normal Reading (NR) vs. Task-Specific Reading (TSR)

The key difference between normal reading (NR) and task-specific reading (TSR) lies in the purpose and focus of the reading task:


![image](https://github.com/user-attachments/assets/4aa58005-48d9-41d2-a2de-d03d1bbba6a6)

**Figure 2: Example sentences on the recording screen: (left) a normal reading sentence, (middle) a control question for a
 normal reading sentence, and (right) a task-specific annotation sentence.**

**Normal Reading (NR):**

* **Objective:** Participants read the sentences naturally, focusing on general comprehension without any specific task.
* **Instructions:** No specific instructions other than to read and understand the text.
* **Example (Figure 2, left):** "He served in the United States Army in World War II, then got a law degree from Tulane University."
* **Control Condition:** Some sentences (12%) are followed by simple comprehension questions with multiple-choice answers, such as: "Which university did he get his degree from?"
* **Cognitive Load:** Reflects natural reading processes and comprehension without added cognitive demands like searching for specific information.

**Task-Specific Reading (TSR):**

* **Objective:** Participants search for a specific relation (e.g., whether a sentence contains a certain type of information or relation).
* **Instructions:** Participants actively annotate the text by deciding whether the relation is present in the sentence.
* **Example (Figure 2, right):** "After this initial success, Ford left Edison Illuminating and, with other investors, formed the Detroit Automobile Company." Participants answer: "Does this sentence contain the founder relation?" (Yes/No).
* **Control Condition:** A minority of sentences (17%) do not include the relation, serving as a control to ensure participants are not biased towards always finding the relation.
* **Cognitive Load:** Higher cognitive load due to the task of actively searching and annotating, rather than passively reading for comprehension.

**Summary of Differences:**

| Feature        | Normal Reading (NR) | Task-Specific Reading (TSR) |
|----------------|----------------------|----------------------------|
| Purpose        | Natural comprehension | Searching for specific relations |
| Focus          | General understanding | Task-specific annotation |
| Cognitive Effort | Low to moderate     | High (due to active searching) |
| Example Interaction | Multiple-choice questions (12% of cases) | Yes/No relation-based annotation |

**In summary, NR simulates everyday reading with occasional comprehension checks, while TSR involves actively searching for a specific piece of information, requiring more focused attention and effort.**

**Examples of Normal Reading (NR) vs. Task-Specific Reading (TSR):**

1. **Normal Reading (NR):**
   * **Task:** Participants read sentences naturally, focusing on comprehension without any specific task.
   * **Example Sentence (Displayed on Screen):** "Albert Einstein developed the theory of relativity, which revolutionized modern physics."
   * **Follow-up Comprehension Question (Control Condition):** "Who developed the theory of relativity?" 
      [1] Isaac Newton
      [2] Albert Einstein
      [3] Nikola Tesla

2. **Task-Specific Reading (TSR):**
   * **Task:** Participants actively search for a specific type of relation in the sentence, e.g., a "scientist-invention" relation.
   * **Example Sentence (Displayed on Screen):** "Marie Curie discovered radium and was awarded two Nobel Prizes for her contributions to science."
   * **Follow-up Question (Annotation Task):** "Does this sentence contain a scientist-invention relation?" 
      [1] Yes
      [2] No

**Key Example Comparisons:**

| Scenario | Normal Reading (NR)                                    | Task-Specific Reading (TSR)                               |
|----------|--------------------------------------------------------|----------------------------------------------------------|
| Sentence Shown | "Barack Obama served as the 44th President of the United States." | "Thomas Edison invented the light bulb in the 19th century." |
| Follow-up Question | "Who was the 44th President?"                     | "Does this sentence contain the inventor relation?"        |
| Answer Options | [1] George Bush, [2] Barack Obama, [3] Bill Clinton | [1] Yes, [2] No                                          |

**Example with Figure 2 Sentences:**

| Feature | Normal Reading Example                                     | Task-Specific Reading Example                               |
|---------|--------------------------------------------------------|----------------------------------------------------------|
| Reading Task | "He served in the United States Army in World War II, then got a law degree from Tulane University." | "After this initial success, Ford left Edison Illuminating and, with other investors, formed the Detroit Automobile Company." |
| Control/Task | "Which university did he get his degree from?"        | "Does this sentence contain the founder relation?"        |
| Answer Options | [1] Austin University, [2] Tulane University, [3] Louisiana State University | [1] Yes, [2] No                                          |

In NR, participants comprehend general content. In TSR, participants evaluate sentences for a specific task or relation.



### **Data Sources**
#### Reading Materials
The EEG signals were recorded while participants read text from two primary sources:
- **Movie Reviews**
- **Wikipedia Articles**

#### EEG Signals
- Each EEG-text pair in the dataset includes a sequence of **word-level EEG features**.
- **Word-Level EEG Features (E)**: For each word, EEG signals consist of 8 frequency bands:
  1. **Theta1 (4–6 Hz)**
  2. **Theta2 (6.5–8 Hz)**
  3. **Alpha1 (8.5–10 Hz)**
  4. **Alpha2 (10.5–13 Hz)**
  5. **Beta1 (13.5–18 Hz)**
  6. **Beta2 (18.5–30 Hz)**
  7. **Gamma1 (30.5–40 Hz)**
  8. **Gamma2 (40–49.5 Hz)**



#### **Feature Construction**
- Each frequency band feature has a **fixed dimension of 105**.
- To construct the final **word-level feature vector**, all 8 frequency band features are concatenated, resulting in a vector of size **840 (E ∈ R⁸⁴⁰)**.
- All features are normalized using **Z-scoring**, as done in prior work (e.g., Willett et al., 2021).



#### **Dataset Splitting**
- The dataset is split into three parts:
  - **Training Set**: 80%
  - **Validation Set**: 10%
  - **Test Set**: 10%

#### Subject Consistency
- Each subset (train, validation, test) is designed to include unique sentences, ensuring **no overlapping sentences** between sets.
- However, the same set of subjects is maintained across all subsets.



#### **Statistics**

|                  | **Training** | **Validation** | **Testing** | 
|------------------|--------------|----------------|-------------|
| **# Pairs**        | 14,567       | 1,811          | 1,821       |
| **# Unique Sentences** | 1,061       | 173            | 146         |
| **# Subjects**     | 30           | 30             | 30          |
| **# Avg. Words**   | 19.89        | 18.80          | 19.23       |

 
Statistics for the ZuCo benchmark. **“# pairs”** means the number of EEG-text pairs, **“# unique sent”** represents the number of unique sentences, **“# subject”** denotes the number of subjects and **“avg.words”** means the average number of words of sentences.


#### **References**
- Hollenstein, N., et al. (2018). *ZuCo: Zurich Cognitive Language Processing Corpus.*
- Hollenstein, N., et al. (2020). *ZuCo 2.0: The Zurich Cognitive Corpus - An Updated Dataset for EEG and Eye-Tracking Research.*
- Willett, D., et al. (2021). *EEG Feature Standardization via Z-Scoring.*
- Wang, S., & Ji, H. (2022). *Unified Representation Learning for EEG-to-Text Tasks.*

This dataset serves as the foundation for training and validating the BrainTranslator model, enabling the transformation of EEG signals into meaningful text representations.



## Task Formulation

**1. Input and Output:**

* The input is a sequence of EEG features denoted as E = {e1, e2, ..., eB}, where each e ∈ R^n is a word-level EEG feature vector.
* The output is a sentence S = {s1, s2, ..., sS}, where s_i represents tokens forming the sentence.

**2. Model Parameters:**

* θ represents the parameters of the sequence-to-sequence model used for generating the text sentence from the EEG features.

**3. Subjects and Dataset Partitioning:**

* Each EEG feature sequence E corresponds to a subject p_i, with all subjects forming a set P.
* During training, EEG-text pairs are used from various subjects in the set P.
* During testing, the sentences are completely unseen, ensuring that the test set also adheres to the same set of subjects P, meaning the model generalizes within the subjects present in the dataset.

**4. Key Constraints:**

* The train, validation, and test sets maintain the same set of subjects P, ensuring consistency in the distribution.


 

