# ThinkInk
## This project aims to develop a system that can decode brain activity , specifically EEG signals, into text. By harnessing the power of deep learning and NLP, we seek to bridge the gap between thought and its expression in meaningful words , enabling silent communication and empowering individuals with speech impairments.

## Contents
0. [Project Overview](#Project-Overview)
0. [Unity Simulation](#Unity-Simulation) 
0. [Model](#model)
0. [Dataset](#dataset)


 
## **Project Overview**

The **Electroencephalography-to-Text (EEG-to-Text) generation** project lies at the intersection of neuroscience, artificial intelligence, and human-computer interaction. This groundbreaking technology focuses on translating brain activity, captured through EEG signals, directly into natural text. It represents a pivotal innovation in **Brain-Computer Interfaces (BCIs)**, enabling novel applications that enhance communication, accessibility, and productivity.

### **Why EEG-to-Text Technology Matters**
- **Purpose**: Provides life-changing solutions for individuals who cannot speak or write due to conditions like ALS, paralysis, or severe motor impairments.  
- **Impact**: By decoding their thoughts into text, the technology restores independence, enhances communication, and improves the quality of life.


## Unity Simulation
[https://drive.google.com/file/d/1PEYYAJ8-8-LfCdbmmV6_J82AdehqO4Tx/view](url)

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


## Dataset

**Dataset: ZuCo Benchmark**
The dataset used for this project is derived from the **ZuCo Benchmark**, which combines data from two EEG datasets: **ZuCo [Hollenstein et al., 2018]** and **ZuCo 2.0 [Hollenstein et al., 2020]**. This benchmark provides a rich corpus of EEG signals and eye-tracking data collected during natural reading activities, making it highly suitable for EEG-to-Text research.

## This A New Dataset: ZuCo 2.0 for Studying Natural Reading and Annotation Processes
This A new dataset, ZuCo 2.0, was recorded and preprocessed to investigate the neural correlates of natural reading and annotation tasks using simultaneous eye-tracking and electroencephalography (EEG). This corpus provides gaze and brain activity data for 739 English sentences. 

* **349 sentences** were presented in a **normal reading paradigm**, where participants read naturally without any specific instructions beyond comprehension.
* **390 sentences** were presented in a **task-specific paradigm**, where participants actively searched for a specific semantic relation type within the sentences, acting as a linguistic annotation task.

ZuCo 2.0 complements the existing ZuCo 1.0 dataset by offering data specifically designed to analyze the cognitive processing differences between natural reading and annotation tasks. The data is freely available for download here: https://osf.io/2urht/.









 

