<!-- GGSIPU Major Project - Dissertation Report -->

<!-- Report Page 1 -->
# Cover Page

Title of the Report: Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text
Sentiment Detection

AIML-452 Major Project - Dissertation

Submitted in partial fulfillment of the requirement for the award of the degree of Bachelor of Technology
in AIML.

Submitted by: NAME OF THE STUDENT, ENROLLMENT NO.

Under the supervision of: NAME OF THE FACULTY SUPERVISOR, DESIGNATION.

Name of the Department, Name of the Institute, Address of the Institute.

May/June 2026

<div style="page-break-after: always;"></div>

<!-- Report Page 2 -->
# Declaration

This is to certify that the material embodied in this Major Project - Dissertation titled Emotionally
Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection being submitted in
the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML
is based on my original work.

It is further certified that this Major Project - Dissertation work has not been submitted in full or in
part to this university or any other university for the award of any other degree or diploma. My
indebtedness to other works has been duly acknowledged at the relevant places.

Name of the Student, Enrollment No.

<div style="page-break-after: always;"></div>

<!-- Report Page 3 -->
# Certificate

This is to certify that the work embodied in this Major Project - Dissertation titled Emotionally
Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection being submitted in
the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML
is original and has been carried out by NAME OF THE STUDENT under my supervision and guidance.

It is further certified that this Major Project - Dissertation work has not been submitted in full or in
part to this university or any other university for the award of any other degree or diploma to the best
of my knowledge and belief.

Name of the Faculty Supervisor, Designation. Name of the HOD, HOD, Name of the Institute.

<div style="page-break-after: always;"></div>

<!-- Report Page 4 -->
# Acknowledgement

I express my sincere gratitude to my faculty supervisor for consistent guidance, technical direction and
academic support during the development of this major project.

I am thankful to the Head of Department, Principal and the Institute for providing the academic
environment and infrastructure needed for project completion.

I also thank mentors, peers, family members and friends for their constructive feedback, encouragement
and patience during design, implementation, testing and report preparation.

<div style="page-break-after: always;"></div>

<!-- Report Page 5 -->
# Abstract

This major project presents an Emotionally Intelligent Animated Mascot Chatbot designed for supportive
human-computer interaction through text, voice, facial expression and emoji-based emotional cues. The
system uses a PyTorch EfficientNetV2-S architecture for facial emotion classification and Grad-CAM visual
explanation, Hugging Face Transformers for free text emotion and sentiment classification,
SpeechRecognition for speech-to-text, NLTK WordNet for synonym-assisted intent understanding, Streamlit
for user interface design and a digital emotional twin log for persistent personalization.

The project avoids TensorFlow, Keras and paid OpenAI APIs. It uses Hugging Face access tokens through
environment variables and public music or video search links for adaptive recommendations. The fusion
engine combines face, voice, text and emoji evidence into a unified User Emotional State Score. The
chatbot then changes tone through therapist, friendly or motivational modes and asks context-specific
reflective questions instead of producing generic FAQ responses.

The report explains the problem statement, software requirement specification, feasibility, architecture,
implementation, training pipeline, testing and future scope according to the GGSIPU Major Project -
Dissertation format.

<div style="page-break-after: always;"></div>

<!-- Report Page 6 -->
# Table of Contents

Declaration i | Certificate ii | Acknowledgement iii | Abstract iv | List of Figures v | List of Tables
vi

Chapter 1 Introduction 1 | Chapter 2 Problem Statement 10 | Chapter 3 Analysis 18 | Chapter 4 Design and
Architecture 33 | Chapter 5 Implementation 46 | Chapter 6 Testing 68 | Chapter 7 Summary and Conclusion
77 | Chapter 8 Limitations and Future Work 81 | Bibliography 86 | Appendix 90

<div style="page-break-after: always;"></div>

<!-- Report Page 7 -->
# List of Figures

Figure 3.1 System Use Case Diagram 29 | Figure 3.2 Context Level DFD 31 | Figure 4.1 Work Breakdown
Structure 34 | Figure 4.2 Module Architecture 37 | Figure 4.3 Activity Flow 41 | Figure 4.4 Class Diagram
44 | Figure 5.1 Streamlit Dashboard Screen 54 | Figure 5.2 Animated Mascot Expressions 57 | Figure 5.3
Grad-CAM Face Analysis 61 | Figure 5.4 Digital Twin Log View 64

<div style="page-break-after: always;"></div>

<!-- Report Page 8 -->
# List of Tables

Table 3.1 Functional Requirements 21 | Table 3.2 Non-functional Requirements 24 | Table 3.3 Feasibility
Study 26 | Table 3.4 Tools and Technologies 28 | Table 5.1 Implementation Modules 48 | Table 5.2
Requirements.txt Dependencies 51 | Table 6.1 Test Cases 70 | Table 6.2 Risk and Mitigation Matrix 75

<div style="page-break-after: always;"></div>

<!-- Report Page 9 -->
# CHAPTER 1: INTRODUCTION

The proposed major project is a development-based artificial intelligence system that studies how a
chatbot can become more emotionally aware when it listens to multiple signals rather than only typed
text. Conventional chatbot demonstrations usually return repeated answers because they do not observe the
user's emotional state, voice energy, facial expression, previous interaction pattern or preferred tone.
This project addresses that limitation through a single Python 3 application suitable for execution in
Visual Studio Code on a Windows 11 laptop.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 10 -->
# 1.1 Background

Emotion-aware computing has become relevant because students, professionals and users often interact with
software during stress, study pressure, loneliness, lack of motivation or emotional overload. A
therapeutic-style assistant cannot replace a clinical professional, but it can provide safe reflection,
emotional awareness, adaptive questions and practical next steps. The system therefore uses ethical AI
monitoring and does not claim medical diagnosis.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 11 -->
# 1.2 Project Theme

The project theme is Cognitive Emotional Intelligence and Adaptive Lifestyle Operational System. The
implemented software acts as an animated mascot chatbot that reads face, voice, text and emoji cues,
computes a fused mood state, recommends supportive content and stores a digital emotional twin for
personalization.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 12 -->
# 1.3 Need of the Project

A major project should demonstrate analysis, design, implementation, testing and social usefulness. This
project demonstrates all of these through PyTorch facial learning, transformer-based language
understanding, voice transcription, recommendation logic, Streamlit UI, Grad-CAM explainability and
structured logging.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 13 -->
# 1.4 Scope

The scope includes a single-file Python application, a training pipeline for a local facial-expression
dataset, free Hugging Face API integration, animated CSS mascot, mic-based voice input where Streamlit
supports it, webcam capture, CSV-based digital twin storage, adaptive recommendations and a viva-friendly
setup guide.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 14 -->
# 1.5 Contribution

The main contribution is a complete multimodal emotional intelligence prototype that is demo-ready
without TensorFlow or Keras. It uses EfficientNetV2-S rather than MobileNetV2 because EfficientNetV2 is
newer, more efficient and suitable for transfer learning with modern PyTorch tooling.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 15 -->
# 1.6 Ethical Position

The application clearly states that it offers supportive guidance only. It avoids diagnostic claims,
shows confidence bands and warns the user when fewer modalities are available. These design choices are
important for responsible AI [1].

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 16 -->
# 1.7 Report Organization

Chapter 1 introduces the project. Chapter 2 defines the problem and objectives. Chapter 3 presents SRS,
feasibility, tools and diagrams. Chapter 4 explains architecture. Chapter 5 details implementation and
setup. Chapter 6 covers testing. Chapter 7 concludes the work. Chapter 8 discusses limitations and future
scope.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 17 -->
# 1.8 Execution Environment

The target machine is a Windows 11 laptop with Intel Core i5 class processor. The application is kept
lightweight by using a Streamlit web UI, CPU-compatible PyTorch inference and optional model training
with small batch sizes.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 18 -->
# 1.9 Major Project Category

This is a Major Project - Dissertation report for AIML-452. It is not an internship report and therefore
focuses on problem formulation, system design, implementation, evaluation and academic references.

IEEE citation placement: multimodal emotion recognition, transformer language models and responsible AI
are cited in the relevant technical chapters [1]-[6].

<div style="page-break-after: always;"></div>

<!-- Report Page 19 -->
# CHAPTER 2: PROBLEM STATEMENT

Most chatbot demonstrations fail to identify the user's mindset because they are limited to typed input
and static FAQ patterns. Such systems do not use speech, facial expression, context history or adaptive
personality. The project problem is to design and implement a chatbot that can infer emotional state from
multiple modalities and respond in a more meaningful, non-generic manner.

<div style="page-break-after: always;"></div>

<!-- Report Page 20 -->
# 2.1 Problem Definition

To develop a free, single-file Python 3 application that uses PyTorch EfficientNetV2-S, Hugging Face
Transformers, NLTK, OpenCV, SpeechRecognition and Streamlit to detect user emotion from text, voice and
facial cues, fuse the detected signals into a User Emotional State Score and display an animated mascot
chatbot that adapts its tone and recommendations.

<div style="page-break-after: always;"></div>

<!-- Report Page 21 -->
# 2.2 Objectives

The objectives are: implement text emotion detection using free transformer models, implement voice
transcription and energy-based mood estimation, implement facial emotion detection using OpenCV and
PyTorch EfficientNetV2-S, implement Grad-CAM explainability, implement multimodal fusion, implement
animated mascot UI, implement adaptive chatbot response generation and implement a digital twin log.

<div style="page-break-after: always;"></div>

<!-- Report Page 22 -->
# 2.3 Real World Relevance

Students often face academic stress, examination pressure and emotional fatigue. A supportive AI
companion can ask reflective questions, encourage healthy micro-actions and recommend calming or
motivating resources. The project is designed as a technical demonstration rather than a medical product.

<div style="page-break-after: always;"></div>

<!-- Report Page 23 -->
# 2.4 Existing System Limitations

Existing FAQ bots are usually rule-based. Many GPT-style chatbots require paid API keys. Several old
facial emotion demos use MobileNetV2 or TensorFlow/Keras. The proposed system avoids paid APIs, avoids
TensorFlow/Keras and uses PyTorch EfficientNetV2-S for modern CNN-based facial analysis.

<div style="page-break-after: always;"></div>

<!-- Report Page 24 -->
# 2.5 Proposed System

The proposed system integrates four emotional signals: text mood, voice mood, face mood and emoji mood.
It computes weighted fusion, displays confidence, logs emotional history and produces context-aware
replies in therapist, friendly and motivational modes.

<div style="page-break-after: always;"></div>

<!-- Report Page 25 -->
# 2.6 Assumptions

The system assumes webcam and microphone permission when live analysis is required. Hugging Face token is
optional and should be supplied through an environment variable. Without a trained .pth file the facial
module uses OpenCV heuristic fallback.

<div style="page-break-after: always;"></div>

<!-- Report Page 26 -->
# 2.7 Expected Outcomes

Expected outcomes include a working Streamlit interface, animated mascot expression change, text and
voice emotion interpretation, EfficientNetV2-S training option, Grad-CAM visualization, recommendations
and CSV downloads for logs.

<div style="page-break-after: always;"></div>

<!-- Report Page 27 -->
# CHAPTER 3: ANALYSIS

The analysis phase converts the project idea into functional, non-functional and technical requirements.
Since the application uses multiple AI modules, analysis also covers feasibility, user roles, data flow,
external API usage and security of tokens.

<div style="page-break-after: always;"></div>

<!-- Report Page 28 -->
# 3.1 Software Requirement Specification

The application shall run as a Streamlit app. It shall accept username, webcam image, microphone
recording where supported, typed text and emoji mood. It shall process each input through its
corresponding AI or heuristic module and produce a fused state.

<div style="page-break-after: always;"></div>

<!-- Report Page 29 -->
# 3.1.1 Functional Requirements

FR1: detect text emotion using Hugging Face Transformers. FR2: enrich intent using NLTK synonyms. FR3:
record or accept voice and transcribe it. FR4: detect facial emotion using OpenCV and PyTorch. FR5:
compute fused mood. FR6: generate adaptive chatbot reply. FR7: show animated mascot. FR8: log digital
twin data. FR9: provide training pipeline. FR10: export resource catalog.

<div style="page-break-after: always;"></div>

<!-- Report Page 30 -->
# 3.1.2 Non-functional Requirements

The system should be portable, free to execute, understandable for viva, secure with respect to API
tokens, responsive on CPU for demo scale, maintainable as a single file, and transparent about confidence
and ethical limitations.

<div style="page-break-after: always;"></div>

<!-- Report Page 31 -->
# 3.2 Feasibility Study

Technical feasibility is high because Python, PyTorch, Streamlit and Hugging Face are widely supported.
Operational feasibility is high because the UI is browser-based. Economic feasibility is high because the
system avoids paid OpenAI and Spotify API usage. Schedule feasibility is achieved by keeping deployment
as a local Streamlit app.

<div style="page-break-after: always;"></div>

<!-- Report Page 32 -->
# 3.3 Tools and Technologies

Python 3 is the programming language. PyTorch and torchvision provide EfficientNetV2-S. Transformers
provide text models. OpenCV handles face detection and image operations. SpeechRecognition handles
speech-to-text. NLTK supports synonyms. Streamlit provides UI. CSV files store the digital twin and
recommender memory.

<div style="page-break-after: always;"></div>

<!-- Report Page 33 -->
# 3.4 Why Hugging Face Instead of Paid GPT API

Hugging Face is suitable because it provides free accounts, free read tokens and many open models. The
project can use google/flan-t5-small through the free inference route or local fallback. OpenAI and
Spotify developer API flows can introduce cost and key-management issues, while public search links
remain free.

<div style="page-break-after: always;"></div>

<!-- Report Page 34 -->
# 3.5 Security Analysis

Tokens must never be hardcoded in source code. The implementation uses HF_TOKEN or
HUGGINGFACEHUB_API_TOKEN environment variables. The report package intentionally removes any pasted
secrets and documents safe token setup.

<div style="page-break-after: always;"></div>

<!-- Report Page 35 -->
# 3.6 Use Case Diagram

Actor: User. Use cases: Login by username, capture face, record voice, type context, choose emoji,
analyze mood, ask chatbot, receive recommendation, train model, download logs. Actor: Developer. Use
cases: prepare dataset, run app, set token, evaluate output.

<div style="page-break-after: always;"></div>

<!-- Report Page 36 -->
# 3.7 Data Flow Diagram

Input layer collects webcam, audio, text and emoji. Processing layer runs facial, voice and text
analysis. Fusion layer computes score. Response layer generates chatbot reply and recommendation. Storage
layer logs CSV records.

<div style="page-break-after: always;"></div>

<!-- Report Page 37 -->
# 3.8 Data Requirements

Training data is organized under dataset/train, dataset/val and dataset/test with class folders such as
happy, sad, angry and neutral. Images may come from public facial expression datasets when licensing
permits [7], [8].

<div style="page-break-after: always;"></div>

<!-- Report Page 38 -->
# 3.9 API Requirements

The Hugging Face token is optional and free. If missing or API access fails, the system uses a local
rule-based emotional coach so the demo remains functional. This is important for viva reliability.

<div style="page-break-after: always;"></div>

<!-- Report Page 39 -->
# CHAPTER 4: DESIGN AND ARCHITECTURE

The system is designed as a modular single-file application. Although the source code is in one file to
satisfy execution simplicity, the internal structure separates runtime setup, catalog generation, text
analysis, voice analysis, facial analysis, fusion, chatbot generation, training, logging and UI
rendering.

<div style="page-break-after: always;"></div>

<!-- Report Page 40 -->
# 4.1 Structure Chart / Work Breakdown Structure

Level 1: Emotionally Intelligent Animated Mascot Chatbot. Level 2: User Interface, Multimodal Input, AI
Analysis, Fusion Engine, Chatbot Engine, Recommender, Digital Twin, Training Pipeline, Ethics Monitor.
Level 3: individual functions for each processing operation.

<div style="page-break-after: always;"></div>

<!-- Report Page 41 -->
# 4.2 Module: User Interface

The Streamlit interface provides tabs for Live Emotion Studio, Mascot Chat, Training, Twin Memory and
Setup. Sidebar controls collect username, tone mode and emoji mood. The UI is intentionally clean for
demonstration and viva explanation.

<div style="page-break-after: always;"></div>

<!-- Report Page 42 -->
# 4.3 Module: Text Intelligence

Text intelligence includes transformer classification, heuristic fallback, NLTK synonym expansion and
intent inference. The output includes mood, confidence, sentiment label, emotion label, intent and
synonym hints.

<div style="page-break-after: always;"></div>

<!-- Report Page 43 -->
# 4.4 Module: Voice Intelligence

Voice intelligence converts microphone audio to text using SpeechRecognition. It also estimates average
wave energy to identify calm or energetic delivery. The transcript is then passed to the same text
intelligence module.

<div style="page-break-after: always;"></div>

<!-- Report Page 44 -->
# 4.5 Module: Facial Intelligence

Facial intelligence detects the largest face using OpenCV Haar cascade. If a trained EfficientNetV2-S
checkpoint exists, PyTorch inference predicts the class and Grad-CAM highlights important regions.
Otherwise a heuristic fallback identifies broad mood indicators.

<div style="page-break-after: always;"></div>

<!-- Report Page 45 -->
# 4.6 Module: EfficientNetV2-S

EfficientNetV2-S is selected because it provides a modern convolutional architecture with efficient
feature extraction and transfer learning support. The classifier head is replaced with dropout and linear
layers for project-specific emotion classes [3].

<div style="page-break-after: always;"></div>

<!-- Report Page 46 -->
# 4.7 Module: Grad-CAM

Grad-CAM registers forward and backward hooks on the last convolutional layer, weights activation maps by
gradients and overlays a heatmap on the face crop. This improves explainability for viva and evaluation
[4].

<div style="page-break-after: always;"></div>

<!-- Report Page 47 -->
# 4.8 Module: Fusion Engine

The fusion engine assigns weights to face, voice, text and emoji signals. The default weights are 0.35,
0.25, 0.25 and 0.15 respectively. The highest weighted mood becomes the fused emotional state and
normalized score becomes User Emotional State Score.

<div style="page-break-after: always;"></div>

<!-- Report Page 48 -->
# 4.9 Module: Chatbot Engine

The chatbot constructs a system prompt with username, fused mood, tone mode, detected intent, digital
twin summary and analysis snapshot. Hugging Face API is used if HF_TOKEN is present; otherwise a local
coach generates the answer.

<div style="page-break-after: always;"></div>

<!-- Report Page 49 -->
# 4.10 Module: Recommender

The recommender contains mood-tagged YouTube, Spotify and YouTube Music public search resources. A
lightweight reinforcement-learning-style score increases novelty and learns from likes or skips stored in
CSV.

<div style="page-break-after: always;"></div>

<!-- Report Page 50 -->
# 4.11 Class Diagram

CatalogEntry stores resource metadata. FolderDataset wraps image-folder training data. Other modules are
implemented as functions to keep the single-file requirement simple.

<div style="page-break-after: always;"></div>

<!-- Report Page 51 -->
# 4.12 Activity Diagram

Start application, enter username, capture multimodal input, analyze individual signals, fuse moods,
update mascot, show recommendation, ask chatbot, log interaction, optionally train EfficientNetV2-S and
download records.

<div style="page-break-after: always;"></div>

<!-- Report Page 52 -->
# CHAPTER 5: IMPLEMENTATION

The implementation is provided as a complete single-file Python program named app.py. It uses only Python
3 libraries specified in requirements.txt and does not import TensorFlow or Keras. The source code is
kept compatible with VS Code and Streamlit.

<div style="page-break-after: always;"></div>

<!-- Report Page 53 -->
# 5.1 File Structure

The project folder contains app.py, requirements.txt, README.md, docs folder, dataset folder created at
runtime, models folder created at runtime, resource_catalog.csv, cei_twin_log.csv and
recommender_stats.csv.

<div style="page-break-after: always;"></div>

<!-- Report Page 54 -->
# 5.2 Requirements.txt

The requirements file contains streamlit, torch, torchvision, transformers, opencv-python, numpy, nltk,
SpeechRecognition and pyttsx3. These are the only required libraries for the implemented solution.

<div style="page-break-after: always;"></div>

<!-- Report Page 55 -->
# 5.3 VS Code Execution Steps

Create a folder, open it in VS Code, create app.py and requirements.txt, open PowerShell terminal, run
python -m venv .venv, activate .venv, upgrade pip, install requirements and run streamlit run app.py.

<div style="page-break-after: always;"></div>

<!-- Report Page 56 -->
# 5.4 Hugging Face Token Setup

Create a free Hugging Face account, open Settings, create an Access Token with read permission and set it
as environment variable HF_TOKEN. The token must not be written inside source code.

<div style="page-break-after: always;"></div>

<!-- Report Page 57 -->
# 5.5 Streamlit UI Implementation

The Streamlit application uses tabs and sidebar controls. The Live Emotion Studio tab performs analysis.
The Mascot Chat tab handles conversation. The Training tab performs EfficientNetV2-S training. The Twin
Memory tab displays logs and catalog.

<div style="page-break-after: always;"></div>

<!-- Report Page 58 -->
# 5.6 Text Emotion Implementation

The implementation loads AutoTokenizer and AutoModelForSequenceClassification from transformers. It runs
softmax inference using PyTorch tensors. If models cannot load, heuristic keywords keep the app
functional.

<div style="page-break-after: always;"></div>

<!-- Report Page 59 -->
# 5.7 Voice Emotion Implementation

The implementation uses SpeechRecognition to process WAV audio. Voice tone is approximated through
waveform energy. The transcript passes into text emotion analysis, allowing combined semantic and
acoustic reasoning.

<div style="page-break-after: always;"></div>

<!-- Report Page 60 -->
# 5.8 Facial Emotion Implementation

The implementation uses OpenCV face detection and PyTorch EfficientNetV2-S classifier. It supports model
checkpoints stored as emotion_efficientnet_v2_s.pth and metadata stored as JSON.

<div style="page-break-after: always;"></div>

<!-- Report Page 61 -->
# 5.9 Grad-CAM Implementation

The Grad-CAM function identifies the last convolutional module, registers hooks, computes gradients,
averages them spatially, forms a heatmap and overlays it over the detected face region.

<div style="page-break-after: always;"></div>

<!-- Report Page 62 -->
# 5.10 Animated Mascot Implementation

The mascot is implemented through HTML and CSS inside Streamlit. Mood changes modify background color,
face color, glow, eye style and mouth curve. It is lightweight and does not require a separate 3D engine.

<div style="page-break-after: always;"></div>

<!-- Report Page 63 -->
# 5.11 Chatbot Response Implementation

The chatbot response pipeline prepares a structured prompt containing the mood, tone, intent and twin
summary. If Hugging Face API returns a response, it is displayed. Otherwise the local emotional coach
generates a deterministic response.

<div style="page-break-after: always;"></div>

<!-- Report Page 64 -->
# 5.12 Digital Twin Implementation

The digital twin is a CSV log containing user, timestamp, modality moods, fused mood, score,
recommendation, chat query, chat reply and feedback. It helps personalize later recommendations.

<div style="page-break-after: always;"></div>

<!-- Report Page 65 -->
# 5.13 Training Pipeline

The training pipeline reads image folders, constructs a FolderDataset, loads EfficientNetV2-S pretrained
weights, optionally freezes the backbone, trains with AdamW and CrossEntropyLoss and saves a .pth
checkpoint.

<div style="page-break-after: always;"></div>

<!-- Report Page 66 -->
# 5.14 Screenshots to Capture

The student should capture screenshots of the Streamlit home screen, webcam emotion result, Grad-CAM
heatmap, mascot chat response, training metadata, twin memory table and setup guide tab after running the
app.

<div style="page-break-after: always;"></div>

<!-- Report Page 67 -->
# 5.15 Implementation Safety

The implementation contains no hardcoded API key. Any token pasted during experimentation must be revoked
and replaced with an environment variable. This is part of safe software engineering practice.

<div style="page-break-after: always;"></div>

<!-- Report Page 68 -->
# CHAPTER 6: TESTING

Testing verifies whether the software meets functional and non-functional requirements. Since the project
is an AI-enabled application, testing includes execution checks, modality checks, fallback checks,
ethical warnings and usability checks.

<div style="page-break-after: always;"></div>

<!-- Report Page 69 -->
# 6.1 Test Strategy

The strategy combines unit-style function checks, integration checks through Streamlit, manual UI
testing, model fallback testing and dataset-path testing. The objective is demo readiness rather than
production clinical validation.

<div style="page-break-after: always;"></div>

<!-- Report Page 70 -->
# 6.2 Test Case 1: Application Launch

Input: streamlit run app.py. Expected Result: Streamlit opens in the browser and shows all tabs. Status:
Pass when dependencies are installed correctly.

<div style="page-break-after: always;"></div>

<!-- Report Page 71 -->
# 6.3 Test Case 2: Text Emotion

Input: I feel stressed about my exam. Expected Result: text mood becomes sad or energetic, intent becomes
stress_relief or study_focus, confidence is shown and chatbot asks reflective questions.

<div style="page-break-after: always;"></div>

<!-- Report Page 72 -->
# 6.4 Test Case 3: Voice Input

Input: recorded WAV speech. Expected Result: transcript is shown when SpeechRecognition succeeds, energy
is computed and voice mood contributes to fusion.

<div style="page-break-after: always;"></div>

<!-- Report Page 73 -->
# 6.5 Test Case 4: Face Image

Input: webcam capture. Expected Result: a face box is detected, method is trained_efficientnet_v2_s when
checkpoint exists or opencv_heuristic otherwise, and an image is displayed.

<div style="page-break-after: always;"></div>

<!-- Report Page 74 -->
# 6.6 Test Case 5: Fusion

Input: happy emoji, positive text and smiling face. Expected Result: fused mood tends toward happy and
mascot changes to happy visual state.

<div style="page-break-after: always;"></div>

<!-- Report Page 75 -->
# 6.7 Test Case 6: Recommendation

Input: fused sad mood. Expected Result: system recommends sad/comfort/calm resource and stores exposure
in recommender_stats.csv.

<div style="page-break-after: always;"></div>

<!-- Report Page 76 -->
# 6.8 Test Case 7: Chatbot

Input: How can I handle pressure before presentation? Expected Result: response includes one reflection,
exactly three questions and one next action.

<div style="page-break-after: always;"></div>

<!-- Report Page 77 -->
# 6.9 Test Case 8: Training

Input: dataset/train with at least two class folders. Expected Result: EfficientNetV2-S training starts
and saves emotion_efficientnet_v2_s.pth.

<div style="page-break-after: always;"></div>

<!-- Report Page 78 -->
# 6.10 Test Case 9: Token Missing

Input: no HF_TOKEN set. Expected Result: local coach fallback works and application does not crash.

<div style="page-break-after: always;"></div>

<!-- Report Page 79 -->
# 6.11 Test Case 10: Security

Input: repository scan. Expected Result: no hardcoded hf_ token appears in app.py or report files.

<div style="page-break-after: always;"></div>

<!-- Report Page 80 -->
# 6.12 Validation Metrics

The training pipeline can compute accuracy through class prediction. Confusion matrix, precision, recall
and F1 score can be added after a balanced validation dataset is available.

<div style="page-break-after: always;"></div>

<!-- Report Page 81 -->
# 6.13 Risk and Mitigation

Risks include camera permission failure, microphone failure, API downtime, insufficient dataset and
overconfident interpretation. Mitigations include fallbacks, ethical warnings, local coach and clear non-
diagnostic wording.

<div style="page-break-after: always;"></div>

<!-- Report Page 82 -->
# CHAPTER 7: SUMMARY AND CONCLUSION

The project successfully demonstrates a major-project-level AI system that combines multimodal emotion
analysis and an animated mascot chatbot in a single Python file.

<div style="page-break-after: always;"></div>

<!-- Report Page 83 -->
# 7.1 Summary

The application integrates PyTorch EfficientNetV2-S, Grad-CAM, Hugging Face Transformers, NLTK,
SpeechRecognition, OpenCV and Streamlit. It meets the requirement of avoiding TensorFlow, Keras and paid
APIs.

<div style="page-break-after: always;"></div>

<!-- Report Page 84 -->
# 7.2 Technical Achievement

The technical achievement is the fusion of four input modes with transparent score display and a non-
generic chatbot response engine. The system is explainable through Grad-CAM and auditable through CSV
logs.

<div style="page-break-after: always;"></div>

<!-- Report Page 85 -->
# 7.3 Academic Contribution

The project connects deep learning, natural language processing, human-computer interaction, recommender
systems and ethical AI. It is suitable for a viva because each module has a clear purpose and can be
demonstrated separately.

<div style="page-break-after: always;"></div>

<!-- Report Page 86 -->
# 7.4 Conclusion

The Emotionally Intelligent Animated Mascot Chatbot provides a practical and innovative demonstration of
cognitive emotional intelligence in a student-friendly environment. It is not a medical tool, but it
shows how emotionally aware AI can provide safer and more personalized support.

<div style="page-break-after: always;"></div>

<!-- Report Page 87 -->
# CHAPTER 8: LIMITATIONS OF THE PROJECT AND FUTURE WORK

Every AI project has limitations. The proposed application is a strong academic prototype, but real-world
deployment would require larger datasets, stronger validation, privacy review and user studies.

<div style="page-break-after: always;"></div>

<!-- Report Page 88 -->
# 8.1 Limitations

Facial emotion recognition can be biased by lighting, camera angle and dataset imbalance. Voice emotion
estimation uses transcript and energy rather than a dedicated speech-emotion transformer. Hugging Face
free API may be unavailable or rate limited. The mascot is CSS-based rather than a full VRM 3D model.

<div style="page-break-after: always;"></div>

<!-- Report Page 89 -->
# 8.2 Future Work

Future work can include full VRM avatar support, lip synchronization, multilingual emotion models,
dedicated speech-emotion recognition, stronger reinforcement learning recommendation, FER2025 dataset
training, mobile APK packaging and secure cloud deployment.

<div style="page-break-after: always;"></div>

<!-- Report Page 90 -->
# 8.3 Mobile and APK Scope

A free practical route is to host Streamlit and add it to Android home screen as a progressive app-like
experience. A later WebView wrapper can convert the hosted URL into an Android APK.

<div style="page-break-after: always;"></div>

<!-- Report Page 91 -->
# 8.4 Research Scope

The system can be extended using larger multimodal datasets such as MER2023, MER2024 and recent facial
expression datasets, subject to license and access. Evaluation can include macro F1, AUC, latency,
usability and user satisfaction.

<div style="page-break-after: always;"></div>

<!-- Report Page 92 -->
# Bibliography

[1] R. W. Picard, Affective Computing, MIT Press, 1997.

[2] A. Vaswani et al., Attention Is All You Need, Advances in Neural Information Processing Systems,
2017.

[3] M. Tan and Q. V. Le, EfficientNetV2: Smaller Models and Faster Training, International Conference on
Machine Learning, 2021.

[4] R. R. Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization, IEEE International Conference on Computer Vision, 2017.

[5] T. Wolf et al., Transformers: State-of-the-Art Natural Language Processing, Conference on Empirical
Methods in Natural Language Processing: System Demonstrations, 2020.

[6] S. Bird, E. Klein and E. Loper, Natural Language Processing with Python, O'Reilly Media, 2009.

[7] I. J. Goodfellow et al., Challenges in Representation Learning: A Report on Three Machine Learning
Contests, Neural Networks, 2015.

[8] P. Ekman and W. V. Friesen, Constants across Cultures in the Face and Emotion, Journal of Personality
and Social Psychology, 1971.

[9] A. Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, Advances in
Neural Information Processing Systems, 2019.

[10] Streamlit Inc., Streamlit Documentation, 2026.

<div style="page-break-after: always;"></div>

<!-- Report Page 93 -->
# Appendix A: Source Code Listing Page 1

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0001: from __future__ import annotations
0002: 
0003: import csv
0004: import html
0005: import io
0006: import json
0007: import os
0008: import random
0009: import re
0010: import textwrap
0011: import urllib.error
0012: import urllib.request
0013: import wave
0014: from collections import Counter
0015: from dataclasses import asdict, dataclass
0016: from datetime import datetime
0017: from functools import lru_cache
0018: from pathlib import Path
0019: from typing import Any
0020: 
0021: import numpy as np
0022: 
0023: try:
0024:     import cv2
0025: except Exception:
0026:     cv2 = None
0027: 
0028: try:
0029:     import nltk
0030:     from nltk.corpus import wordnet as wn
0031:     NLTK_AVAILABLE = True
0032: except Exception:
0033:     nltk = None
0034:     wn = None
```
<div style="page-break-after: always;"></div>

<!-- Report Page 94 -->
# Appendix A: Source Code Listing Page 2

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0035:     NLTK_AVAILABLE = False
0036: 
0037: try:
0038:     import pyttsx3
0039: except Exception:
0040:     pyttsx3 = None
0041: 
0042: try:
0043:     import speech_recognition as sr
0044: except Exception:
0045:     sr = None
0046: 
0047: try:
0048:     import streamlit as st
0049: except Exception:
0050:     st = None
0051: 
0052: try:
0053:     import torch
0054:     import torch.nn as nn
0055:     from torch.utils.data import DataLoader, Dataset
0056:     from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
0057:     TORCH_AVAILABLE = True
0058: except Exception:
0059:     torch = None
0060:     nn = None
0061:     DataLoader = None
0062:     Dataset = object
0063:     EfficientNet_V2_S_Weights = None
0064:     efficientnet_v2_s = None
0065:     TORCH_AVAILABLE = False
0066: 
0067: try:
0068:     from transformers import AutoModelForSequenceClassification, AutoTokenizer
```
<div style="page-break-after: always;"></div>

<!-- Report Page 95 -->
# Appendix A: Source Code Listing Page 3

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0069:     from transformers.utils import logging as hf_logging
0070:     hf_logging.set_verbosity_error()
0071:     TRANSFORMERS_AVAILABLE = True
0072: except Exception:
0073:     AutoModelForSequenceClassification = None
0074:     AutoTokenizer = None
0075:     TRANSFORMERS_AVAILABLE = False
0076: 
0077: 
0078: APP_TITLE = "Emotionally Intelligent Animated Mascot Chatbot"
0079: APP_SUBTITLE = "PyTorch EfficientNetV2-S, Grad-CAM, voice, face, text emotion fusion, and free Hugging Face API chat."
0080: BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
0081: MODEL_DIR = BASE_DIR / "models"
0082: DATASET_DIR = BASE_DIR / "dataset"
0083: MODEL_PATH = MODEL_DIR / "emotion_efficientnet_v2_s.pth"
0084: MODEL_META_PATH = MODEL_DIR / "emotion_efficientnet_v2_s.json"
0085: TWIN_LOG_PATH = BASE_DIR / "cei_twin_log.csv"
0086: RECOMMENDER_STATS_PATH = BASE_DIR / "recommender_stats.csv"
0087: CATALOG_EXPORT_PATH = BASE_DIR / "resource_catalog.csv"
0088: IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
0089: 
0090: HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
0091: HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "google/flan-t5-small").strip() or "google/flan-t5-small"
0092: TEXT_EMOTION_MODEL = os.getenv("TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base").strip()
0093: TEXT_SENTIMENT_MODEL = os.getenv("TEXT_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english").strip()
0094: 
0095: MOOD_CHOICES = ["happy", "sad", "calm", "energetic"]
0096: EMOJI_MAP = {
0097:     "happy": "happy",
0098:     "sad": "sad",
0099:     "calm": "calm",
0100:     "energetic": "energetic",
0101: }
0102: EMOTION_TO_MOOD = {
```
<div style="page-break-after: always;"></div>

<!-- Report Page 96 -->
# Appendix A: Source Code Listing Page 4

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0103:     "happy": "happy", "joy": "happy", "love": "happy", "positive": "happy",
0104:     "surprise": "energetic", "excitement": "energetic", "angry": "energetic",
0105:     "anger": "energetic", "frustration": "energetic",
0106:     "fear": "sad", "sad": "sad", "sadness": "sad", "negative": "sad",
0107:     "neutral": "calm", "calm": "calm", "relaxed": "calm",
0108: }
0109: DEFAULT_FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
0110: 
0111: FALLBACK_ACTIONS = {
0112:     "happy": "Write one win, one gratitude point, and one next meaningful goal.",
0113:     "sad": "Do two minutes of slow breathing, drink water, and message one trusted person.",
0114:     "calm": "Protect this state with one low-distraction focus sprint.",
0115:     "energetic": "Channel your energy into one important task or a short movement break.",
0116: }
0117: MOOD_QUESTION_BANK = {
0118:     "happy": ["What created this positive shift?", "How can you repeat it?", "Who can you share this progress with?"],
0119:     "sad": ["What feels heaviest right now?", "What helped last time?", "What tiny step can make the next hour easier?"],
0120:     "calm": ["What is protecting your balance?", "Which routine is helping focus?", "What boundary will maintain calm?"],
0121:     "energetic": ["Is the energy useful or overloaded?", "Which priority deserves it first?", "What boundary keeps it healthy?"],
0122: }
0123: TONE_GUIDES = {
0124:     "Therapist": "Warm, reflective, structured, and non-diagnostic.",
0125:     "Friendly": "Natural, caring, and specific.",
0126:     "Motivational": "Positive, action-oriented, and disciplined.",
0127: }
0128: INTENT_SEEDS = {
0129:     "stress_relief": {"stress", "overwhelmed", "pressure", "anxious", "burnout"},
0130:     "motivation": {"motivation", "discipline", "goal", "progress", "improve"},
0131:     "study_focus": {"study", "exam", "assignment", "project", "focus"},
0132:     "confidence": {"confidence", "presentation", "interview", "fear", "nervous"},
0133:     "loneliness": {"alone", "lonely", "isolated", "empty"},
0134:     "self_reflection": {"reflect", "journal", "understand", "meaning"},
0135: }
0136: 
```
<div style="page-break-after: always;"></div>

<!-- Report Page 97 -->
# Appendix A: Source Code Listing Page 5

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0137: 
0138: @dataclass(frozen=True)
0139: class CatalogEntry:
0140:     id: str
0141:     mood: str
0142:     title: str
0143:     url: str
0144:     source: str
0145:     resource_type: str
0146:     playable: bool
0147:     tags: str
0148:     offline_fallback: str
0149: 
0150: 
0151: def require_streamlit() -> None:
0152:     if st is None:
0153:         raise RuntimeError("Streamlit is not installed. Run: streamlit run app.py")
0154: 
0155: 
0156: def utc_now() -> str:
0157:     return datetime.utcnow().isoformat()
0158: 
0159: 
0160: def slugify(value: str) -> str:
0161:     text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
0162:     return text.strip("_") or "unknown"
0163: 
0164: 
0165: def ensure_csv(path: Path, headers: list[str]) -> None:
0166:     if path.exists():
0167:         return
0168:     with path.open("w", newline="", encoding="utf-8") as handle:
0169:         writer = csv.DictWriter(handle, fieldnames=headers)
0170:         writer.writeheader()
```
<div style="page-break-after: always;"></div>

<!-- Report Page 98 -->
# Appendix A: Source Code Listing Page 6

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0171: 
0172: 
0173: def ensure_runtime_files() -> None:
0174:     MODEL_DIR.mkdir(parents=True, exist_ok=True)
0175:     DATASET_DIR.mkdir(parents=True, exist_ok=True)
0176:     ensure_csv(TWIN_LOG_PATH, [
0177:         "User", "TimeUTC", "FaceLabel", "FaceMood", "VoiceMood", "TextMood", "EmojiMood",
0178:         "FusedMood", "StateScore", "ToneMode", "RecommendedId", "RecommendedTitle",
0179:         "RecommendedUrl", "RecommendedSource", "ChatQuery", "ChatReply", "Feedback",
0180:     ])
0181:     ensure_csv(RECOMMENDER_STATS_PATH, ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"])
0182: 
0183: 
0184: def read_rows(path: Path) -> list[dict[str, str]]:
0185:     if not path.exists():
0186:         return []
0187:     with path.open("r", newline="", encoding="utf-8") as handle:
0188:         return list(csv.DictReader(handle))
0189: 
0190: 
0191: def write_rows(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
0192:     with path.open("w", newline="", encoding="utf-8") as handle:
0193:         writer = csv.DictWriter(handle, fieldnames=headers)
0194:         writer.writeheader()
0195:         for row in rows:
0196:             writer.writerow({h: row.get(h, "") for h in headers})
0197: 
0198: 
0199: def append_row(path: Path, headers: list[str], row: dict[str, Any]) -> None:
0200:     exists = path.exists()
0201:     with path.open("a", newline="", encoding="utf-8") as handle:
0202:         writer = csv.DictWriter(handle, fieldnames=headers)
0203:         if not exists:
0204:             writer.writeheader()
```
<div style="page-break-after: always;"></div>

<!-- Report Page 99 -->
# Appendix A: Source Code Listing Page 7

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0205:         writer.writerow({h: row.get(h, "") for h in headers})
0206: 
0207: 
0208: @lru_cache(maxsize=1)
0209: def build_resource_catalog() -> tuple[CatalogEntry, ...]:
0210:     direct = {
0211:         "happy": ["feel good playlist", "gratitude meditation", "confidence boost"],
0212:         "sad": ["self compassion meditation", "comfort songs", "gentle piano healing"],
0213:         "calm": ["lofi focus session", "rain sounds for study", "mindful breathing guide"],
0214:         "energetic": ["workout motivation mix", "focus sprint soundtrack", "productivity music"],
0215:     }
0216:     entries: list[CatalogEntry] = []
0217:     for mood, queries in direct.items():
0218:         for index, query in enumerate(queries, start=1):
0219:             for source, base in [
0220:                 ("YouTube", "https://www.youtube.com/results?search_query="),
0221:                 ("Spotify", "https://open.spotify.com/search/"),
0222:                 ("YouTube Music", "https://music.youtube.com/search?q="),
0223:             ]:
0224:                 entries.append(CatalogEntry(
0225:                     id=f"{mood}_{slugify(source)}_{index}",
0226:                     mood=mood,
0227:                     title=f"{mood.title()} {source} resource: {query.title()}",
0228:                     url=base + urllib.parse.quote_plus(query),
0229:                     source=source,
0230:                     resource_type="public_search",
0231:                     playable=False,
0232:                     tags=f"{mood},{slugify(query)}",
0233:                     offline_fallback=FALLBACK_ACTIONS[mood],
0234:                 ))
0235:     return tuple(entries)
0236: 
0237: 
0238: def catalog_rows() -> list[dict[str, Any]]:
```
<div style="page-break-after: always;"></div>

<!-- Report Page 100 -->
# Appendix A: Source Code Listing Page 8

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0239:     return [asdict(item) for item in build_resource_catalog()]
0240: 
0241: 
0242: def export_catalog(path: Path = CATALOG_EXPORT_PATH) -> Path:
0243:     rows = catalog_rows()
0244:     if rows:
0245:         write_rows(path, list(rows[0].keys()), rows)
0246:     return path
0247: 
0248: 
0249: def get_recommender_stats() -> list[dict[str, str]]:
0250:     ensure_runtime_files()
0251:     return read_rows(RECOMMENDER_STATS_PATH)
0252: 
0253: 
0254: def upsert_feedback(item_id: str, feedback: str = "shown") -> None:
0255:     rows = get_recommender_stats()
0256:     headers = ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"]
0257:     row = next((r for r in rows if r.get("ItemId") == item_id), None)
0258:     if row is None:
0259:         row = {"ItemId": item_id, "Exposures": "0", "Likes": "0", "Skips": "0", "LastShown": "", "LastFeedback": ""}
0260:         rows.append(row)
0261:     exposures = int(row.get("Exposures", "0") or 0)
0262:     likes = int(row.get("Likes", "0") or 0)
0263:     skips = int(row.get("Skips", "0") or 0)
0264:     if feedback == "shown":
0265:         exposures += 1
0266:         row["LastShown"] = utc_now()
0267:     elif feedback == "liked":
0268:         likes += 1
0269:         row["LastFeedback"] = "liked"
0270:     elif feedback == "skipped":
0271:         skips += 1
0272:         row["LastFeedback"] = "skipped"
```
<div style="page-break-after: always;"></div>

<!-- Report Page 101 -->
# Appendix A: Source Code Listing Page 9

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0273:     row.update({"Exposures": str(exposures), "Likes": str(likes), "Skips": str(skips)})
0274:     write_rows(RECOMMENDER_STATS_PATH, headers, rows)
0275: 
0276: 
0277: def user_history(user: str) -> list[str]:
0278:     return [r.get("RecommendedId", "") for r in read_rows(TWIN_LOG_PATH) if r.get("User") == user and r.get("RecommendedId")]
0279: 
0280: 
0281: def recommend_item(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
0282:     candidates = [r for r in catalog_rows() if r["mood"] == mood] or catalog_rows()
0283:     recent = set(user_history(user)[-last_n:])
0284:     stats = {r["ItemId"]: r for r in get_recommender_stats()}
0285:     scored: list[tuple[float, dict[str, Any]]] = []
0286:     for item in candidates:
0287:         s = stats.get(item["id"], {})
0288:         exposures = int(s.get("Exposures", "0") or 0)
0289:         likes = int(s.get("Likes", "0") or 0)
0290:         skips = int(s.get("Skips", "0") or 0)
0291:         score = 0.35 / (exposures + 1) + 0.45 * likes / max(exposures, 1) - 0.20 * skips / max(exposures, 1)
0292:         score -= 0.50 if item["id"] in recent else 0.0
0293:         score += random.uniform(0.0, 0.05)
0294:         scored.append((score, item))
0295:     scored.sort(key=lambda p: p[0], reverse=True)
0296:     choice = scored[0][1]
0297:     upsert_feedback(choice["id"], "shown")
0298:     return choice
0299: 
0300: 
0301: def log_interaction(user: str, face_label: str, face_mood: str, voice_mood: str, text_mood: str, emoji_mood: str,
0302:                     fused_mood: str, state_score: float, tone_mode: str, recommendation: dict[str, Any],
0303:                     chat_query: str = "", chat_reply: str = "", feedback: str = "") -> None:
0304:     append_row(TWIN_LOG_PATH, [
0305:         "User", "TimeUTC", "FaceLabel", "FaceMood", "VoiceMood", "TextMood", "EmojiMood",
0306:         "FusedMood", "StateScore", "ToneMode", "RecommendedId", "RecommendedTitle",
```
<div style="page-break-after: always;"></div>

<!-- Report Page 102 -->
# Appendix A: Source Code Listing Page 10

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0307:         "RecommendedUrl", "RecommendedSource", "ChatQuery", "ChatReply", "Feedback",
0308:     ], {
0309:         "User": user, "TimeUTC": utc_now(), "FaceLabel": face_label, "FaceMood": face_mood,
0310:         "VoiceMood": voice_mood, "TextMood": text_mood, "EmojiMood": emoji_mood, "FusedMood": fused_mood,
0311:         "StateScore": f"{state_score:.4f}", "ToneMode": tone_mode,
0312:         "RecommendedId": recommendation.get("id", ""), "RecommendedTitle": recommendation.get("title", ""),
0313:         "RecommendedUrl": recommendation.get("url", ""), "RecommendedSource": recommendation.get("source", ""),
0314:         "ChatQuery": chat_query, "ChatReply": chat_reply, "Feedback": feedback,
0315:     })
0316: 
0317: 
0318: def twin_summary(user: str, limit: int = 12) -> str:
0319:     rows = [r for r in read_rows(TWIN_LOG_PATH) if r.get("User") == user]
0320:     if not rows:
0321:         return "No digital emotional twin history is available yet."
0322:     moods = [r.get("FusedMood", "calm") for r in rows[-limit:]]
0323:     dominant = Counter(moods).most_common(1)[0][0]
0324:     return f"Dominant mood: {dominant}. Recent sequence: {', '.join(moods[-5:])}."
0325: 
0326: 
0327: def normalize_label_to_mood(label: str) -> str:
0328:     token = slugify(label).replace("_", " ")
0329:     for key, mood in EMOTION_TO_MOOD.items():
0330:         if key in token:
0331:             return mood
0332:     return "calm"
0333: 
0334: 
0335: def tokenize_words(text: str) -> list[str]:
0336:     return re.findall(r"[a-zA-Z']+", text.lower())
0337: 
0338: 
0339: def ensure_nltk_resources() -> None:
0340:     if not NLTK_AVAILABLE:
```
<div style="page-break-after: always;"></div>

<!-- Report Page 103 -->
# Appendix A: Source Code Listing Page 11

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0341:         return
0342:     for corpus in ("wordnet", "omw-1.4"):
0343:         try:
0344:             nltk.data.find(f"corpora/{corpus}")
0345:         except LookupError:
0346:             try:
0347:                 nltk.download(corpus, quiet=True)
0348:             except Exception:
0349:                 return
0350: 
0351: 
0352: @lru_cache(maxsize=256)
0353: def synonyms(word: str) -> tuple[str, ...]:
0354:     if not NLTK_AVAILABLE:
0355:         return tuple()
0356:     ensure_nltk_resources()
0357:     found: list[str] = []
0358:     try:
0359:         for synset in wn.synsets(word):
0360:             for lemma in synset.lemmas():
0361:                 candidate = lemma.name().replace("_", " ").lower().strip()
0362:                 if candidate and candidate != word.lower() and candidate not in found:
0363:                     found.append(candidate)
0364:                 if len(found) >= 8:
0365:                     return tuple(found)
0366:     except Exception:
0367:         return tuple()
0368:     return tuple(found)
0369: 
0370: 
0371: def infer_intent(text: str) -> dict[str, Any]:
0372:     tokens = set(tokenize_words(text))
0373:     best_intent = "general_support"
0374:     best_score = 0
```
<div style="page-break-after: always;"></div>

<!-- Report Page 104 -->
# Appendix A: Source Code Listing Page 12

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0375:     best_hits: list[str] = []
0376:     for intent, seeds in INTENT_SEEDS.items():
0377:         expanded = set(seeds)
0378:         for seed in seeds:
0379:             expanded.update(tokenize_words(" ".join(synonyms(seed))))
0380:         hits = sorted(tokens.intersection(expanded))
0381:         if len(hits) > best_score:
0382:             best_intent, best_score, best_hits = intent, len(hits), hits
0383:     preview = sorted({s for t in list(tokens)[:6] for s in synonyms(t)[:2]})[:8]
0384:     return {"intent": best_intent, "matched_terms": best_hits, "synonyms_preview": preview}
0385: 
0386: 
0387: @lru_cache(maxsize=1)
0388: def load_classifier(model_name: str):
0389:     if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
0390:         return None, None
0391:     try:
0392:         tokenizer = AutoTokenizer.from_pretrained(model_name)
0393:         model = AutoModelForSequenceClassification.from_pretrained(model_name)
0394:         model.eval()
0395:         return tokenizer, model
0396:     except Exception:
0397:         return None, None
0398: 
0399: 
0400: def classify_text(model_name: str, text: str) -> tuple[str, float, dict[str, float]]:
0401:     tokenizer, model = load_classifier(model_name)
0402:     if tokenizer is None or model is None or not TORCH_AVAILABLE:
0403:         return "", 0.0, {}
0404:     with torch.no_grad():
0405:         encoded = tokenizer(text[:512], return_tensors="pt", truncation=True)
0406:         logits = model(**encoded).logits
0407:         probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
0408:     labels = model.config.id2label if hasattr(model, "config") else {}
```
<div style="page-break-after: always;"></div>

<!-- Report Page 105 -->
# Appendix A: Source Code Listing Page 13

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0409:     all_scores = {str(labels.get(i, f"class_{i}")): float(p) for i, p in enumerate(probs)}
0410:     label = max(all_scores, key=all_scores.get)
0411:     return label, all_scores[label], all_scores
0412: 
0413: 
0414: def simple_text_mood(text: str) -> str:
0415:     lower = text.lower()
0416:     if any(t in lower for t in ("sad", "lonely", "cry", "hurt", "grief")):
0417:         return "sad"
0418:     if any(t in lower for t in ("happy", "joy", "love", "great", "awesome")):
0419:         return "happy"
0420:     if any(t in lower for t in ("angry", "mad", "stress", "frustrat", "rage")):
0421:         return "energetic"
0422:     return "calm"
0423: 
0424: 
0425: def analyze_text(text: str) -> dict[str, Any]:
0426:     text = (text or "").strip()
0427:     if not text:
0428:         return {"text": "", "mood": "calm", "confidence": 0.5, "emotion_label": "neutral",
0429:                 "emotion_score": 0.5, "sentiment_label": "NEUTRAL", "sentiment_score": 0.5,
0430:                 "intent": "general_support", "matched_terms": [], "synonyms_preview": [],
0431:                 "mood_scores": {m: 0.0 for m in MOOD_CHOICES}, "method": "empty_text"}
0432:     mood_scores = {m: 0.0 for m in MOOD_CHOICES}
0433:     emotion_label, emotion_score, emotion_all = classify_text(TEXT_EMOTION_MODEL, text)
0434:     sentiment_label, sentiment_score, _ = classify_text(TEXT_SENTIMENT_MODEL, text)
0435:     if emotion_all:
0436:         for label, score in emotion_all.items():
0437:             mood_scores[normalize_label_to_mood(label)] += float(score)
0438:         method = "transformers"
0439:     else:
0440:         emotion_label, emotion_score, method = simple_text_mood(text), 0.55, "heuristic"
0441:         mood_scores[emotion_label] = 1.0
0442:     sentiment_label = sentiment_label.upper() if sentiment_label else "NEUTRAL"
```
<div style="page-break-after: always;"></div>

<!-- Report Page 106 -->
# Appendix A: Source Code Listing Page 14

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0443:     sentiment_score = sentiment_score or 0.5
0444:     if sentiment_label.startswith("NEG"):
0445:         mood_scores["sad"] += 0.12
0446:     if sentiment_label.startswith("POS"):
0447:         mood_scores["happy"] += 0.12
0448:     intent = infer_intent(text)
0449:     mood = max(mood_scores.items(), key=lambda p: p[1])[0]
0450:     return {"text": text, "mood": mood, "confidence": float(max(max(mood_scores.values()), emotion_score, sentiment_score)),
0451:             "emotion_label": emotion_label or "neutral", "emotion_score": float(emotion_score),
0452:             "sentiment_label": sentiment_label, "sentiment_score": float(sentiment_score),
0453:             "intent": intent["intent"], "matched_terms": intent["matched_terms"],
0454:             "synonyms_preview": intent["synonyms_preview"], "mood_scores": mood_scores, "method": method}
0455: 
0456: 
0457: def transcribe_audio(audio_bytes: bytes | None) -> tuple[str, str]:
0458:     if not audio_bytes:
0459:         return "", "no_audio"
0460:     if sr is None:
0461:         return "", "speechrecognition_missing"
0462:     recognizer = sr.Recognizer()
0463:     try:
0464:         with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
0465:             audio = recognizer.record(source)
0466:         return recognizer.recognize_google(audio).strip(), "google_web_speech"
0467:     except Exception:
0468:         return "", "transcription_failed"
0469: 
0470: 
0471: def wav_energy(audio_bytes: bytes | None) -> float:
0472:     if not audio_bytes:
0473:         return 0.0
0474:     try:
0475:         with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
0476:             frames = wf.readframes(wf.getnframes())
```
<div style="page-break-after: always;"></div>

<!-- Report Page 107 -->
# Appendix A: Source Code Listing Page 15

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0477:             sample_width = wf.getsampwidth()
0478:             channels = wf.getnchannels()
0479:         if sample_width == 1:
0480:             data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
0481:             max_abs = 128.0
0482:         elif sample_width == 2:
0483:             data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
0484:             max_abs = 32768.0
0485:         else:
0486:             return 0.0
0487:         if channels > 1:
0488:             data = data.reshape(-1, channels).mean(axis=1)
0489:         return float(np.mean(np.abs(data)) / max_abs)
0490:     except Exception:
0491:         return 0.0
0492: 
0493: 
0494: def analyze_voice(audio_bytes: bytes | None, username: str) -> dict[str, Any]:
0495:     transcript, method = transcribe_audio(audio_bytes)
0496:     text_result = analyze_text(transcript) if transcript else analyze_text("")
0497:     scores = dict(text_result["mood_scores"])
0498:     energy = wav_energy(audio_bytes)
0499:     if energy > 0.14:
0500:         scores["energetic"] += 0.15
0501:     if energy < 0.04:
0502:         scores["calm"] += 0.08
0503:     mood = max(scores.items(), key=lambda p: p[1])[0]
0504:     return {"transcript": transcript, "transcription_method": method, "energy": energy, "mood": mood,
0505:             "confidence": float(max(scores.values()) if scores else 0.0),
0506:             "spoken_name_detected": bool(username and transcript and username.lower() in transcript.lower()),
0507:             "text_result": text_result}
0508: 
0509: 
0510: @lru_cache(maxsize=1)
```
<div style="page-break-after: always;"></div>

<!-- Report Page 108 -->
# Appendix A: Source Code Listing Page 16

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0511: def face_cascade():
0512:     if cv2 is None:
0513:         return None
0514:     cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
0515:     return cascade if not cascade.empty() else None
0516: 
0517: 
0518: @lru_cache(maxsize=1)
0519: def smile_cascade():
0520:     if cv2 is None:
0521:         return None
0522:     cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
0523:     return cascade if not cascade.empty() else None
0524: 
0525: 
0526: def decode_image(image_bytes: bytes | None) -> np.ndarray | None:
0527:     if not image_bytes or cv2 is None:
0528:         return None
0529:     return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
0530: 
0531: 
0532: def largest_face_box(image_bgr: np.ndarray):
0533:     cascade = face_cascade()
0534:     if cascade is None or cv2 is None:
0535:         return None
0536:     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
0537:     faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
0538:     return None if len(faces) == 0 else max(faces, key=lambda b: int(b[2] * b[3]))
0539: 
0540: 
0541: def heuristic_face_label(face_bgr: np.ndarray) -> tuple[str, float]:
0542:     if cv2 is None:
0543:         return "neutral", 0.40
0544:     gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
```
<div style="page-break-after: always;"></div>

<!-- Report Page 109 -->
# Appendix A: Source Code Listing Page 17

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0545:     brightness = float(gray.mean())
0546:     texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
0547:     smiles = []
0548:     sc = smile_cascade()
0549:     if sc is not None:
0550:         try:
0551:             smiles = sc.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
0552:         except Exception:
0553:             smiles = []
0554:     if len(smiles) > 0 or brightness > 150:
0555:         return "happy", 0.58
0556:     if texture > 450 and brightness < 130:
0557:         return "angry", 0.47
0558:     if brightness < 95:
0559:         return "sad", 0.46
0560:     return "neutral", 0.44
0561: 
0562: 
0563: def image_to_tensor(image_rgb: np.ndarray) -> Any:
0564:     if not TORCH_AVAILABLE or cv2 is None:
0565:         return None
0566:     resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
0567:     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
0568:     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
0569:     arr = np.transpose((resized - mean) / std, (2, 0, 1))
0570:     return torch.from_numpy(arr).unsqueeze(0).float()
0571: 
0572: 
0573: def build_model(num_classes: int, pretrained: bool = True):
0574:     if not TORCH_AVAILABLE:
0575:         return None
0576:     try:
0577:         weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
0578:         model = efficientnet_v2_s(weights=weights)
```
<div style="page-break-after: always;"></div>

<!-- Report Page 110 -->
# Appendix A: Source Code Listing Page 18

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0579:     except Exception:
0580:         model = efficientnet_v2_s(weights=None)
0581:     in_features = model.classifier[1].in_features
0582:     model.classifier = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
0583:     return model
0584: 
0585: 
0586: def save_checkpoint(model: Any, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
0587:     if not TORCH_AVAILABLE:
0588:         return
0589:     MODEL_DIR.mkdir(parents=True, exist_ok=True)
0590:     payload = {"architecture": "efficientnet_v2_s", "class_names": class_names,
0591:                "model_state_dict": model.state_dict(), "saved_at_utc": utc_now()}
0592:     if extra:
0593:         payload.update(extra)
0594:     torch.save(payload, MODEL_PATH)
0595:     MODEL_META_PATH.write_text(json.dumps({k: v for k, v in payload.items() if k != "model_state_dict"}, indent=2), encoding="utf-8")
0596: 
0597: 
0598: @lru_cache(maxsize=1)
0599: def load_checkpoint():
0600:     if not TORCH_AVAILABLE or not MODEL_PATH.exists():
0601:         return None, {}
0602:     try:
0603:         checkpoint = torch.load(MODEL_PATH, map_location="cpu")
0604:         classes = checkpoint.get("class_names") or DEFAULT_FACE_CLASSES
0605:         model = build_model(len(classes), pretrained=False)
0606:         model.load_state_dict(checkpoint["model_state_dict"], strict=False)
0607:         model.eval()
0608:         return model, {"class_names": classes}
0609:     except Exception:
0610:         return None, {}
0611: 
0612: 
```
<div style="page-break-after: always;"></div>

<!-- Report Page 111 -->
# Appendix A: Source Code Listing Page 19

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0613: def last_conv_module(model: Any):
0614:     if model is None:
0615:         return None
0616:     last = None
0617:     for module in model.modules():
0618:         if TORCH_AVAILABLE and isinstance(module, nn.Conv2d):
0619:             last = module
0620:     return last
0621: 
0622: 
0623: def gradcam_heatmap(model: Any, tensor: Any, target_index: int | None = None) -> np.ndarray | None:
0624:     if not TORCH_AVAILABLE or model is None or tensor is None:
0625:         return None
0626:     layer = last_conv_module(model)
0627:     if layer is None:
0628:         return None
0629:     activations, gradients = [], []
0630:     f_handle = layer.register_forward_hook(lambda _m, _i, out: activations.append(out.detach()))
0631:     b_handle = layer.register_full_backward_hook(lambda _m, _gin, gout: gradients.append(gout[0].detach()))
0632:     try:
0633:         model.zero_grad(set_to_none=True)
0634:         logits = model(tensor)
0635:         if target_index is None:
0636:             target_index = int(torch.argmax(logits, dim=1).item())
0637:         logits[:, target_index].sum().backward()
0638:         acts, grads = activations[-1][0], gradients[-1][0]
0639:         weights = grads.mean(dim=(1, 2))
0640:         cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
0641:         if float(cam.max()) <= 0:
0642:             return None
0643:         return np.uint8(255 * (cam / cam.max()).cpu().numpy())
0644:     except Exception:
0645:         return None
0646:     finally:
```
<div style="page-break-after: always;"></div>

<!-- Report Page 112 -->
# Appendix A: Source Code Listing Page 20

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0647:         f_handle.remove()
0648:         b_handle.remove()
0649: 
0650: 
0651: def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray | None) -> np.ndarray | None:
0652:     if cv2 is None or heatmap is None:
0653:         return None
0654:     heatmap = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
0655:     colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
0656:     return cv2.addWeighted(colored, 0.35, face_bgr, 0.65, 0.0)
0657: 
0658: 
0659: def predict_face(image_bytes: bytes | None) -> dict[str, Any]:
0660:     image = decode_image(image_bytes)
0661:     if image is None or cv2 is None:
0662:         return {"label": "neutral", "mood": "calm", "confidence": 0.0, "method": "no_image",
0663:                 "overlay_rgb": None, "face_found": False}
0664:     box = largest_face_box(image)
0665:     display = image.copy()
0666:     if box is None:
0667:         x, y, w, h = 0, 0, image.shape[1], image.shape[0]
0668:         face = image
0669:     else:
0670:         x, y, w, h = map(int, box)
0671:         face = image[y:y + h, x:x + w]
0672:         cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)
0673:     model, meta = load_checkpoint()
0674:     if model is not None:
0675:         try:
0676:             tensor = image_to_tensor(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
0677:             probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
0678:             classes = meta.get("class_names") or DEFAULT_FACE_CLASSES
0679:             idx = int(np.argmax(probs))
0680:             label = classes[idx] if idx < len(classes) else f"class_{idx}"
```
<div style="page-break-after: always;"></div>

<!-- Report Page 113 -->
# Appendix A: Source Code Listing Page 21

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0681:             overlay = overlay_heatmap(face, gradcam_heatmap(model, tensor, idx))
0682:             if overlay is not None and box is not None:
0683:                 display[y:y + h, x:x + w] = cv2.resize(overlay, (w, h))
0684:             return {"label": label, "mood": normalize_label_to_mood(label), "confidence": float(probs[idx]),
0685:                     "method": "trained_efficientnet_v2_s", "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
0686:                     "face_found": box is not None}
0687:         except Exception:
0688:             pass
0689:     label, confidence = heuristic_face_label(face)
0690:     return {"label": label, "mood": normalize_label_to_mood(label), "confidence": confidence,
0691:             "method": "opencv_heuristic", "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
0692:             "face_found": box is not None}
0693: 
0694: 
0695: class FolderDataset(Dataset):
0696:     def __init__(self, root: Path, class_to_idx: dict[str, int] | None = None, train: bool = False):
0697:         self.root = root
0698:         self.train = train
0699:         classes = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
0700:         self.class_to_idx = class_to_idx or {name: i for i, name in enumerate(classes)}
0701:         self.samples: list[tuple[Path, int]] = []
0702:         for name, idx in self.class_to_idx.items():
0703:             for fp in (root / name).rglob("*") if (root / name).exists() else []:
0704:                 if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS:
0705:                     self.samples.append((fp, idx))
0706: 
0707:     def __len__(self) -> int:
0708:         return len(self.samples)
0709: 
0710:     def __getitem__(self, index: int):
0711:         path, label = self.samples[index]
0712:         image = cv2.imread(str(path)) if cv2 is not None else None
0713:         if image is None:
0714:             image = np.zeros((224, 224, 3), dtype=np.uint8)
```
<div style="page-break-after: always;"></div>

<!-- Report Page 114 -->
# Appendix A: Source Code Listing Page 22

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0715:         rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
0716:         if self.train and random.random() < 0.5:
0717:             rgb = cv2.flip(rgb, 1)
0718:         tensor = image_to_tensor(rgb)[0]
0719:         return tensor, torch.tensor(label, dtype=torch.long)
0720: 
0721: 
0722: def dataset_has_images(split_dir: Path) -> bool:
0723:     return split_dir.exists() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in split_dir.rglob("*"))
0724: 
0725: 
0726: def dataset_summary(root: Path = DATASET_DIR) -> list[dict[str, Any]]:
0727:     rows: list[dict[str, Any]] = []
0728:     for split in ("train", "val", "test"):
0729:         split_dir = root / split
0730:         if split_dir.exists():
0731:             for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
0732:                 rows.append({"split": split, "class_name": class_dir.name,
0733:                              "count": sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)})
0734:     return rows
0735: 
0736: 
0737: def build_loaders(dataset_root: Path, batch_size: int):
0738:     if not TORCH_AVAILABLE or cv2 is None:
0739:         raise RuntimeError("PyTorch and OpenCV are required for training.")
0740:     if not dataset_has_images(dataset_root / "train"):
0741:         raise RuntimeError("No images found in dataset/train.")
0742:     train_ds = FolderDataset(dataset_root / "train", train=True)
0743:     val_ds = FolderDataset(dataset_root / "val", class_to_idx=train_ds.class_to_idx) if dataset_has_images(dataset_root / "val") else None
0744:     test_ds = FolderDataset(dataset_root / "test", class_to_idx=train_ds.class_to_idx) if dataset_has_images(dataset_root / "test") else None
0745:     return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
0746:             DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None,
0747:             DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0) if test_ds else None,
0748:             list(train_ds.class_to_idx.keys()))
```
<div style="page-break-after: always;"></div>

<!-- Report Page 115 -->
# Appendix A: Source Code Listing Page 23

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0749: 
0750: 
0751: def train_model(dataset_root: Path, batch_size: int, epochs: int, lr: float, freeze_backbone: bool, progress_callback=None):
0752:     train_loader, val_loader, test_loader, class_names = build_loaders(dataset_root, batch_size)
0753:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
0754:     model = build_model(len(class_names), pretrained=True).to(device)
0755:     if freeze_backbone:
0756:         for param in model.features.parameters():
0757:             param.requires_grad = False
0758:     optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
0759:     criterion = nn.CrossEntropyLoss()
0760:     history = []
0761:     for epoch in range(1, epochs + 1):
0762:         model.train()
0763:         loss_sum = correct = total = 0
0764:         for images, labels in train_loader:
0765:             images, labels = images.to(device), labels.to(device)
0766:             optimizer.zero_grad(set_to_none=True)
0767:             logits = model(images)
0768:             loss = criterion(logits, labels)
0769:             loss.backward()
0770:             optimizer.step()
0771:             loss_sum += float(loss.item()) * labels.size(0)
0772:             correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
0773:             total += int(labels.size(0))
0774:         row = {"epoch": float(epoch), "train_loss": round(loss_sum / max(total, 1), 4),
0775:                "train_accuracy": round(correct / max(total, 1), 4)}
0776:         history.append(row)
0777:         if progress_callback:
0778:             progress_callback(epoch / epochs, row)
0779:     save_checkpoint(model.cpu(), class_names, {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr})
0780:     load_checkpoint.cache_clear()
0781:     return history, {"message": "Training complete; evaluate using validation or test dataset."}, class_names
0782: 
```
<div style="page-break-after: always;"></div>

<!-- Report Page 116 -->
# Appendix A: Source Code Listing Page 24

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0783: 
0784: def fuse_emotions(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], emoji_mood: str):
0785:     scores = {m: 0.0 for m in MOOD_CHOICES}
0786:     scores[face_result.get("mood", "calm")] += 0.35 * max(float(face_result.get("confidence") or 0.45), 0.30)
0787:     scores[voice_result.get("mood", "calm")] += 0.25 * max(float(voice_result.get("confidence") or 0.45), 0.30)
0788:     scores[text_result.get("mood", "calm")] += 0.25 * max(float(text_result.get("confidence") or 0.45), 0.30)
0789:     scores[emoji_mood] += 0.15
0790:     fused = max(scores.items(), key=lambda p: p[1])[0]
0791:     return fused, scores, float(scores[fused] / (sum(scores.values()) or 1.0))
0792: 
0793: 
0794: def ethical_report(face_result: dict[str, Any], voice_result: dict[str, Any], text_result: dict[str, Any], scores: dict[str, float]):
0795:     warnings = ["This assistant provides supportive guidance, not medical diagnosis."]
0796:     modalities = int(face_result.get("method") != "no_image") + int(bool(voice_result.get("transcript"))) + int(bool(text_result.get("text")))
0797:     if face_result.get("method") == "opencv_heuristic":
0798:         warnings.append("Face analysis is in heuristic fallback mode until a .pth model is trained.")
0799:     if modalities < 2:
0800:         warnings.append("Fusion is stronger when at least two modalities are provided.")
0801:     maximum = max(scores.values()) if scores else 0.0
0802:     band = "High" if maximum >= 0.45 else "Medium" if maximum >= 0.28 else "Low"
0803:     return {"modalities_used": modalities, "confidence_band": band, "warnings": warnings}
0804: 
0805: 
0806: def build_system_prompt(username: str, mood: str, tone_mode: str, summary: str, intent: str, snapshot: str) -> str:
0807:     return textwrap.dedent(f"""
0808:     You are an emotionally intelligent therapeutic chatbot and animated mascot.
0809:     User name: {username}
0810:     Current fused mood: {mood}
0811:     Tone mode: {tone_mode}
0812:     Tone guide: {TONE_GUIDES.get(tone_mode, TONE_GUIDES['Therapist'])}
0813:     Detected intent: {intent}
0814:     Digital twin summary: {summary}
0815:     Emotion snapshot: {snapshot}
0816:     Give one supportive reflection, exactly three reflective questions, and one practical action.
```
<div style="page-break-after: always;"></div>

<!-- Report Page 117 -->
# Appendix A: Source Code Listing Page 25

File: app.py

The following listing is part of the complete single-file implementation supplied with this report.

```python
0817:     Do not diagnose disease or claim certainty.
0818:     """).strip()
0819: 
0820: 
0821: def huggingface_chat(system_prompt: str, user_prompt: str) -> str | None:
0822:     if not HF_TOKEN:
0823:         return None
0824:     payload = {"inputs": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n",
0825:                "parameters": {"max_new_tokens": 220, "temperature": 0.7, "top_p": 0.9, "return_full_text": False},
0826:                "options": {"wait_for_model": True}}
0827:     request = urllib.request.Request(
0828:         f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}",
0829:         data=json.dumps(payload).encode("utf-8"),
0830:         headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
0831:         method="POST",
0832:     )
0833:     try:
0834:         with urllib.request.urlopen(request, timeout=90) as response:
0835:             parsed = json.loads(response.read().decode("utf-8"))
0836:     except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception):
0837:         return None
0838:     if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
0839:         return str(parsed[0].get("generated_text", "")).strip() or None
0840:     if isinstance(parsed, dict):
0841:         return str(parsed.get("generated_text", "")).strip() or None
0842:     return None
0843: 
0844: 
0845: def local_coach(question: str, mood: str, tone_mode: str, summary: str, intent: str) -> str:
0846:     questions = MOOD_QUESTION_BANK.get(mood, MOOD_QUESTION_BANK["calm"])
0847:     opener = {"Therapist": "I hear you, and I want to respond thoughtfully.",
0848:               "Friendly": "I am with you, and we can handle this together.",
0849:               "Motivational": "You are capable, and we can turn this into action right now."}.get(tone_mode, "I am here.")
0850:     return textwrap.dedent(f"""
```
<div style="page-break-after: always;"></div>

<!-- Report Page 118 -->
# Appendix B: Requirements File

requirements.txt

```text
streamlit
torch
torchvision
transformers
opencv-python
numpy
nltk
SpeechRecognition
pyttsx3
```
<div style="page-break-after: always;"></div>

<!-- Report Page 119 -->
# Appendix C: Dataset Folder Structure

dataset/train/happy, dataset/train/sad, dataset/train/angry, dataset/train/neutral

dataset/val/happy, dataset/val/sad, dataset/val/angry, dataset/val/neutral

dataset/test/happy, dataset/test/sad, dataset/test/angry, dataset/test/neutral

<div style="page-break-after: always;"></div>

<!-- Report Page 120 -->
# Appendix D: Screenshot Checklist

Capture these screenshots after running the application: home screen, live analysis, face Grad-CAM,
mascot chat, training tab, twin memory table, setup guide and resource catalog download.
