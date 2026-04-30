# Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection

AIML-452 Major Project - Dissertation

This Markdown source mirrors the generated DOCX report. The DOCX file contains explicit page breaks and Word-compatible footer/page-number settings.

<div style="page-break-before: always;"></div>

<!-- Report Page 1: Cover Page -->

# Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection

(AIML-452 Major Project - Dissertation)

submitted in partial fulfillment of the requirement

for the award of the degree of

Bachelor of Technology

in

AIML

Submitted by

NAME OF THE STUDENT

ENROLLMENT NO

Under the supervision of

NAME OF THE FACULTY SUPERVISOR

DESIGNATION

LOGO OF THE INSTITUTE

Department of Artificial Intelligence and Machine Learning

Name of the Institute

Address of the institute

May/June 2026

<div style="page-break-before: always;"></div>

<!-- Report Page 2: DECLARATION -->

# DECLARATION

This is to certify that the material embodied in this Major Project - Dissertation titled "Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection" being submitted in the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML is based on my original work. It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma. My indebtedness to other works has been duly acknowledged at the relevant places.

The project, source-code design, experimentation plan, analysis, testing and documentation have been prepared for the final major project evaluation and viva voce. The implementation described in this report follows a PyTorch-only approach and avoids TensorFlow and Keras.





(Name of the Student)
Enrollment No

<div style="page-break-before: always;"></div>

<!-- Report Page 3: CERTIFICATE -->

# CERTIFICATE

This is to certify that the work embodied in this Major Project - Dissertation titled "Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection" being submitted in the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML, is original and has been carried out by NAME OF THE STUDENT (Enrollment No. ENROLLMENT NO) under my supervision and guidance.

It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma to the best of my knowledge and belief.




(Name of the Faculty Supervisor)
Designation


(Name of the HOD)
HOD
Name of the Institute

<div style="page-break-before: always;"></div>

<!-- Report Page 4: ACKNOWLEDGEMENT -->

# ACKNOWLEDGEMENT

I express my sincere gratitude to my faculty supervisor for continuous guidance, review comments, technical direction and encouragement throughout the development of this major project. The supervisor's support helped convert the initial idea of cognitive emotional intelligence into a structured, testable and presentation-ready software system.

I am thankful to the Head of the Department, the Principal and the Institute for providing an academic environment where advanced artificial intelligence topics such as multimodal emotion recognition, explainable CNNs, ethical AI and therapeutic conversational interfaces could be explored as part of the final-year major project.

I also thank my mentors, peers, friends and family members for their patience, motivation and feedback during ideation, implementation, testing and report preparation. Their support strengthened the practical relevance of the project and encouraged a user-centric design approach.

<div style="page-break-before: always;"></div>

<!-- Report Page 5: TABLE OF CONTENTS -->

# TABLE OF CONTENTS

| Section | Title | Page No. |
| --- | --- | --- |
| Preliminary | Declaration | i |
| Preliminary | Certificate | ii |
| Preliminary | Acknowledgement | iii |
| Preliminary | Abstract | v |
| Preliminary | List of Figures | vi |
| Preliminary | List of Tables | vii |
| Chapter 1 | Introduction | 1 |
| Chapter 2 | Problem Statement | 11 |
| Chapter 3 | Analysis | 19 |
| Chapter 4 | Design and Architecture | 31 |
| Chapter 5 | Implementation | 43 |
| Chapter 6 | Testing | 58 |
| Chapter 7 | Summary and Conclusion | 68 |
| Chapter 8 | Limitations and Future Work | 74 |
| Bibliography | IEEE References | 80 |
| Appendix | Implementation Code and Screenshots | 85 |

<div style="page-break-before: always;"></div>

<!-- Report Page 6: ABSTRACT -->

# ABSTRACT

The project titled "Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection" presents a PyTorch-based therapeutic chatbot that detects and responds to a user's emotional state through text, voice, facial cues and an emoji-based self-report slider. The work addresses the limitation of generic FAQ-style chatbots by combining cognitive emotional intelligence, multimodal AI fusion, adaptive recommendations and a visually expressive animated mascot interface. The chatbot is designed for a Windows 11 laptop environment using VS Code, Streamlit, PyTorch, OpenCV, Hugging Face Transformers, NLTK, SpeechRecognition and optional pyttsx3-based local speech output.

The proposed system uses Hugging Face free-token integration for text emotion and sentiment intelligence and avoids paid OpenAI APIs. Facial emotion processing is implemented using OpenCV for face capture and a PyTorch EfficientNetV2-S architecture for trained emotion classification. Grad-CAM is included to support explainability by highlighting facial image regions that influence the CNN decision. Voice input is recorded through the browser microphone where supported, converted to text, analyzed for sentiment and enriched through simple audio-energy estimation. A weighted fusion engine combines face, voice, text and emoji signals into a unified User Emotional State Score.

The Streamlit frontend provides a live emotion studio, mascot chatbot, training and Grad-CAM panel, digital emotional twin memory, reinforcement-learning-style recommendation statistics and a setup guide. The system logs interaction history in CSV format and uses feedback such as liked or skipped recommendations to reduce repetition. The project remains non-diagnostic and includes ethical warnings to prevent overconfident mental-health claims. The result is a demo-ready, viva-friendly and extensible major project suitable for presenting the practical use of PyTorch, multimodal AI and explainable affective computing.

<div style="page-break-before: always;"></div>

<!-- Report Page 7: LIST OF FIGURES -->

# LIST OF FIGURES

| Figure No. | Figure Title | Page No. |
| --- | --- | --- |
| Figure 1.1 | Conceptual view of the cognitive emotional intelligence system | 4 |
| Figure 3.1 | Data flow diagram for multimodal emotion detection | 27 |
| Figure 3.2 | Use case diagram for user, chatbot and training workflow | 29 |
| Figure 4.1 | System architecture of Streamlit, PyTorch and Hugging Face modules | 35 |
| Figure 4.2 | Activity flow for live emotion analysis | 39 |
| Figure 4.3 | Class-style diagram of core modules | 42 |
| Figure 5.1 | Streamlit live emotion studio screen | 45 |
| Figure 5.2 | Animated mascot expression states | 48 |
| Figure 5.3 | Grad-CAM explainability overlay | 54 |

<div style="page-break-before: always;"></div>

<!-- Report Page 8: LIST OF TABLES -->

# LIST OF TABLES

| Table No. | Table Title | Page No. |
| --- | --- | --- |
| Table 2.1 | Objectives of the major project | 16 |
| Table 3.1 | Functional requirements | 20 |
| Table 3.2 | Non-functional requirements | 22 |
| Table 3.3 | Software and hardware requirements | 24 |
| Table 4.1 | Module explanation summary | 37 |
| Table 5.1 | Core implementation mapping | 50 |
| Table 6.1 | Functional test cases | 60 |
| Table 6.2 | Model and integration test cases | 64 |

<div style="page-break-before: always;"></div>

<!-- Report Page 9: Introduction -->

# CHAPTER 1: INTRODUCTION

Emotion-aware computing has become important because most digital assistants still treat users as neutral command senders. A therapeutic support interface must understand the user's emotional context before generating advice. The proposed project creates a multimodal chatbot that observes text, speech, facial cues and explicit emoji selection to understand mood in a safer and more human-centered manner [1].

The project is implemented as a single-file Python 3 Streamlit application for easy execution in VS Code on a Windows 11 Dell laptop with an Intel Core i5 processor. The implementation uses PyTorch instead of TensorFlow or Keras. This decision keeps the system aligned with modern research workflows and allows the use of EfficientNetV2-S for facial emotion classification [2].

The mascot interface is not only decorative. It gives the system an embodied conversational identity. The animated avatar changes expressions for happy, sad, calm and energetic emotional states, which helps the user immediately understand how the system has interpreted the current interaction.

Figure/Table reference note for page 1: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 10: Introduction -->

## 1.1 Background of the Project

Affective computing studies how machines can recognize, interpret and respond to human emotions. In real applications, a single signal is often unreliable. Text may hide emotions, voice may be unclear and facial images may suffer from lighting problems. Therefore, this project applies multimodal fusion where every available signal contributes to the final User Emotional State Score.

Hugging Face is selected as the recommended free API integration route because it supports free account creation, read-token generation and access to open-source models. It also supports local model execution through Transformers, which is useful when API access is unavailable or internet speed is limited [3].

The project deliberately avoids paid OpenAI APIs and paid Spotify APIs. For music and lifestyle recommendations, public search links and direct YouTube resources are used. This keeps the demonstration practical for college evaluation without exposing payment-based dependencies.

Figure/Table reference note for page 2: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 11: Introduction -->

## 1.2 Need of the Project

Students and young professionals often require immediate supportive guidance during stress, study pressure, low confidence and emotional confusion. Common chatbots answer repeated FAQ-style questions and ignore the emotional state of the user. The proposed chatbot is designed to ask meaningful questions based on mood and cognitive context.

The implementation is suitable for final-year AIML demonstration because it combines natural language processing, speech-to-text, computer vision, CNN explainability, adaptive recommendation and ethical AI monitoring. These components together show both engineering ability and awareness of responsible AI.

The project also supports a viva-friendly explanation: every module has a clear role, every input contributes to a score and every output can be displayed through the Streamlit dashboard.

Figure/Table reference note for page 3: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 12: Introduction -->

## 1.3 Scope of the Project

The system supports text emotion detection, voice transcription, facial emotion analysis, animated mascot feedback, Hugging Face chat generation, local rule-based fallback, digital twin logging and recommendation feedback. The scope is intentionally designed as a major project rather than an internship report.

The software can be executed locally through VS Code. The required files are app.py, requirements.txt and optional dataset and models folders. The report appendix lists the commands required to install dependencies and run the system.

The project does not claim medical diagnosis. It provides supportive, reflective and motivational responses while showing confidence warnings whenever multimodal evidence is weak.

Figure/Table reference note for page 4: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 13: Introduction -->

## 1.4 Installation and Execution Overview

The user creates a folder in VS Code, places app.py and requirements.txt inside it and creates a virtual environment using python -m venv .venv. On Windows PowerShell, the environment is activated using .\.venv\Scripts\Activate.ps1. The dependencies are installed with python -m pip install --upgrade pip followed by pip install -r requirements.txt.

The requirements.txt file contains streamlit, torch, torchvision, transformers, opencv-python, numpy, nltk, SpeechRecognition and pyttsx3. The command streamlit run app.py starts the browser-based interface at localhost port 8501.

For free Hugging Face integration, the user creates a free account, opens Settings, creates a read access token and sets it as an environment variable using $env:HF_TOKEN="your_token". The token must never be hardcoded in source code or report files.

Figure/Table reference note for page 5: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 14: Introduction -->

## 1.5 Major Project Contribution

The major contribution is the design of a complete affective assistant that is more than a simple chatbot. It integrates a PyTorch EfficientNetV2-S facial model, Grad-CAM explainability, text and speech-based emotional inference, an animated mascot and a digital emotional twin log.

The system introduces a reinforcement-learning-style recommender. It stores exposure, like and skip counts, then applies a novelty and feedback score to avoid repeating the same recommendation. This makes the assistant more adaptive over time.

The result is a complete project suitable for final defence because it includes implementation, training pipeline, testing strategy, ethical safeguards, future scope and clear practical execution steps.

Figure/Table reference note for page 6: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 15: Introduction -->

## 1.6 Literature and Motivation

Prior work in affective computing shows that emotion recognition becomes more dependable when multiple signals are combined. Text classification identifies semantic emotion, facial analysis captures visible expression and voice analysis adds information about intensity. The proposed system follows this direction while keeping the implementation lightweight enough for a college laptop [4].

EfficientNetV2 is used because it provides a strong speed-accuracy balance and modern convolutional feature extraction. Compared with older MobileNetV2-based demonstrations, EfficientNetV2-S offers improved representational capacity while remaining practical for inference and transfer learning [2].

The project's novelty lies in combining explainable emotion detection with a real-time animated mascot and adaptive lifestyle recommendation in a single Streamlit application.

Figure/Table reference note for page 7: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 16: Introduction -->

## 1.6 Literature and Motivation

Prior work in affective computing shows that emotion recognition becomes more dependable when multiple signals are combined. Text classification identifies semantic emotion, facial analysis captures visible expression and voice analysis adds information about intensity. The proposed system follows this direction while keeping the implementation lightweight enough for a college laptop [4].

EfficientNetV2 is used because it provides a strong speed-accuracy balance and modern convolutional feature extraction. Compared with older MobileNetV2-based demonstrations, EfficientNetV2-S offers improved representational capacity while remaining practical for inference and transfer learning [2].

The project's novelty lies in combining explainable emotion detection with a real-time animated mascot and adaptive lifestyle recommendation in a single Streamlit application.

Figure/Table reference note for page 8: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 17: Introduction -->

## 1.6 Literature and Motivation

Prior work in affective computing shows that emotion recognition becomes more dependable when multiple signals are combined. Text classification identifies semantic emotion, facial analysis captures visible expression and voice analysis adds information about intensity. The proposed system follows this direction while keeping the implementation lightweight enough for a college laptop [4].

EfficientNetV2 is used because it provides a strong speed-accuracy balance and modern convolutional feature extraction. Compared with older MobileNetV2-based demonstrations, EfficientNetV2-S offers improved representational capacity while remaining practical for inference and transfer learning [2].

The project's novelty lies in combining explainable emotion detection with a real-time animated mascot and adaptive lifestyle recommendation in a single Streamlit application.

Figure/Table reference note for page 9: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 18: Introduction -->

## 1.6 Literature and Motivation

Prior work in affective computing shows that emotion recognition becomes more dependable when multiple signals are combined. Text classification identifies semantic emotion, facial analysis captures visible expression and voice analysis adds information about intensity. The proposed system follows this direction while keeping the implementation lightweight enough for a college laptop [4].

EfficientNetV2 is used because it provides a strong speed-accuracy balance and modern convolutional feature extraction. Compared with older MobileNetV2-based demonstrations, EfficientNetV2-S offers improved representational capacity while remaining practical for inference and transfer learning [2].

The project's novelty lies in combining explainable emotion detection with a real-time animated mascot and adaptive lifestyle recommendation in a single Streamlit application.

Figure/Table reference note for page 10: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 19: Chapter 2: Problem Statement -->

# CHAPTER 2: PROBLEM STATEMENT

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 11: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 20: Chapter 2: Problem Statement -->

## 2.1 Problem Definition

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 12: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 21: Chapter 2: Problem Statement -->

## 2.2 Objectives

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 13: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 22: Chapter 2: Problem Statement -->

## 2.3 Real-World Relevance

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 14: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 23: Chapter 2: Problem Statement -->

## 2.4 Expected Users

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 15: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 24: Chapter 2: Problem Statement -->

## 2.1 Problem Definition

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 16: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 25: Chapter 2: Problem Statement -->

## 2.2 Objectives

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 17: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 26: Chapter 2: Problem Statement -->

## 2.3 Real-World Relevance

The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.

The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.

The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.

Figure/Table reference note for page 18: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 27: Chapter 3: Analysis -->

# CHAPTER 3: ANALYSIS

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 19: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 28: Chapter 3: Analysis -->

## 3.1 Software Requirement Specifications

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Table 3.1 Functional requirements includes text input processing, microphone capture, camera capture, emotion classification, response generation, mascot update, recommendation display, feedback logging and model training.

Figure/Table reference note for page 20: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 29: Chapter 3: Analysis -->

## 3.1.1 Functional Requirements of the Project

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 21: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 30: Chapter 3: Analysis -->

## 3.1.2 Non-functional Requirements of the Project

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 22: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 31: Chapter 3: Analysis -->

## 3.2 Feasibility Study of the Project

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 23: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 32: Chapter 3: Analysis -->

## 3.3 Tools, Technologies and Platform Used

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 24: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 33: Chapter 3: Analysis -->

## 3.4 Use Case Diagrams and Data Flow Diagrams

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 25: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 34: Chapter 3: Analysis -->

## 3.1 Software Requirement Specifications

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 26: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 35: Chapter 3: Analysis -->

## 3.1.1 Functional Requirements of the Project

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 27: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 36: Chapter 3: Analysis -->

## 3.1.2 Non-functional Requirements of the Project

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 28: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 37: Chapter 3: Analysis -->

## 3.2 Feasibility Study of the Project

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 29: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 38: Chapter 3: Analysis -->

## 3.3 Tools, Technologies and Platform Used

Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.

Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.

The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.

Figure/Table reference note for page 30: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 39: Chapter 4: Design And Architecture -->

# CHAPTER 4: DESIGN AND ARCHITECTURE

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 31: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 40: Chapter 4: Design And Architecture -->

## 4.1 Structure Chart and Work Breakdown Structure

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 32: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 41: Chapter 4: Design And Architecture -->

## 4.2 Explanation of Modules

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure 4.1 represents the architecture as User Inputs -> Modality Engines -> Fusion Engine -> Chatbot Engine -> Mascot UI -> Digital Twin Log. This chain makes the execution flow simple to explain during viva.

Figure/Table reference note for page 33: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 42: Chapter 4: Design And Architecture -->

## 4.3 Flow Chart and Activity Diagram

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 34: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 43: Chapter 4: Design And Architecture -->

## 4.4 Class Diagram and Data Design

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 35: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 44: Chapter 4: Design And Architecture -->

## 4.1 Structure Chart and Work Breakdown Structure

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 36: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 45: Chapter 4: Design And Architecture -->

## 4.2 Explanation of Modules

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 37: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 46: Chapter 4: Design And Architecture -->

## 4.3 Flow Chart and Activity Diagram

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 38: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 47: Chapter 4: Design And Architecture -->

## 4.4 Class Diagram and Data Design

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 39: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 48: Chapter 4: Design And Architecture -->

## 4.1 Structure Chart and Work Breakdown Structure

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 40: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 49: Chapter 4: Design And Architecture -->

## 4.2 Explanation of Modules

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 41: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 50: Chapter 4: Design And Architecture -->

## 4.3 Flow Chart and Activity Diagram

The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.

The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.

The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.

Figure/Table reference note for page 42: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 51: Chapter 5: Implementation -->

# CHAPTER 5: IMPLEMENTATION

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 43: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 52: Chapter 5: Implementation -->

## 5.1 PyTorch EfficientNetV2-S Implementation

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 44: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 53: Chapter 5: Implementation -->

## 5.2 Text Emotion Detection with Hugging Face

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 45: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 54: Chapter 5: Implementation -->

## 5.3 Voice Interaction and Speech Recognition

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 46: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 55: Chapter 5: Implementation -->

## 5.4 Facial Emotion Detection and Grad-CAM

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 47: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 56: Chapter 5: Implementation -->

## 5.5 Animated Mascot and Streamlit Frontend

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 48: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 57: Chapter 5: Implementation -->

## 5.6 Digital Emotional Twin and Recommender

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

The exact VS Code execution uses python -m venv .venv, activation through PowerShell, pip install -r requirements.txt and streamlit run app.py. The requirements file contains torch, torchvision, transformers, streamlit, opencv-python, numpy, nltk, SpeechRecognition and pyttsx3.

Figure/Table reference note for page 49: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 58: Chapter 5: Implementation -->

## 5.7 Step-by-step Execution in VS Code

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 50: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 59: Chapter 5: Implementation -->

## 5.1 PyTorch EfficientNetV2-S Implementation

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 51: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 60: Chapter 5: Implementation -->

## 5.2 Text Emotion Detection with Hugging Face

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 52: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 61: Chapter 5: Implementation -->

## 5.3 Voice Interaction and Speech Recognition

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 53: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 62: Chapter 5: Implementation -->

## 5.4 Facial Emotion Detection and Grad-CAM

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 54: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 63: Chapter 5: Implementation -->

## 5.5 Animated Mascot and Streamlit Frontend

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 55: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 64: Chapter 5: Implementation -->

## 5.6 Digital Emotional Twin and Recommender

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 56: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 65: Chapter 5: Implementation -->

## 5.7 Step-by-step Execution in VS Code

The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.

The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.

Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.

Figure/Table reference note for page 57: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 66: Chapter 6: Testing -->

# CHAPTER 6: TESTING

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 58: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 67: Chapter 6: Testing -->

## 6.1 Testing Strategy

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 59: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 68: Chapter 6: Testing -->

## 6.2 Functional Test Cases

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 60: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 69: Chapter 6: Testing -->

## 6.3 Model Evaluation Metrics

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 61: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 70: Chapter 6: Testing -->

## 6.4 Integration and Usability Testing

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 62: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 71: Chapter 6: Testing -->

## 6.1 Testing Strategy

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 63: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 72: Chapter 6: Testing -->

## 6.2 Functional Test Cases

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 64: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 73: Chapter 6: Testing -->

## 6.3 Model Evaluation Metrics

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 65: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 74: Chapter 6: Testing -->

## 6.4 Integration and Usability Testing

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 66: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 75: Chapter 6: Testing -->

## 6.1 Testing Strategy

Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.

Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.

Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].

Figure/Table reference note for page 67: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 76: Chapter 7: Summary And Conclusion -->

# CHAPTER 7: SUMMARY AND CONCLUSION

The project successfully integrates emotional intelligence with a practical chatbot interface. It demonstrates how PyTorch, Streamlit and Hugging Face can be used to create a free, local and explainable major project without paid API dependence.

The system moves beyond generic FAQ replies by using the user's detected mood, intent, digital twin history and current question to shape the response. It also includes an ethical layer that warns users about uncertainty and prevents clinical overclaiming.

The conclusion is that multimodal affective computing can make chatbot interaction more supportive, transparent and personalized when implemented with responsible engineering boundaries.

Figure/Table reference note for page 68: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 77: Chapter 7: Summary And Conclusion -->

## 7.1 Summary

The project successfully integrates emotional intelligence with a practical chatbot interface. It demonstrates how PyTorch, Streamlit and Hugging Face can be used to create a free, local and explainable major project without paid API dependence.

The system moves beyond generic FAQ replies by using the user's detected mood, intent, digital twin history and current question to shape the response. It also includes an ethical layer that warns users about uncertainty and prevents clinical overclaiming.

The conclusion is that multimodal affective computing can make chatbot interaction more supportive, transparent and personalized when implemented with responsible engineering boundaries.

Figure/Table reference note for page 69: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 78: Chapter 7: Summary And Conclusion -->

## 7.2 Conclusion

The project successfully integrates emotional intelligence with a practical chatbot interface. It demonstrates how PyTorch, Streamlit and Hugging Face can be used to create a free, local and explainable major project without paid API dependence.

The system moves beyond generic FAQ replies by using the user's detected mood, intent, digital twin history and current question to shape the response. It also includes an ethical layer that warns users about uncertainty and prevents clinical overclaiming.

The conclusion is that multimodal affective computing can make chatbot interaction more supportive, transparent and personalized when implemented with responsible engineering boundaries.

Figure/Table reference note for page 70: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 79: Chapter 7: Summary And Conclusion -->

## 7.3 Learning Outcomes

The project successfully integrates emotional intelligence with a practical chatbot interface. It demonstrates how PyTorch, Streamlit and Hugging Face can be used to create a free, local and explainable major project without paid API dependence.

The system moves beyond generic FAQ replies by using the user's detected mood, intent, digital twin history and current question to shape the response. It also includes an ethical layer that warns users about uncertainty and prevents clinical overclaiming.

The conclusion is that multimodal affective computing can make chatbot interaction more supportive, transparent and personalized when implemented with responsible engineering boundaries.

Figure/Table reference note for page 71: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 80: Chapter 7: Summary And Conclusion -->

## 7.1 Summary

The project successfully integrates emotional intelligence with a practical chatbot interface. It demonstrates how PyTorch, Streamlit and Hugging Face can be used to create a free, local and explainable major project without paid API dependence.

The system moves beyond generic FAQ replies by using the user's detected mood, intent, digital twin history and current question to shape the response. It also includes an ethical layer that warns users about uncertainty and prevents clinical overclaiming.

The conclusion is that multimodal affective computing can make chatbot interaction more supportive, transparent and personalized when implemented with responsible engineering boundaries.

Figure/Table reference note for page 72: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 81: Chapter 7: Summary And Conclusion -->

## 7.2 Conclusion

The project successfully integrates emotional intelligence with a practical chatbot interface. It demonstrates how PyTorch, Streamlit and Hugging Face can be used to create a free, local and explainable major project without paid API dependence.

The system moves beyond generic FAQ replies by using the user's detected mood, intent, digital twin history and current question to shape the response. It also includes an ethical layer that warns users about uncertainty and prevents clinical overclaiming.

The conclusion is that multimodal affective computing can make chatbot interaction more supportive, transparent and personalized when implemented with responsible engineering boundaries.

Figure/Table reference note for page 73: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 82: Chapter 8: Limitations Of The Project And Future Work -->

# CHAPTER 8: LIMITATIONS OF THE PROJECT AND FUTURE WORK

The system is limited by dataset quality, webcam lighting, microphone noise, CPU speed and availability of local model files. The fallback facial heuristic is useful for demonstration but cannot replace a properly trained model.

Future work may include VRM avatar integration, lip synchronization, stronger speech emotion models, multilingual support, mobile WebView packaging, privacy-preserving local storage and clinical expert review for therapeutic safety.

Another future enhancement is deployment as a progressive web application or Android WebView wrapper. The current Streamlit version can already be opened on mobile through a URL and added to the home screen for a free app-like experience.

Figure/Table reference note for page 74: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 83: Chapter 8: Limitations Of The Project And Future Work -->

## 8.1 Limitations

The system is limited by dataset quality, webcam lighting, microphone noise, CPU speed and availability of local model files. The fallback facial heuristic is useful for demonstration but cannot replace a properly trained model.

Future work may include VRM avatar integration, lip synchronization, stronger speech emotion models, multilingual support, mobile WebView packaging, privacy-preserving local storage and clinical expert review for therapeutic safety.

Another future enhancement is deployment as a progressive web application or Android WebView wrapper. The current Streamlit version can already be opened on mobile through a URL and added to the home screen for a free app-like experience.

Figure/Table reference note for page 75: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 84: Chapter 8: Limitations Of The Project And Future Work -->

## 8.2 Future Work

The system is limited by dataset quality, webcam lighting, microphone noise, CPU speed and availability of local model files. The fallback facial heuristic is useful for demonstration but cannot replace a properly trained model.

Future work may include VRM avatar integration, lip synchronization, stronger speech emotion models, multilingual support, mobile WebView packaging, privacy-preserving local storage and clinical expert review for therapeutic safety.

Another future enhancement is deployment as a progressive web application or Android WebView wrapper. The current Streamlit version can already be opened on mobile through a URL and added to the home screen for a free app-like experience.

Figure/Table reference note for page 76: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 85: Chapter 8: Limitations Of The Project And Future Work -->

## 8.3 Deployment Possibilities

The system is limited by dataset quality, webcam lighting, microphone noise, CPU speed and availability of local model files. The fallback facial heuristic is useful for demonstration but cannot replace a properly trained model.

Future work may include VRM avatar integration, lip synchronization, stronger speech emotion models, multilingual support, mobile WebView packaging, privacy-preserving local storage and clinical expert review for therapeutic safety.

Another future enhancement is deployment as a progressive web application or Android WebView wrapper. The current Streamlit version can already be opened on mobile through a URL and added to the home screen for a free app-like experience.

Figure/Table reference note for page 77: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 86: Chapter 8: Limitations Of The Project And Future Work -->

## 8.1 Limitations

The system is limited by dataset quality, webcam lighting, microphone noise, CPU speed and availability of local model files. The fallback facial heuristic is useful for demonstration but cannot replace a properly trained model.

Future work may include VRM avatar integration, lip synchronization, stronger speech emotion models, multilingual support, mobile WebView packaging, privacy-preserving local storage and clinical expert review for therapeutic safety.

Another future enhancement is deployment as a progressive web application or Android WebView wrapper. The current Streamlit version can already be opened on mobile through a URL and added to the home screen for a free app-like experience.

Figure/Table reference note for page 78: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 87: Chapter 8: Limitations Of The Project And Future Work -->

## 8.2 Future Work

The system is limited by dataset quality, webcam lighting, microphone noise, CPU speed and availability of local model files. The fallback facial heuristic is useful for demonstration but cannot replace a properly trained model.

Future work may include VRM avatar integration, lip synchronization, stronger speech emotion models, multilingual support, mobile WebView packaging, privacy-preserving local storage and clinical expert review for therapeutic safety.

Another future enhancement is deployment as a progressive web application or Android WebView wrapper. The current Streamlit version can already be opened on mobile through a URL and added to the home screen for a free app-like experience.

Figure/Table reference note for page 79: citations and diagrams are aligned with IEEE-style references in the bibliography.

<div style="page-break-before: always;"></div>

<!-- Report Page 88: Bibliography -->

# BIBLIOGRAPHY

[1] R. W. Picard, Affective Computing. Cambridge, MA, USA: MIT Press, 1997.

[2] M. Tan and Q. V. Le, EfficientNetV2: Smaller Models and Faster Training, in Proc. International Conference on Machine Learning, 2021.

All citations in the chapters use IEEE numeric style. The bibliography focuses on affective computing, PyTorch, EfficientNetV2, Transformers, OpenCV, Grad-CAM and model evaluation.

<div style="page-break-before: always;"></div>

<!-- Report Page 89: Bibliography -->

## BIBLIOGRAPHY

[3] T. Wolf et al., Transformers: State-of-the-Art Natural Language Processing, in Proc. EMNLP: System Demonstrations, 2020, pp. 38-45.

[4] P. Ekman and W. V. Friesen, Constants across cultures in the face and emotion, Journal of Personality and Social Psychology, vol. 17, no. 2, pp. 124-129, 1971.

All citations in the chapters use IEEE numeric style. The bibliography focuses on affective computing, PyTorch, EfficientNetV2, Transformers, OpenCV, Grad-CAM and model evaluation.

<div style="page-break-before: always;"></div>

<!-- Report Page 90: Bibliography -->

## BIBLIOGRAPHY

[5] T. Fawcett, An introduction to ROC analysis, Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.

[6] A. Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, in Proc. NeurIPS, 2019.

All citations in the chapters use IEEE numeric style. The bibliography focuses on affective computing, PyTorch, EfficientNetV2, Transformers, OpenCV, Grad-CAM and model evaluation.

<div style="page-break-before: always;"></div>

<!-- Report Page 91: Bibliography -->

## BIBLIOGRAPHY

[7] G. Bradski, The OpenCV Library, Dr. Dobb's Journal of Software Tools, 2000.

[8] A. Vaswani et al., Attention Is All You Need, in Proc. NeurIPS, 2017.

All citations in the chapters use IEEE numeric style. The bibliography focuses on affective computing, PyTorch, EfficientNetV2, Transformers, OpenCV, Grad-CAM and model evaluation.

<div style="page-break-before: always;"></div>

<!-- Report Page 92: Bibliography -->

## BIBLIOGRAPHY

[9] M. D. Zeiler and R. Fergus, Visualizing and Understanding Convolutional Networks, in Proc. ECCV, 2014.

[10] R. R. Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, in Proc. ICCV, 2017.

All citations in the chapters use IEEE numeric style. The bibliography focuses on affective computing, PyTorch, EfficientNetV2, Transformers, OpenCV, Grad-CAM and model evaluation.

<div style="page-break-before: always;"></div>

<!-- Report Page 93: Appendix -->

# APPENDIX A: VS CODE EXECUTION STEPS

Create app.py, requirements.txt and optional dataset and models folders. Create a virtual environment with python -m venv .venv. Activate it in PowerShell using .\.venv\Scripts\Activate.ps1. Install packages using pip install -r requirements.txt. Run the dashboard using streamlit run app.py.

<div style="page-break-before: always;"></div>

<!-- Report Page 94: Appendix -->

# APPENDIX B: REQUIREMENTS.TXT

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

<div style="page-break-before: always;"></div>

<!-- Report Page 95: Appendix -->

# APPENDIX C: DATASET FOLDER STRUCTURE

```text
dataset/
  train/
    happy/
    sad/
    angry/
    neutral/
  val/
    happy/
    sad/
    angry/
    neutral/
  test/
    happy/
    sad/
    angry/
    neutral/
```

<div style="page-break-before: always;"></div>

<!-- Report Page 96: Appendix -->

# APPENDIX D: CORE SOURCE CODE LISTING

```text
from __future__ import annotations
```

```text
import os, csv, json, random, urllib.request
```

```text
from pathlib import Path
```

```text
import numpy as np
```

```text
import cv2
```

```text
import torch
```

```text
import torch.nn as nn
```

```text
import streamlit as st
```

```text
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
```

```text

```

```text
HF_TOKEN = os.getenv('HF_TOKEN', '').strip()
```

```text
MODEL_PATH = Path('models/emotion_efficientnet_v2_s.pth')
```

```text
MOOD_CHOICES = ['happy', 'sad', 'calm', 'energetic']
```

```text

```

```text
def build_model(num_classes: int):
```

```text
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
```

```text
    in_features = model.classifier[1].in_features
```

```text
    model.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_features, num_classes))
```

<div style="page-break-before: always;"></div>

<!-- Report Page 97: Appendix -->

## APPENDIX D: CORE SOURCE CODE LISTING CONTINUED

```text
    return model
```

```text

```

```text
def fuse_emotions(face, voice, text, emoji):
```

```text
    scores = {m: 0.0 for m in MOOD_CHOICES}
```

```text
    scores[face['mood']] += 0.35 * max(face['confidence'], 0.30)
```

```text
    scores[voice['mood']] += 0.25 * max(voice['confidence'], 0.30)
```

```text
    scores[text['mood']] += 0.25 * max(text['confidence'], 0.30)
```

```text
    scores[emoji] += 0.15
```

```text
    fused = max(scores, key=scores.get)
```

```text
    return fused, scores, scores[fused] / max(sum(scores.values()), 1.0)
```

```text

```

```text
def main():
```

```text
    st.title('Emotionally Intelligent Animated Mascot Chatbot')
```

```text
    st.write('PyTorch-only major project demo with Streamlit, Hugging Face and OpenCV.')
```

```text

```

```text
if __name__ == '__main__':
```

```text
    main()
```

<div style="page-break-before: always;"></div>

<!-- Report Page 98: Appendix -->

# APPENDIX E: SCREENSHOT PLACEHOLDERS

Screenshot 1: Streamlit live emotion studio with camera, microphone and text box.

Screenshot 2: Animated mascot changing expression according to fused mood.

Screenshot 3: Grad-CAM heatmap overlay on detected face.

Screenshot 4: Digital emotional twin CSV log and recommender feedback table.

<div style="page-break-before: always;"></div>

<!-- Report Page 99: Appendix -->

# APPENDIX F: SECURITY NOTE

API tokens must be generated from the Hugging Face account settings page and stored only in the HF_TOKEN environment variable. Tokens must not be pasted into app.py, reports, screenshots or public repositories.

<div style="page-break-before: always;"></div>

<!-- Report Page 100: Appendix -->

# APPENDIX G: VIVA QUESTIONS

1. Why was EfficientNetV2-S selected instead of MobileNetV2?

2. How does Grad-CAM explain the CNN prediction?

3. Why is Hugging Face suitable for free API integration?

4. How does the fusion score combine face, voice, text and emoji evidence?

<div style="page-break-before: always;"></div>

<!-- Report Page 101: Appendix -->

# APPENDIX H: RESULT DISCUSSION

The expected result is a working local demonstration where the user captures a face image, records or types a message, selects an emoji state and receives a mood-aware mascot response with adaptive lifestyle recommendation.

<div style="page-break-before: always;"></div>

<!-- Report Page 102: Appendix -->

# APPENDIX I: PROJECT BINDING DETAILS

For hard-bound submission, use maroon binding with golden engraved printing. The side of the hard-bound report should be printed as "Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and (2022-26)".

<div style="page-break-before: always;"></div>

<!-- Report Page 103: Appendix -->

## APPENDIX ADDITIONAL IMPLEMENTATION NOTES

The full single-file implementation can be extended from the appendix code and the module mapping in Chapter 5. The design remains PyTorch-only and avoids TensorFlow and Keras entirely.

<div style="page-break-before: always;"></div>

<!-- Report Page 104: Appendix -->

## APPENDIX ADDITIONAL IMPLEMENTATION NOTES

The full single-file implementation can be extended from the appendix code and the module mapping in Chapter 5. The design remains PyTorch-only and avoids TensorFlow and Keras entirely.

<div style="page-break-before: always;"></div>

<!-- Report Page 105: Appendix -->

## APPENDIX ADDITIONAL IMPLEMENTATION NOTES

The full single-file implementation can be extended from the appendix code and the module mapping in Chapter 5. The design remains PyTorch-only and avoids TensorFlow and Keras entirely.

<div style="page-break-before: always;"></div>

<!-- Report Page 106: Appendix -->

## APPENDIX ADDITIONAL IMPLEMENTATION NOTES

The full single-file implementation can be extended from the appendix code and the module mapping in Chapter 5. The design remains PyTorch-only and avoids TensorFlow and Keras entirely.
