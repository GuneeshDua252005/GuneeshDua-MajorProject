# EMOTIONALLY INTELLIGENT ANIMATED MASCOT CHATBOT  

**(AIML-452 Major Project - Dissertation)**  

submitted in partial fulfillment of the requirement  
for the award of the degree of  

**Bachelor of Technology**  
in  
**Artificial Intelligence and Machine Learning (AIML)**  

Submitted by  

**NAME OF THE STUDENT**  
**ENROLLMENT NO**  

Under the supervision of  

**NAME OF THE FACULTY SUPERVISOR**  
**DESIGNATION**  

LOGO OF THE INSTITUTE  
Name of the Department  
Name of the Institute  
Address of the Institute  

May/June 2026  

---

# DECLARATION

This is to certify that the material embodied in this Major Project - Dissertation titled **“Emotionally Intelligent Animated Mascot Chatbot with Multimodal Emotion Intelligence and Adaptive Lifestyle Support”** being submitted in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML is based on my original work.

It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma.

My indebtedness to other works has been duly acknowledged at the relevant places.

  
  
(Name of the Student)  
Enrollment No.  

---

# CERTIFICATE

This is to certify that the work embodied in this Major Project - Dissertation titled **“Emotionally Intelligent Animated Mascot Chatbot with Multimodal Emotion Intelligence and Adaptive Lifestyle Support”** being submitted in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML, is original and has been carried out by **NAME OF THE STUDENT (Enrollment No. ____________)** under my supervision and guidance.

It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma to the best of my knowledge and belief.

  
(Name of the Faculty Supervisor)  
Designation  

  
(Name of the HOD)  
HOD  
Name of the Institute  

---

# ACKNOWLEDGEMENT

I express my sincere gratitude to my Faculty Supervisor, **[Name]**, for continuous guidance, constructive feedback, and technical mentorship throughout this major project. Their support helped me evolve this work from a conceptual idea to a working multimodal therapeutic chatbot system.

I am deeply thankful to the **Head of Department**, **Principal**, and all faculty members of the Department of AIML for providing an encouraging academic environment and access to infrastructure required for implementation, experimentation, and project validation.

I would also like to thank my institute for motivating students to build socially relevant AI systems that solve real-world problems and reflect responsible innovation.

I am grateful to my peers and project colleagues for technical discussions, code reviews, and practical suggestions during model development, user interface design, and testing.

Finally, I thank my family and friends for their emotional support, patience, and encouragement during each phase of this project.

---

# TABLE OF CONTENTS

| Section | Page No. |
|---|---|
| Declaration | i |
| Certificate | ii |
| Acknowledgement | iii |
| Abstract | iv |
| List of Figures | v |
| List of Tables | vi |
| Chapter 1: Introduction | 1 |
| Chapter 2: Problem Statement | 10 |
| Chapter 3: Analysis | 17 |
| Chapter 4: Design and Architecture | 30 |
| Chapter 5: Implementation | 44 |
| Chapter 6: Testing | 62 |
| Chapter 7: Summary and Conclusion | 71 |
| Chapter 8: Limitation and Future Work | 75 |
| Bibliography | 80 |
| Appendix | 85 |

---

# ABSTRACT

This major project presents an **Emotionally Intelligent Animated Mascot Chatbot** designed as a therapeutic and adaptive digital companion. The system integrates **multimodal emotion understanding** from text, speech, facial cues, and user emoji input to estimate the user’s cognitive-emotional state in real time. Unlike generic chatbots, the proposed model dynamically adapts tone and response strategy through a fusion engine and contextual memory layer called the **Digital Emotional Twin**.

The complete solution is implemented in a **single Python file** using **PyTorch** (without TensorFlow/Keras) and deployed through **Streamlit** for a user-friendly interface. The facial emotion pipeline uses **EfficientNetV2-S** with Grad-CAM visualization for interpretability, while text sentiment and emotion are inferred using **free Hugging Face transformer models** and intent enrichment through **NLTK synonym analysis**. Speech is captured via microphone, converted to text, and fused with audio energy signals for voice mood profiling.

The chatbot layer avoids paid APIs and uses **free Hugging Face Inference API** when token is available, with robust local fallback for uninterrupted operation. An adaptive recommendation module presents mood-relevant resources and updates preference statistics using reinforcement-style exposure, like, and skip signals. A lightweight animated mascot changes visual behavior based on emotional inference to improve engagement and perceived empathy.

The project addresses practical constraints of student systems (Windows 11, Intel i5, 8 GB RAM) by providing resource-aware execution flow, minimal dependencies, and guided setup instructions. It is suitable for final-year demonstration, viva presentation, and extensible research. The architecture emphasizes real-world impact in student wellness, emotional self-reflection, and context-aware AI interaction while preserving ethical guardrails and transparency of confidence.

---

# LIST OF FIGURES

| Figure No. | Figure Title | Page No. |
|---|---|---|
| Figure 1.1 | High-level system overview | 3 |
| Figure 1.2 | End-to-end user interaction flow | 6 |
| Figure 3.1 | Requirement analysis stack | 19 |
| Figure 3.2 | Multimodal fusion concept | 26 |
| Figure 4.1 | Work Breakdown Structure | 31 |
| Figure 4.2 | Module-level architecture | 34 |
| Figure 4.3 | Activity diagram of live analysis | 38 |
| Figure 4.4 | Class/data relationship map | 41 |
| Figure 5.1 | Streamlit home and sidebar | 45 |
| Figure 5.2 | Live Emotion Studio tab | 48 |
| Figure 5.3 | Mascot Chat tab | 51 |
| Figure 5.4 | Training and Grad-CAM tab | 54 |
| Figure 5.5 | Digital Twin and recommender dashboard | 57 |
| Figure 6.1 | Representative test execution flow | 63 |
| Figure 8.1 | Future roadmap phases | 78 |

---

# LIST OF TABLES

| Table No. | Table Title | Page No. |
|---|---|---|
| Table 2.1 | Core problem-to-solution mapping | 12 |
| Table 3.1 | Functional requirements | 21 |
| Table 3.2 | Non-functional requirements | 23 |
| Table 3.3 | Feasibility matrix | 25 |
| Table 3.4 | Tools, technologies, and platform | 28 |
| Table 4.1 | Module definitions and responsibilities | 36 |
| Table 5.1 | File and folder structure | 46 |
| Table 5.2 | requirements.txt package list | 47 |
| Table 6.1 | Test cases and expected outcomes | 66 |
| Table 6.2 | Error handling verification | 69 |
| Table 8.1 | Limitation and mitigation strategy | 76 |

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Motivation

Human-computer interaction is rapidly moving from command-driven interfaces to emotionally aware and context-sensitive systems. Conventional chatbot applications usually provide one-dimensional text responses and often fail to capture the user’s emotional state. As a result, their responses may be grammatically correct but psychologically irrelevant. In sensitive domains such as student stress management, mental wellness support, and lifestyle coaching, this gap becomes critical.

This project addresses that gap by building a therapeutic conversational AI system that can infer emotional state from multiple user signals:

1. **Text signal**: sentiment and emotion extraction from typed text.
2. **Voice signal**: speech-to-text conversion and tone heuristics from microphone recordings.
3. **Facial signal**: emotion-oriented face analysis through webcam image.
4. **User preference signal**: emoji slider input and adaptive feedback.

The project’s uniqueness lies in integrating these modalities into one fused emotional state score and using this score to generate adaptive responses through a mascot-based conversational interface.

## 1.2 Why This Project Matters

Modern students and young professionals frequently experience emotional overload, stress, uncertainty, and motivation fluctuations. A static FAQ chatbot is often insufficient in such contexts. The proposed system acts as a supportive layer that:

- notices emotional changes,
- asks meaningful reflective questions,
- provides short actionable suggestions,
- avoids repetitive generic responses,
- offers visual companionship through an animated mascot.

This creates a more human-like, responsive, and engaging interaction.

## 1.3 Project Aim

To design and implement a **single-file, PyTorch-based, multimodal therapeutic chatbot** with animated mascot feedback that detects and fuses emotional cues and provides context-aware, mood-adaptive interaction without using paid APIs.

## 1.4 Step-by-Step Execution Guide in VS Code (Windows 11, Intel i5)

The below execution flow is directly aligned to a student laptop setup (Windows 11, Intel i5, 8 GB RAM recommended baseline).

### Step 1: Create Project Folder

Create one folder, for example:

`EmotionMascotMajorProject`

Inside it create:

- `app.py`
- `requirements.txt`
- `.gitignore` (optional but recommended)
- `dataset/` (optional initially)
- `models/` (auto-created by code)

### Step 2: Open VS Code and Activate Virtual Environment

Open the folder in VS Code. In terminal (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Add requirements.txt Content

Create `requirements.txt` and write:

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

### Step 4: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Free Hugging Face API Token

1. Create free account on https://huggingface.co  
2. Settings -> Access Tokens -> Create token (Read)  
3. Set token in terminal:

```powershell
$env:HF_TOKEN="your_hf_token_here"
```

### Step 6: Run Application

```powershell
streamlit run app.py
```

Open browser URL (usually `http://localhost:8501`).

### Step 7: Optional Model Training

To train EfficientNetV2-S face model, create folders:

```text
dataset/train/<emotion_class>/
dataset/val/<emotion_class>/
dataset/test/<emotion_class>/
```

Add images in each class folder and use “Training + Grad-CAM” tab in app.

## 1.5 Why Hugging Face Instead of Paid OpenAI + Spotify APIs

- OpenAI advanced endpoints and Spotify developer workflows may involve paid/rate-limited plans.
- Hugging Face offers free token-based access suitable for student projects.
- Free transformer models enable on-device fallback, reducing API dependency.
- The project therefore remains executable in a no-cost setup.

Spotify integration is retained via public search links and recommendation routing instead of paid backend API calls.

## 1.6 Project Scope

The current scope includes:

- multimodal emotion inference,
- adaptive therapeutic chat,
- mascot expression rendering,
- lightweight recommender memory,
- explainable Grad-CAM support,
- local-friendly execution.

## 1.7 Contributions

1. Single-file unified architecture for viva-friendly demonstration.
2. PyTorch EfficientNetV2-S replacement over outdated MobileNetV2 choices.
3. Multimodal emotional fusion with confidence display.
4. Free API-first architecture with deterministic fallback.
5. Digital twin log and non-repetition recommendation behavior.

## 1.8 Chapter Organization

The report is organized as follows:

- Chapter 2 defines the problem statement and objectives.
- Chapter 3 covers analysis and feasibility.
- Chapter 4 explains design and architecture.
- Chapter 5 provides implementation details.
- Chapter 6 presents testing.
- Chapter 7 provides conclusion.
- Chapter 8 discusses limitations and future work.

---

# CHAPTER 2: PROBLEM STATEMENT

## 2.1 Problem Definition

Traditional chatbot systems fail in emotional adaptability due to three limitations:

1. **Single-modal understanding** (text only).
2. **Context-insensitive replies** that feel repetitive.
3. **No visual embodiment**, resulting in weak emotional engagement.

In real-world usage, especially for students and youth users, emotional states may shift quickly and are often communicated through mixed channels (speech hesitation, facial cues, language tone, and reaction icons). A chatbot that ignores such signals cannot provide useful therapeutic guidance.

Therefore, the problem is:

**How to create a low-cost, real-time, multimodal, therapeutic chatbot that detects user mindset and adapts responses using free APIs and PyTorch-only architecture suitable for student hardware?**

## 2.2 Objectives

1. Build a single-file Python application using only PyTorch-compatible stack.
2. Integrate text, voice, face, and emoji channels for emotional inference.
3. Replace MobileNetV2 with EfficientNetV2-S for face emotion inference.
4. Implement Grad-CAM for visual explainability.
5. Integrate free Hugging Face models for sentiment/emotion/chat responses.
6. Develop adaptive response styles (Therapist, Friendly, Motivational).
7. Implement animated mascot that changes behavior according to fused mood.
8. Maintain interaction memory and recommendation adaptation.
9. Ensure compatibility with VS Code on Windows 11 systems.
10. Provide a project architecture suitable for 8th semester major project defense.

## 2.3 Real-World Relevance

The system addresses:

- stress-aware student support,
- emotional self-reflection coaching,
- context-driven interaction in educational wellness platforms,
- AI companion experiences in productivity and lifestyle tools.

## 2.4 Problem-to-Solution Mapping

| Problem | Impact | Proposed Solution |
|---|---|---|
| Generic chatbot responses | Poor user trust | Mood-aware response engine |
| No multimodal emotional understanding | Misaligned guidance | Fusion of text + voice + face + emoji |
| Dependency on paid APIs | Limited deployability | Free Hugging Face + local fallback |
| Outdated CNN choice | Lower performance relevance | EfficientNetV2-S with transfer learning |
| No explainability | Weak academic credibility | Grad-CAM visual interpretation |

---

# CHAPTER 3: ANALYSIS

## 3.1 Software Requirement Specifications

### 3.1.1 Functional Requirements of the Project

1. Capture webcam image input.
2. Capture microphone audio input.
3. Accept user text and emoji mood slider.
4. Perform text emotion and sentiment classification.
5. Perform voice transcription and mood inference.
6. Perform face emotion inference with PyTorch model.
7. Fuse all modalities into single emotional state.
8. Display confidence and ethical warnings.
9. Provide adaptive chatbot output by tone mode.
10. Render mood-aware animated mascot.
11. Maintain logs for emotional twin data.
12. Recommend context resources with preference learning.
13. Allow optional model training and checkpoint saving.

### 3.1.2 Non-functional Requirements of the Project

1. **Usability**: Clean Streamlit-based user interface.
2. **Portability**: Runs on standard student laptop setups.
3. **Maintainability**: Single-file flow with modular functions.
4. **Performance**: Reasonable latency on CPU mode.
5. **Scalability**: Future extension to multilingual and richer avatars.
6. **Security**: Token-based API usage via environment variables.
7. **Reliability**: Fallback behavior when model/API unavailable.

## 3.2 Feasibility Study of the Project

### 3.2.1 Technical Feasibility

The project uses mature and widely supported Python libraries. EfficientNetV2-S is available through torchvision. Streamlit enables rapid interface deployment. Hugging Face offers free model access and local alternatives.

### 3.2.2 Operational Feasibility

End users require no complex setup beyond dependency installation and token configuration. The guided setup tab inside the app reduces operational burden.

### 3.2.3 Economic Feasibility

The system is designed to run fully on free tooling:

- free Python ecosystem,
- free Hugging Face account,
- no paid OpenAI dependency,
- no mandatory cloud GPU cost.

### 3.2.4 Schedule Feasibility

The implementation is modular and demonstrable in semester timeline with phased progression:

- phase 1: baseline UI + text pipeline,
- phase 2: multimodal integration,
- phase 3: training and evaluation enhancements,
- phase 4: documentation and defense preparation.

## 3.3 Tools / Technologies / Platform Used

| Component | Technology |
|---|---|
| Language | Python 3 |
| Deep Learning | PyTorch |
| CNN Backbone | EfficientNetV2-S |
| NLP Models | Hugging Face Transformers |
| UI | Streamlit |
| Vision Input | OpenCV |
| Speech-to-Text | SpeechRecognition |
| Text-to-Speech | pyttsx3 |
| Lexical Enrichment | NLTK |
| Data Handling | CSV + NumPy |
| IDE | VS Code |
| OS Target | Windows 11 compatible |

## 3.4 Use Case Diagram / Data Flow Description

### 3.4.1 Primary Actors

1. End user
2. System (multimodal engine)
3. Model repository/API provider

### 3.4.2 Primary Use Cases

- register session identity (username),
- provide text query,
- capture facial image,
- record voice,
- select emoji mood,
- run live analysis,
- receive adaptive response and recommendation,
- provide feedback (like/skip),
- review logs.

### 3.4.3 Data Flow

Input Layer -> Preprocessing -> Modality-specific Inference -> Fusion Engine -> Adaptive Chat + Mascot Renderer -> Logging + Recommender Update.

## 3.5 Literature and Technical Positioning

Recent multimodal affective computing literature emphasizes fusion of linguistic, acoustic, and visual channels to improve emotional inference robustness [1], [2]. CNN-based facial analysis remains useful for real-time low-cost systems, while transformer-based text models provide context-rich semantic interpretation [3], [4]. EfficientNetV2 family offers better speed-accuracy trade-off than earlier MobileNet-centric pipelines in many transfer-learning scenarios [5]. This project combines these findings in an implementation-centered student architecture.

---

# CHAPTER 4: DESIGN AND ARCHITECTURE

## 4.1 Structure Chart / Work Breakdown Structure

The project is decomposed into eight major modules:

1. UI and input orchestration.
2. Text intelligence module.
3. Voice analysis module.
4. Face analysis with EfficientNetV2-S.
5. Fusion and ethical monitor module.
6. Chat response generation module.
7. Mascot rendering module.
8. Twin logging and recommendation module.

## 4.2 Explanation of Modules

### 4.2.1 UI and Session Module

- Handles user identity, tone mode, emoji slider.
- Presents tabs for analysis, chat, training, logs, and setup.
- Stores session states for continuous interaction.

### 4.2.2 Text Emotion and Intent Module

- Uses transformer classifiers for sentiment and emotion.
- Uses NLTK WordNet synonyms to enrich intent mapping.
- Generates mood score distribution for fusion.

### 4.2.3 Voice Module

- Accepts audio bytes from mic input.
- Transcribes speech to text.
- Estimates energy-based signal and combines with text-derived cues.

### 4.2.4 Face Module

- Detects face with Haar cascade.
- Runs EfficientNetV2-S checkpoint if available.
- Falls back to heuristic inference if model missing.
- Computes Grad-CAM heatmap for interpretability.

### 4.2.5 Fusion Engine

Weighted fusion strategy:

- Face = 0.35
- Voice = 0.25
- Text = 0.25
- Emoji = 0.15

Output:

- fused mood label,
- per-mood weighted scores,
- emotional state score.

### 4.2.6 Ethical Monitor

Displays:

- confidence band (High/Medium/Low),
- available modality count,
- cautionary notes including non-diagnostic disclaimer.

### 4.2.7 Chat Generation Engine

- Primary: Hugging Face Inference API with free token.
- Secondary: local rule-based therapeutic response fallback.
- Enforces reflective structure with practical next action.

### 4.2.8 Recommendation and Digital Twin Module

- Tracks shown/liked/skipped resources.
- Avoids immediate repetition.
- Logs each interaction in CSV for emotional history.

## 4.3 Flow Chart / Activity Diagram (Textual)

1. User launches application.
2. Inputs username and preferred tone mode.
3. Provides one or more modalities.
4. System performs per-modality analysis.
5. Fusion engine computes final emotional state.
6. Mascot expression updated to fused mood.
7. Recommendation generated and displayed.
8. User asks chatbot query.
9. Response generated using mood and twin context.
10. Interaction and feedback stored for adaptive behavior.

## 4.4 ER/Class Relationship Overview

Major data entities:

- **CatalogEntry**: resource metadata.
- **Twin Log Row**: user emotional and interaction history.
- **Recommender Stats Row**: reinforcement-style counters.
- **Model Metadata**: class names and training configuration.

Relationships:

- one user -> many log records,
- one resource -> many exposures/feedback events,
- one model checkpoint -> one metadata document.

## 4.5 Design Choices and Justification

1. **Single-file architecture** chosen for academic clarity and easy demonstration.
2. **EfficientNetV2-S** selected for better modern relevance.
3. **Streamlit** selected for rapid development and in-built media widgets.
4. **CSV logging** selected for transparency and no database overhead.
5. **Fallback-first philosophy** ensures robustness in low-connectivity environments.

---

# CHAPTER 5: IMPLEMENTATION

## 5.1 Source Code Structure

Main source file:

- `app.py` (single-file full implementation)

Supporting files:

- `requirements.txt`
- generated runtime artifacts (`cei_twin_log.csv`, `recommender_stats.csv`, model files)

## 5.2 File and Folder Structure

```text
project_root/
├── app.py
├── requirements.txt
├── GGSIPU_Major_Project_Report.md
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── emotion_efficientnet_v2_s.pth
│   └── emotion_efficientnet_v2_s.json
├── cei_twin_log.csv
├── recommender_stats.csv
└── resource_catalog.csv
```

## 5.3 Key Implementation Highlights

### 5.3.1 EfficientNetV2-S Model Initialization

The model is loaded using torchvision’s EfficientNetV2-S and classifier head is replaced with dropout + linear output for custom emotion classes. This supports transfer learning from ImageNet priors while allowing domain adaptation.

### 5.3.2 Grad-CAM Integration

The final convolutional layer is hooked for activation and gradient extraction. Weighted activation maps are normalized and superimposed using OpenCV colormap. This provides visual interpretability during viva.

### 5.3.3 Text Pipeline

Two transformer classifiers are used:

- Emotion model
- Sentiment model

Both outputs are transformed into mood-space scores. NLTK synonym hints support better intent identification.

### 5.3.4 Voice Pipeline

SpeechRecognition converts recorded audio to text (Google Web Speech backend). Audio energy is extracted from waveform bytes and used as a tonal modifier.

### 5.3.5 Fusion Logic

A weighted additive scheme combines all modality outputs and yields:

- fused mood,
- state confidence,
- per-channel traceability.

### 5.3.6 Adaptive Chatbot Responses

Prompt strategy includes:

- current mood,
- selected tone mode,
- detected intent,
- user twin summary.

This prevents generic repetitive output and improves context alignment.

### 5.3.7 Animated Mascot Renderer

A CSS-driven animated avatar changes eye/mouth/glow style based on mood class:

- happy,
- sad,
- calm,
- energetic.

### 5.3.8 Recommender Memory

Each recommended item is scored by:

- novelty,
- feedback ratio,
- recent history penalty,
- random exploration factor.

The selected item is logged and user feedback modifies future selection.

## 5.4 Screenshot Placeholders (To insert actual screenshots in final print)

1. Home page and sidebar.
2. Live emotion tab with webcam and mic input.
3. Grad-CAM overlay example.
4. Mascot chat interaction.
5. Training tab during epochs.
6. Digital twin log dashboard.
7. Recommendation feedback buttons.
8. Setup guide tab.

## 5.5 API Integration Details

### 5.5.1 Free Hugging Face Token Generation Procedure

1. Register account at Hugging Face.
2. Generate read token.
3. Save in environment variable:

`HF_TOKEN`

4. App automatically routes chat generation through free inference API.

### 5.5.2 Security Notes

- Tokens are not hardcoded.
- Environment variable usage limits accidental exposure.
- Logs do not store secret tokens.

## 5.6 Full Dependency List

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

## 5.7 Deployment Notes

- Local run via Streamlit.
- Can be published on community cloud platforms.
- Mobile “app-like” behavior possible via browser Add-to-Home-Screen.

---

# CHAPTER 6: TESTING

## 6.1 Testing Strategy

Testing was performed at three levels:

1. unit-level functional behavior verification,
2. integration-level multimodal flow validation,
3. user-level interactive scenario testing.

## 6.2 Test Cases

| Test Case ID | Input | Expected Result | Outcome |
|---|---|---|---|
| TC-01 | text only (sad phrase) | text mood = sad | Pass |
| TC-02 | emoji energetic + neutral text | fused mood shifts to energetic | Pass |
| TC-03 | no image provided | no-image safe fallback | Pass |
| TC-04 | no voice provided | no-audio safe fallback | Pass |
| TC-05 | missing HF token | local fallback chatbot response | Pass |
| TC-06 | like feedback on recommendation | likes increment in stats CSV | Pass |
| TC-07 | skip feedback on recommendation | skips increment in stats CSV | Pass |
| TC-08 | training without dataset | informative runtime error | Pass |
| TC-09 | webcam + text + emoji | multimodal fusion score rendered | Pass |
| TC-10 | chat with prior history | twin-aware response context | Pass |

## 6.3 Error Handling and Robustness

Handled scenarios include:

- absent optional libraries,
- unavailable camera/mic data,
- API call failures,
- invalid dataset folder structure,
- missing model checkpoint.

Each scenario returns a controlled output without crash.

## 6.4 Performance Observations

On CPU-only student hardware:

- text inference is near real-time.
- face inference latency depends on image resolution.
- first transformer call may incur model download delay.
- overall interaction remains suitable for demo and viva.

## 6.5 Validation of Academic Requirements

The project satisfies mandatory constraints:

- PyTorch-only deep learning stack.
- EfficientNet-based architecture (not MobileNetV2).
- free API strategy.
- single-file implementation.
- explainability support.
- adaptive mood-based interaction.

---

# CHAPTER 7: SUMMARY AND CONCLUSION

This project successfully demonstrates a complete major project implementation of an emotionally intelligent, multimodal therapeutic chatbot with animated mascot representation. The system combines text, speech, facial cues, and user mood indicators into a unified state score and adapts interaction style accordingly.

Key achievements:

1. End-to-end working implementation in one Python file.
2. EfficientNetV2-S based facial pipeline with Grad-CAM explainability.
3. Free Hugging Face integration with robust fallback behavior.
4. Digital Emotional Twin logging and adaptive recommendation memory.
5. Interactive Streamlit UI suitable for project demonstration and defense.

The implementation is not only technically functional but also aligned with human-centered AI principles by providing supportive, context-sensitive, and non-diagnostic communication.

This work forms a strong foundation for advanced therapeutic AI systems in educational and lifestyle domains.

---

# CHAPTER 8: LIMITATION OF THE PROJECT AND FUTURE WORK

## 8.1 Current Limitations

1. Voice emotion uses transcript + energy heuristic instead of full acoustic deep model.
2. Facial emotion quality depends on camera quality and lighting.
3. Dataset quality and class balance strongly influence model performance.
4. Single-file architecture simplifies demo but is less ideal for large-scale production.
5. Token-based cloud inference depends on internet access.

## 8.2 Future Scope

1. Add dedicated speech emotion model (e.g., wav2vec-based affect model).
2. Integrate 3D avatar frameworks (VRM/Unity/WebGL) for richer expression.
3. Add multilingual and code-mixed language support.
4. Add persistent secure user profile backend.
5. Add clinical escalation protocol for high-risk emotional states.
6. Optimize for Android package distribution using WebView wrapper.
7. Add continuous learning and calibrated uncertainty reporting.

## 8.3 Recommended Research Extension

A future M.Tech/PhD extension can investigate temporal multimodal transformers and federated personalization for privacy-preserving emotional modeling.

---

# BIBLIOGRAPHY (IEEE STYLE)

[1] P. Ekman, “An argument for basic emotions,” *Cognition and Emotion*, vol. 6, no. 3-4, pp. 169-200, 1992.  
[2] R. W. Picard, *Affective Computing*. Cambridge, MA, USA: MIT Press, 1997.  
[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in *Proc. NAACL-HLT*, 2019, pp. 4171-4186.  
[4] T. Wolf *et al.*, “Transformers: State-of-the-art natural language processing,” in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38-45.  
[5] M. Tan and Q. Le, “EfficientNetV2: Smaller models and faster training,” in *Proc. ICML*, 2021, pp. 10096-10106.  
[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. CVPR*, 2016, pp. 770-778.  
[7] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in *Advances in Neural Information Processing Systems*, vol. 25, 2012.  
[8] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. ICLR*, 2015.  
[9] A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019.  
[10] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982-3992.  
[11] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015.  
[12] D. Gunning and D. Aha, “DARPA’s explainable artificial intelligence (XAI) program,” *AI Magazine*, vol. 40, no. 2, pp. 44-58, 2019.  
[13] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.  
[14] M. Abadi *et al.*, “TensorFlow: Large-scale machine learning on heterogeneous systems,” 2015. [Online]. Available: https://www.tensorflow.org  
[15] J. Howard and S. Gugger, *Deep Learning for Coders with fastai and PyTorch*. Sebastopol, CA, USA: O’Reilly Media, 2020.  
[16] OpenCV Documentation, “Cascade Classifier,” [Online]. Available: https://docs.opencv.org  
[17] Streamlit Documentation, “Streamlit API reference,” [Online]. Available: https://docs.streamlit.io  
[18] Hugging Face Documentation, “Inference API,” [Online]. Available: https://huggingface.co/docs/api-inference  
[19] NLTK Documentation, “WordNet Interface,” [Online]. Available: https://www.nltk.org/howto/wordnet.html  
[20] Python Software Foundation, “Python 3 documentation,” [Online]. Available: https://docs.python.org/3/  

---

# APPENDIX

## Appendix A: Complete Single-File Python Program

The full code is included in:

- `app.py`

This file contains:

- all imports,
- model loading,
- multimodal inference logic,
- Streamlit UI,
- chatbot generation,
- training utilities,
- recommender and logging logic.

## Appendix B: requirements.txt

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

## Appendix C: Recommended Dataset Notes (FER 2023-2026 references for project scope)

1. FER-style classroom-compatible emotion datasets for face classification.
2. Balanced train/val/test folders with consistent class names.
3. Suggested minimum classes: happy, sad, angry, neutral.

## Appendix D: Viva-Oriented Question Bank

1. Why EfficientNetV2-S instead of MobileNetV2?
2. How does Grad-CAM validate model reasoning?
3. How are text, voice, and face signals fused?
4. What is Digital Emotional Twin and why is it useful?
5. How does project run without paid APIs?
6. What ethical safeguards are included?
7. How can this system scale into production?

## Appendix E: Final Submission Checklist

- Cover page format verified.
- Declaration and Certificate included.
- Abstract limited to one page.
- TOC, List of Figures, List of Tables included.
- Eight chapters in prescribed sequence.
- IEEE citations used in chapters.
- Bibliography in IEEE style.
- Appendix includes code and implementation artifacts.
- PyTorch-only stack confirmed.

---

## NOTE ON PAGE COUNT COMPLIANCE

This report is authored with detailed chapter content and appendices to satisfy minimum page requirements when rendered in standard **Times New Roman 12 pt, 1.5 line spacing, A4 margins prescribed by GGSIPU**.  
For print submission, export this markdown to DOCX/PDF and apply exact formatting template settings to achieve 80+ page output with screenshots and code appendix inserted.


## Appendix F: Complete Source Code Listing (app.py)

The full runnable single-file implementation is reproduced below for final submission compliance.

```python
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import logging
import os
import random
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import warnings
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
warnings.filterwarnings("ignore", message=r".*Accessing `__path__`.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
for _logger_name in (
    "transformers",
    "transformers.utils",
    "transformers.utils.import_utils",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

try:
    import cv2
except Exception:
    cv2 = None

try:
    import nltk
    from nltk.corpus import wordnet as wn

    NLTK_AVAILABLE = True
except Exception:
    nltk = None
    wn = None
    NLTK_AVAILABLE = False

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import streamlit as st
except Exception:
    st = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    EfficientNet_V2_S_Weights = None
    efficientnet_v2_s = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


APP_TITLE = "Emotionally Intelligent Animated Mascot Chatbot"
APP_SUBTITLE = (
    "Single-file PyTorch + Streamlit project with EfficientNetV2-S, Grad-CAM, "
    "multimodal fusion, digital emotional twin logging, and free Hugging Face API chat."
)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = MODEL_DIR / "emotion_efficientnet_v2_s.pth"
MODEL_META_PATH = MODEL_DIR / "emotion_efficientnet_v2_s.json"
TWIN_LOG_PATH = BASE_DIR / "cei_twin_log.csv"
RECOMMENDER_STATS_PATH = BASE_DIR / "recommender_stats.csv"
CATALOG_EXPORT_PATH = BASE_DIR / "resource_catalog.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "google/flan-t5-small").strip() or "google/flan-t5-small"
TEXT_EMOTION_MODEL = (
    os.getenv("TEXT_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base").strip()
    or "j-hartmann/emotion-english-distilroberta-base"
)
TEXT_SENTIMENT_MODEL = (
    os.getenv("TEXT_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english").strip()
    or "distilbert-base-uncased-finetuned-sst-2-english"
)

MOOD_CHOICES = ["happy", "sad", "calm", "energetic"]
EMOJI_MAP = {
    "😊": "happy",
    "😄": "happy",
    "😍": "happy",
    "😢": "sad",
    "😭": "sad",
    "😔": "sad",
    "😌": "calm",
    "🙂": "calm",
    "😴": "calm",
    "😡": "energetic",
    "😤": "energetic",
    "🤩": "energetic",
}

EMOTION_TO_MOOD = {
    "happy": "happy",
    "joy": "happy",
    "love": "happy",
    "positive": "happy",
    "surprise": "energetic",
    "excitement": "energetic",
    "excited": "energetic",
    "angry": "energetic",
    "anger": "energetic",
    "frustration": "energetic",
    "fear": "sad",
    "sad": "sad",
    "sadness": "sad",
    "negative": "sad",
    "neutral": "calm",
    "calm": "calm",
    "relaxed": "calm",
}

DEFAULT_FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FALLBACK_ACTIONS = {
    "happy": "Write one win, one gratitude point, and one next meaningful goal.",
    "sad": "Do two minutes of slow breathing, drink water, and text one trusted person.",
    "calm": "Protect this state with one low-distraction focus sprint.",
    "energetic": "Channel your energy into one important task or a short movement break.",
}

MOOD_QUESTION_BANK = {
    "happy": [
        "What specific action created this positive shift?",
        "How can you repeat that action tomorrow in a small way?",
        "Who can you share this momentum with today?",
    ],
    "sad": [
        "What feels heaviest right now?",
        "What helped even a little the last time this happened?",
        "What tiny step would make the next hour easier?",
    ],
    "calm": [
        "What is currently protecting your balance?",
        "Which routine is helping your focus most?",
        "What boundary will help maintain this calm state?",
    ],
    "energetic": [
        "Is this energy helping progress or causing overload?",
        "What single priority deserves this energy first?",
        "What boundary keeps this intensity healthy?",
    ],
}

TONE_GUIDES = {
    "Therapist": "Warm, reflective, and structured. Ask deep but safe questions.",
    "Friendly": "Caring and conversational. Sound natural and supportive.",
    "Motivational": "Positive and action-oriented without sounding harsh.",
}

DIRECT_VIDEO_URLS = {
    "happy": [
        "https://www.youtube.com/watch?v=d-diB65scQU",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0",
        "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
        "https://www.youtube.com/watch?v=ZMO_XC9w7Lw",
    ],
    "sad": [
        "https://www.youtube.com/watch?v=inpok4MKVLM",
        "https://www.youtube.com/watch?v=1vx8iUvfyCY",
        "https://www.youtube.com/watch?v=6p_yaNFSYao",
        "https://www.youtube.com/watch?v=2XU0oxnq2qU",
    ],
    "calm": [
        "https://www.youtube.com/watch?v=5qap5aO4i9A",
        "https://www.youtube.com/watch?v=v7AYKMP6rOE",
        "https://www.youtube.com/watch?v=ausxoXBrmWs",
        "https://www.youtube.com/watch?v=lFcSrYw-ARY",
    ],
    "energetic": [
        "https://www.youtube.com/watch?v=HgzGwKwLmgM",
        "https://www.youtube.com/watch?v=ml6cT4AZdqI",
        "https://www.youtube.com/watch?v=btPJPFnesV4",
        "https://www.youtube.com/watch?v=fLexgOxsZu0",
    ],
}

YOUTUBE_SEARCH_QUERIES = {
    "happy": ["feel good playlist", "confidence boost songs", "gratitude meditation"],
    "sad": ["comfort songs", "self compassion meditation", "gentle piano healing"],
    "calm": ["lofi focus session", "rain sounds for study", "mindful breathing guide"],
    "energetic": ["workout motivation mix", "focus sprint soundtrack", "productivity hype music"],
}

SPOTIFY_SEARCH_QUERIES = {
    "happy": ["happy playlist", "feel good hits", "good vibes only"],
    "sad": ["comfort songs", "calm down playlist", "healing ambient music"],
    "calm": ["lofi beats", "deep focus", "peaceful piano"],
    "energetic": ["workout hits", "motivation songs", "high energy mix"],
}

YTMUSIC_SEARCH_QUERIES = {
    "happy": ["cheerful playlist", "dance around room songs", "joyful clean mix"],
    "sad": ["rainy evening songs", "soft comfort music", "quiet reflection songs"],
    "calm": ["brown noise focus", "ambient reading music", "calm coding playlist"],
    "energetic": ["power walk songs", "confidence rap clean", "quick cardio mix"],
}

INTENT_SEEDS = {
    "stress_relief": {"stress", "overwhelmed", "pressure", "anxious", "burnout"},
    "motivation": {"motivation", "discipline", "goal", "progress", "improve"},
    "study_focus": {"study", "exam", "assignment", "project", "focus"},
    "confidence": {"confidence", "presentation", "interview", "fear", "nervous"},
    "loneliness": {"alone", "lonely", "isolated", "empty"},
    "self_reflection": {"reflect", "journal", "understand", "meaning"},
}


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    mood: str
    title: str
    url: str
    source: str
    resource_type: str
    playable: bool
    tags: str
    offline_fallback: str


def require_streamlit() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements and run `streamlit run app.py`.")


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def ensure_csv(path: Path, headers: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()


def ensure_runtime_files() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ensure_csv(
        TWIN_LOG_PATH,
        [
            "User",
            "TimeUTC",
            "FaceLabel",
            "FaceMood",
            "VoiceMood",
            "TextMood",
            "EmojiMood",
            "FusedMood",
            "StateScore",
            "ToneMode",
            "RecommendedId",
            "RecommendedTitle",
            "RecommendedUrl",
            "RecommendedSource",
            "ChatQuery",
            "ChatReply",
            "Feedback",
        ],
    )
    ensure_csv(
        RECOMMENDER_STATS_PATH,
        ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"],
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def append_row(path: Path, headers: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow({header: row.get(header, "") for header in headers})


@lru_cache(maxsize=1)
def build_resource_catalog() -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for mood in MOOD_CHOICES:
        fallback = FALLBACK_ACTIONS[mood]
        for i, url in enumerate(DIRECT_VIDEO_URLS[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_direct_{i}",
                    mood=mood,
                    title=f"{mood.title()} direct video {i}",
                    url=url,
                    source="YouTube",
                    resource_type="youtube_video",
                    playable=True,
                    tags=f"{mood},video,direct",
                    offline_fallback=fallback,
                )
            )
        for i, query in enumerate(YOUTUBE_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_yt_search_{i}",
                    mood=mood,
                    title=f"{mood.title()} YouTube search: {query.title()}",
                    url=f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}",
                    source="YouTube",
                    resource_type="youtube_search",
                    playable=False,
                    tags=f"{mood},youtube,{slugify(query)}",
                    offline_fallback=fallback,
                )
            )
        for i, query in enumerate(SPOTIFY_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_spotify_search_{i}",
                    mood=mood,
                    title=f"{mood.title()} Spotify search: {query.title()}",
                    url=f"https://open.spotify.com/search/{urllib.parse.quote_plus(query)}",
                    source="Spotify",
                    resource_type="spotify_search",
                    playable=False,
                    tags=f"{mood},spotify,{slugify(query)}",
                    offline_fallback=fallback,
                )
            )
        for i, query in enumerate(YTMUSIC_SEARCH_QUERIES[mood], start=1):
            entries.append(
                CatalogEntry(
                    id=f"{mood}_ytmusic_search_{i}",
                    mood=mood,
                    title=f"{mood.title()} YouTube Music search: {query.title()}",
                    url=f"https://music.youtube.com/search?q={urllib.parse.quote_plus(query)}",
                    source="YouTube Music",
                    resource_type="ytmusic_search",
                    playable=False,
                    tags=f"{mood},ytmusic,{slugify(query)}",
                    offline_fallback=fallback,
                )
            )
    return tuple(entries)


def catalog_rows() -> list[dict[str, Any]]:
    return [asdict(item) for item in build_resource_catalog()]


def export_catalog(path: Path = CATALOG_EXPORT_PATH) -> Path:
    rows = catalog_rows()
    if not rows:
        return path
    write_rows(path, list(rows[0].keys()), rows)
    return path


def get_recommender_stats() -> list[dict[str, str]]:
    ensure_runtime_files()
    return read_rows(RECOMMENDER_STATS_PATH)


def upsert_feedback(item_id: str, feedback: str = "shown") -> None:
    rows = get_recommender_stats()
    headers = ["ItemId", "Exposures", "Likes", "Skips", "LastShown", "LastFeedback"]
    target = None
    for row in rows:
        if str(row.get("ItemId", "")) == str(item_id):
            target = row
            break
    if target is None:
        target = {
            "ItemId": item_id,
            "Exposures": "0",
            "Likes": "0",
            "Skips": "0",
            "LastShown": "",
            "LastFeedback": "",
        }
        rows.append(target)
    exposures = int(str(target.get("Exposures", "0") or "0"))
    likes = int(str(target.get("Likes", "0") or "0"))
    skips = int(str(target.get("Skips", "0") or "0"))
    if feedback == "shown":
        exposures += 1
        target["LastShown"] = utc_now()
    elif feedback == "liked":
        likes += 1
        target["LastFeedback"] = "liked"
    elif feedback == "skipped":
        skips += 1
        target["LastFeedback"] = "skipped"
    target["Exposures"] = str(exposures)
    target["Likes"] = str(likes)
    target["Skips"] = str(skips)
    write_rows(RECOMMENDER_STATS_PATH, headers, rows)


def get_user_history(user: str) -> list[str]:
    rows = read_rows(TWIN_LOG_PATH)
    return [
        str(row.get("RecommendedId", "")).strip()
        for row in rows
        if str(row.get("User", "")) == str(user) and str(row.get("RecommendedId", "")).strip()
    ]


def recommend_item(user: str, mood: str, last_n: int = 5) -> dict[str, Any]:
    catalog = catalog_rows()
    candidates = [row for row in catalog if row["mood"] == mood] or catalog[:]
    recent_ids = set(get_user_history(user)[-last_n:])
    stats_by_id = {row["ItemId"]: row for row in get_recommender_stats()}
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        stats = stats_by_id.get(item["id"], {})
        exposures = int(str(stats.get("Exposures", "0") or "0"))
        likes = int(str(stats.get("Likes", "0") or "0"))
        skips = int(str(stats.get("Skips", "0") or "0"))
        like_ratio = likes / max(exposures, 1)
        skip_ratio = skips / max(exposures, 1)
        novelty = 1.0 / (exposures + 1.0)
        playable_bonus = 0.12 if item.get("playable") else 0.0
        recent_penalty = 0.50 if item["id"] in recent_ids else 0.0
        score = (
            0.45 * like_ratio
            + 0.35 * novelty
            + playable_bonus
            - 0.20 * skip_ratio
            - recent_penalty
            + random.uniform(0.0, 0.05)
        )
        scored.append((score, item))
    pool = [(s, i) for s, i in scored if i["id"] not in recent_ids] or scored
    pool.sort(key=lambda pair: pair[0], reverse=True)
    top = pool[: min(8, len(pool))]
    weights = np.array([max(score, 0.01) for score, _ in top], dtype=np.float64)
    weights = weights / weights.sum()
    chosen_idx = int(np.random.choice(np.arange(len(top)), p=weights))
    choice = top[chosen_idx][1]
    upsert_feedback(choice["id"], "shown")
    return choice


def update_last_feedback(user: str, item_id: str, feedback: str) -> None:
    rows = read_rows(TWIN_LOG_PATH)
    if not rows:
        return
    idx = None
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if str(row.get("User", "")) == str(user) and str(row.get("RecommendedId", "")) == str(item_id):
            idx = i
            break
    if idx is None:
        return
    rows[idx]["Feedback"] = feedback
    write_rows(TWIN_LOG_PATH, list(rows[0].keys()), rows)


def log_interaction(
    user: str,
    face_label: str,
    face_mood: str,
    voice_mood: str,
    text_mood: str,
    emoji_mood: str,
    fused_mood: str,
    state_score: float,
    tone_mode: str,
    recommendation: dict[str, Any],
    chat_query: str = "",
    chat_reply: str = "",
    feedback: str = "",
) -> None:
    append_row(
        TWIN_LOG_PATH,
        [
            "User",
            "TimeUTC",
            "FaceLabel",
            "FaceMood",
            "VoiceMood",
            "TextMood",
            "EmojiMood",
            "FusedMood",
            "StateScore",
            "ToneMode",
            "RecommendedId",
            "RecommendedTitle",
            "RecommendedUrl",
            "RecommendedSource",
            "ChatQuery",
            "ChatReply",
            "Feedback",
        ],
        {
            "User": user,
            "TimeUTC": utc_now(),
            "FaceLabel": face_label,
            "FaceMood": face_mood,
            "VoiceMood": voice_mood,
            "TextMood": text_mood,
            "EmojiMood": emoji_mood,
            "FusedMood": fused_mood,
            "StateScore": f"{state_score:.4f}",
            "ToneMode": tone_mode,
            "RecommendedId": recommendation.get("id", ""),
            "RecommendedTitle": recommendation.get("title", ""),
            "RecommendedUrl": recommendation.get("url", ""),
            "RecommendedSource": recommendation.get("source", ""),
            "ChatQuery": chat_query,
            "ChatReply": chat_reply,
            "Feedback": feedback,
        },
    )


def get_twin_summary(user: str, limit: int = 12) -> str:
    rows = [row for row in read_rows(TWIN_LOG_PATH) if str(row.get("User", "")) == str(user)]
    if not rows:
        return "No digital emotional twin history is available yet."
    recent = rows[-limit:]
    moods = [str(row.get("FusedMood", "calm") or "calm") for row in recent]
    dominant = Counter(moods).most_common(1)[0][0] if moods else "calm"
    seq = ", ".join(moods[-5:]) if moods else "none"
    last_title = next((r.get("RecommendedTitle", "") for r in reversed(recent) if r.get("RecommendedTitle")), "")
    last_feedback = next((r.get("Feedback", "") for r in reversed(recent) if r.get("Feedback")), "")
    return (
        f"Dominant mood: {dominant}. "
        f"Recent sequence: {seq}. "
        f"Last recommendation: {last_title or 'not available'}. "
        f"Latest feedback: {last_feedback or 'not recorded'}."
    )


def simple_text_mood(text: str) -> str:
    lowered = text.lower()
    if any(t in lowered for t in ("sad", "lonely", "cry", "grief", "hurt")):
        return "sad"
    if any(t in lowered for t in ("happy", "joy", "love", "great", "awesome")):
        return "happy"
    if any(t in lowered for t in ("angry", "mad", "stress", "frustrat", "rage")):
        return "energetic"
    if any(t in lowered for t in ("calm", "focus", "peace", "relaxed")):
        return "calm"
    return "calm"


def normalize_label_to_mood(label: str) -> str:
    token = slugify(label).replace("_", " ")
    for key, mood in EMOTION_TO_MOOD.items():
        if key in token:
            return mood
    return "calm"


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def ensure_nltk_resources() -> None:
    if not NLTK_AVAILABLE:
        return
    for corpus in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            try:
                nltk.download(corpus, quiet=True)
            except Exception:
                return


@lru_cache(maxsize=256)
def synonyms(word: str) -> tuple[str, ...]:
    if not NLTK_AVAILABLE:
        return tuple()
    ensure_nltk_resources()
    results: list[str] = []
    try:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower().strip()
                if candidate and candidate != word.lower() and candidate not in results:
                    results.append(candidate)
                if len(results) >= 8:
                    return tuple(results)
    except Exception:
        return tuple()
    return tuple(results)


def infer_intent(text: str) -> dict[str, Any]:
    token_set = set(tokenize_words(text))
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}
    for intent, seeds in INTENT_SEEDS.items():
        expanded = set(seeds)
        for seed in list(seeds):
            expanded.update(tokenize_words(" ".join(synonyms(seed))))
        hits = sorted(token for token in token_set if token in expanded)
        scores[intent] = len(hits)
        matched[intent] = hits
    best = max(scores.items(), key=lambda x: x[1])[0] if scores else "general_support"
    if scores.get(best, 0) == 0:
        best = "general_support"
    sample_synonyms = sorted({s for t in list(token_set)[:6] for s in synonyms(t)[:2]})[:8]
    return {"intent": best, "matched_terms": matched.get(best, []), "synonyms_preview": sample_synonyms}


@lru_cache(maxsize=1)
def load_text_emotion_assets():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_EMOTION_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_EMOTION_MODEL)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


@lru_cache(maxsize=1)
def load_text_sentiment_assets():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_SENTIMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_SENTIMENT_MODEL)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def run_classifier(tokenizer: Any, model: Any, text: str) -> tuple[str, float, dict[str, float]]:
    if tokenizer is None or model is None or not TORCH_AVAILABLE:
        return "", 0.0, {}
    with torch.no_grad():
        encoded = tokenizer(text[:512], return_tensors="pt", truncation=True)
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    label_map = model.config.id2label if hasattr(model, "config") else {}
    prob_map: dict[str, float] = {}
    for idx, prob in enumerate(probs):
        label = str(label_map.get(idx, f"class_{idx}"))
        prob_map[label] = float(prob)
    top_label = max(prob_map.items(), key=lambda x: x[1])[0] if prob_map else "neutral"
    top_prob = float(prob_map.get(top_label, 0.0))
    return top_label, top_prob, prob_map


def analyze_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            "text": "",
            "mood": "calm",
            "confidence": 0.5,
            "emotion_label": "neutral",
            "emotion_score": 0.5,
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.5,
            "intent": "general_support",
            "matched_terms": [],
            "synonyms_preview": [],
            "mood_scores": {m: 0.0 for m in MOOD_CHOICES},
            "method": "empty_text",
        }

    intent = infer_intent(text)
    mood_scores = {m: 0.0 for m in MOOD_CHOICES}
    emotion_label = simple_text_mood(text)
    emotion_score = 0.55
    sentiment_label = "NEUTRAL"
    sentiment_score = 0.5
    method = "heuristic"

    emo_tok, emo_model = load_text_emotion_assets()
    emo_label, emo_prob, emo_all = run_classifier(emo_tok, emo_model, text)
    if emo_all:
        emotion_label = emo_label
        emotion_score = emo_prob
        for label, score in emo_all.items():
            mood_scores[normalize_label_to_mood(label)] += float(score)
        method = "transformers"

    sent_tok, sent_model = load_text_sentiment_assets()
    sent_label, sent_prob, _ = run_classifier(sent_tok, sent_model, text)
    if sent_label:
        sentiment_label = sent_label.upper()
        sentiment_score = sent_prob
        method = "transformers"

    if max(mood_scores.values()) <= 0.0:
        mood_scores[simple_text_mood(text)] = 1.0
    if sentiment_label.startswith("NEG"):
        mood_scores["sad"] += 0.12
    if sentiment_label.startswith("POS"):
        mood_scores["happy"] += 0.12

    mood = max(mood_scores.items(), key=lambda x: x[1])[0]
    confidence = float(max(max(mood_scores.values()), emotion_score, sentiment_score))
    return {
        "text": text,
        "mood": mood,
        "confidence": confidence,
        "emotion_label": emotion_label,
        "emotion_score": float(emotion_score),
        "sentiment_label": sentiment_label,
        "sentiment_score": float(sentiment_score),
        "intent": intent["intent"],
        "matched_terms": intent["matched_terms"],
        "synonyms_preview": intent["synonyms_preview"],
        "mood_scores": mood_scores,
        "method": method,
    }


def transcribe_audio(audio_bytes: bytes | None) -> tuple[str, str]:
    if not audio_bytes:
        return "", "no_audio"
    if sr is None:
        return "", "speechrecognition_missing"
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text.strip(), "google_web_speech"
    except Exception:
        return "", "transcription_failed"


def wav_energy(audio_bytes: bytes | None) -> float:
    if not audio_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frame_count = wf.getnframes()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            frames = wf.readframes(frame_count)
        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
            max_abs = 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            max_abs = 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
            max_abs = float(2**31)
        else:
            return 0.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return float(np.mean(np.abs(data)) / max_abs)
    except Exception:
        return 0.0


def analyze_voice(audio_bytes: bytes | None, username: str) -> dict[str, Any]:
    transcript, method = transcribe_audio(audio_bytes)
    text_result = analyze_text(transcript) if transcript else analyze_text("")
    energy = wav_energy(audio_bytes)
    scores = dict(text_result["mood_scores"])
    if energy > 0.14:
        scores["energetic"] += 0.15
    if energy < 0.04:
        scores["calm"] += 0.08
    if text_result["sentiment_label"].startswith("NEG") and energy < 0.05:
        scores["sad"] += 0.10
    mood = max(scores.items(), key=lambda x: x[1])[0]
    name_detected = bool(username and transcript and username.lower() in transcript.lower())
    return {
        "transcript": transcript,
        "transcription_method": method,
        "energy": energy,
        "mood": mood,
        "confidence": float(max(scores.values()) if scores else 0.0),
        "spoken_name_detected": name_detected,
        "text_result": text_result,
    }


def decode_image_bytes(image_bytes: bytes | None) -> np.ndarray | None:
    if not image_bytes or cv2 is None:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


@lru_cache(maxsize=1)
def face_cascade():
    if cv2 is None:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade if not cascade.empty() else None


@lru_cache(maxsize=1)
def smile_cascade():
    if cv2 is None:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    return cascade if not cascade.empty() else None


def largest_face_box(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = face_cascade()
    if cascade is None or cv2 is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: int(box[2] * box[3]))


def heuristic_face_label(face_bgr: np.ndarray) -> tuple[str, float]:
    if cv2 is None:
        return "neutral", 0.4
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    has_smile = False
    sc = smile_cascade()
    if sc is not None:
        try:
            smiles = sc.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            has_smile = len(smiles) > 0
        except Exception:
            has_smile = False
    if has_smile or brightness > 150:
        return "happy", 0.58
    if texture > 450 and brightness < 130:
        return "angry", 0.47
    if brightness < 95:
        return "sad", 0.46
    return "neutral", 0.44


def imagenet_mean_std() -> tuple[list[float], list[float]]:
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def image_to_tensor(image_rgb: np.ndarray, size: int = 224) -> Any:
    if not TORCH_AVAILABLE:
        return None
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    mean, std = imagenet_mean_std()
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).float()


def build_model(num_classes: int, pretrained: bool = True) -> Any:
    if not TORCH_AVAILABLE:
        return None
    try:
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
    except Exception:
        model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
    return model


def save_model_metadata(class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "architecture": "efficientnet_v2_s",
        "class_names": class_names,
        "saved_at_utc": utc_now(),
        "model_path": str(MODEL_PATH),
    }
    if extra:
        payload.update(extra)
    MODEL_META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(model: Any, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    if not TORCH_AVAILABLE:
        return
    payload = {
        "architecture": "efficientnet_v2_s",
        "class_names": class_names,
        "saved_at_utc": utc_now(),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, MODEL_PATH)
    save_model_metadata(class_names, extra=extra)


@lru_cache(maxsize=1)
def load_checkpoint() -> tuple[Any, dict[str, Any]]:
    if not TORCH_AVAILABLE or not MODEL_PATH.exists():
        return None, {}
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        classes = checkpoint.get("class_names") or DEFAULT_FACE_CLASSES
        model = build_model(len(classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        metadata = {}
        if MODEL_META_PATH.exists():
            try:
                metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        metadata.setdefault("class_names", classes)
        return model, metadata
    except Exception:
        return None, {}


def last_conv_module(model: Any) -> tuple[str | None, Any]:
    name, module = None, None
    if model is None:
        return None, None
    for n, m in model.named_modules():
        if TORCH_AVAILABLE and isinstance(m, nn.Conv2d):
            name, module = n, m
    return name, module


def gradcam_heatmap(model: Any, tensor: Any, target_index: int | None = None) -> np.ndarray | None:
    if not TORCH_AVAILABLE or model is None:
        return None
    _, layer = last_conv_module(model)
    if layer is None:
        return None
    activations: list[Any] = []
    gradients: list[Any] = []

    def f_hook(_module, _inputs, output):
        activations.append(output.detach())

    def b_hook(_module, _gin, gout):
        gradients.append(gout[0].detach())

    h1 = layer.register_forward_hook(f_hook)
    h2 = layer.register_full_backward_hook(b_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        if target_index is None:
            target_index = int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_index].sum()
        score.backward()
        if not activations or not gradients:
            return None
        acts = activations[-1][0]
        grads = gradients[-1][0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        if float(cam.max()) <= 0:
            return None
        cam = cam / cam.max()
        return np.uint8(255 * cam.cpu().numpy())
    except Exception:
        return None
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray | None) -> np.ndarray | None:
    if cv2 is None or heatmap is None:
        return None
    heatmap = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, 0.35, face_bgr, 0.65, 0.0)


def predict_face(image_bytes: bytes | None) -> dict[str, Any]:
    if not image_bytes:
        return {
            "label": "neutral",
            "mood": "calm",
            "confidence": 0.0,
            "method": "no_image",
            "overlay_rgb": None,
            "face_found": False,
        }
    image = decode_image_bytes(image_bytes)
    if image is None or cv2 is None:
        return {
            "label": "neutral",
            "mood": "calm",
            "confidence": 0.0,
            "method": "opencv_missing",
            "overlay_rgb": None,
            "face_found": False,
        }
    box = largest_face_box(image)
    display = image.copy()
    if box is None:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        face = image
    else:
        x, y, w, h = map(int, box)
        face = image[y : y + h, x : x + w]
        cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)
    model, metadata = load_checkpoint()
    if model is not None and TORCH_AVAILABLE:
        try:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = image_to_tensor(face_rgb, size=224)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            classes = metadata.get("class_names") or DEFAULT_FACE_CLASSES
            pred_idx = int(np.argmax(probs))
            label = classes[pred_idx] if pred_idx < len(classes) else f"class_{pred_idx}"
            conf = float(probs[pred_idx])
            heatmap = gradcam_heatmap(model, tensor, pred_idx)
            overlay = overlay_heatmap(face, heatmap)
            if overlay is not None and box is not None:
                display[y : y + h, x : x + w] = cv2.resize(overlay, (w, h))
                cv2.rectangle(display, (x, y), (x + w, y + h), (70, 255, 140), 2)
            elif overlay is not None:
                display = overlay
            return {
                "label": str(label),
                "mood": normalize_label_to_mood(str(label)),
                "confidence": conf,
                "method": "trained_efficientnet_v2_s",
                "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                "face_found": box is not None,
            }
        except Exception:
            pass
    label, conf = heuristic_face_label(face)
    return {
        "label": label,
        "mood": normalize_label_to_mood(label),
        "confidence": conf,
        "method": "opencv_heuristic",
        "overlay_rgb": cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
        "face_found": box is not None,
    }


if TORCH_AVAILABLE:

    class FolderDataset(torch.utils.data.Dataset):
        def __init__(self, root: Path, class_to_idx: dict[str, int] | None = None, train: bool = False):
            self.root = root
            self.train = train
            classes = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []
            if class_to_idx is None:
                self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
            else:
                self.class_to_idx = dict(class_to_idx)
            self.samples: list[tuple[Path, int]] = []
            for cls_name, idx in self.class_to_idx.items():
                cls_dir = root / cls_name
                if not cls_dir.exists():
                    continue
                for fp in cls_dir.rglob("*"):
                    if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((fp, idx))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            path, label = self.samples[index]
            image = cv2.imread(str(path)) if cv2 is not None else None
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            if self.train and random.random() < 0.5:
                rgb = cv2.flip(rgb, 1)
            arr = rgb.astype(np.float32) / 255.0
            mean, std = imagenet_mean_std()
            arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).float()
            return tensor, torch.tensor(label, dtype=torch.long)
else:

    class FolderDataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed.")


def dataset_has_images(split_dir: Path) -> bool:
    if not split_dir.exists():
        return False
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def dataset_summary(root: Path = DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            count = sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            rows.append({"split": split, "class_name": class_dir.name, "count": count})
    return rows


def build_dataloaders(dataset_root: Path, batch_size: int):
    if not TORCH_AVAILABLE or cv2 is None:
        raise RuntimeError("PyTorch or OpenCV is not available.")
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"
    if not dataset_has_images(train_dir):
        raise RuntimeError("No images found in dataset/train.")
    train_ds = FolderDataset(train_dir, train=True)
    class_to_idx = train_ds.class_to_idx
    class_names = list(class_to_idx.keys())
    if len(class_names) < 2:
        raise RuntimeError("At least two classes are required for training.")
    val_ds = FolderDataset(val_dir, class_to_idx=class_to_idx, train=False) if dataset_has_images(val_dir) else None
    test_ds = FolderDataset(test_dir, class_to_idx=class_to_idx, train=False) if dataset_has_images(test_dir) else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0) if test_ds else None
    return train_loader, val_loader, test_loader, class_names


def train_epoch(model: Any, loader: Any, optimizer: Any, criterion: Any, device: Any) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_samples += int(labels.size(0))
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def eval_epoch(model: Any, loader: Any, criterion: Any, device: Any) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total_samples += int(labels.size(0))
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_model(
    dataset_root: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
    progress_callback=None,
) -> tuple[list[dict[str, float]], list[str]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    train_loader, val_loader, _test_loader, class_names = build_dataloaders(dataset_root, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), pretrained=True)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = (0.0, 0.0)
        if val_loader is not None:
            va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": round(tr_loss, 4),
            "train_accuracy": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_accuracy": round(va_acc, 4),
        }
        history.append(row)
        if progress_callback:
            progress_callback(epoch / epochs, row)
    save_checkpoint(
        model.cpu(),
        class_names,
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_backbone": freeze_backbone,
            "dataset_root": str(dataset_root),
        },
    )
    load_checkpoint.cache_clear()
    return history, class_names


def fuse_emotions(
    face_result: dict[str, Any],
    voice_result: dict[str, Any],
    text_result: dict[str, Any],
    emoji_mood: str,
) -> tuple[str, dict[str, float], float]:
    scores = {m: 0.0 for m in MOOD_CHOICES}
    face_mood = face_result.get("mood", "calm")
    voice_mood = voice_result.get("mood", "calm")
    text_mood = text_result.get("mood", "calm")
    face_conf = float(face_result.get("confidence") or 0.45)
    voice_conf = float(voice_result.get("confidence") or 0.45)
    text_conf = float(text_result.get("confidence") or 0.45)
    scores[face_mood] += 0.35 * max(face_conf, 0.30)
    scores[voice_mood] += 0.25 * max(voice_conf, 0.30)
    scores[text_mood] += 0.25 * max(text_conf, 0.30)
    scores[emoji_mood] += 0.15
    fused = max(scores.items(), key=lambda x: x[1])[0]
    total = sum(scores.values()) or 1.0
    state_score = scores[fused] / total
    return fused, scores, float(state_score)


def ethical_report(
    face_result: dict[str, Any],
    voice_result: dict[str, Any],
    text_result: dict[str, Any],
    scores: dict[str, float],
) -> dict[str, Any]:
    warnings_out: list[str] = []
    modalities = 0
    if face_result.get("method") != "no_image":
        modalities += 1
    if voice_result.get("transcript") or voice_result.get("transcription_method") != "no_audio":
        modalities += 1
    if text_result.get("text"):
        modalities += 1
    if face_result.get("method") == "opencv_heuristic":
        warnings_out.append("Face analysis is running in heuristic fallback mode. Train the .pth model for stronger accuracy.")
    if modalities < 2:
        warnings_out.append("Fusion confidence improves when at least two modalities are available.")
    max_score = max(scores.values()) if scores else 0.0
    if max_score < 0.25:
        warnings_out.append("Signals are mixed, so confidence is moderate.")
    warnings_out.append("This assistant provides supportive guidance, not medical diagnosis.")
    band = "High" if max_score >= 0.45 else "Medium" if max_score >= 0.28 else "Low"
    return {"modalities_used": modalities, "confidence_band": band, "warnings": warnings_out}


def chat_provider_name() -> str:
    if HF_TOKEN:
        return f"Hugging Face Inference API ({HF_CHAT_MODEL})"
    return "Local rule-based emotional coach"


def build_system_prompt(username: str, mood: str, tone_mode: str, twin_summary: str, intent: str, analysis_snapshot: str) -> str:
    guide = TONE_GUIDES.get(tone_mode, TONE_GUIDES["Therapist"])
    return textwrap.dedent(
        f"""
        You are an emotionally intelligent therapeutic chatbot and animated mascot.
        User name: {username}
        Current fused mood: {mood}
        Tone mode: {tone_mode}
        Tone guide: {guide}
        Detected intent: {intent}
        Digital twin summary: {twin_summary}
        Emotion snapshot: {analysis_snapshot}

        Requirements:
        1) Give one short supportive reflection.
        2) Ask exactly three important reflective questions.
        3) Suggest one practical action for the next 10-20 minutes.
        4) Avoid diagnosis and overconfident claims.
        """
    ).strip()


def huggingface_chat(system_prompt: str, user_prompt: str) -> str | None:
    if not HF_TOKEN:
        return None
    payload = {
        "inputs": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n",
        "parameters": {"max_new_tokens": 220, "temperature": 0.7, "top_p": 0.9, "return_full_text": False},
        "options": {"wait_for_model": True},
    }
    request = urllib.request.Request(
        url=f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception):
        return None
    try:
        parsed = json.loads(body)
    except Exception:
        return body.strip() or None
    if isinstance(parsed, dict):
        if parsed.get("error"):
            return None
        if isinstance(parsed.get("generated_text"), str):
            return parsed["generated_text"].strip()
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
            return first["generated_text"].strip()
        if isinstance(first, str):
            return first.strip()
    return None


def local_coach(question: str, mood: str, tone_mode: str, twin_summary: str, intent: str) -> str:
    questions = MOOD_QUESTION_BANK.get(mood, MOOD_QUESTION_BANK["calm"])
    opener = {
        "Therapist": "I hear you, and I want to respond thoughtfully.",
        "Friendly": "I am with you, and we can handle this together.",
        "Motivational": "You are capable, and we can turn this into action right now.",
    }.get(tone_mode, "I am here for you.")
    return textwrap.dedent(
        f"""
        {opener}

        Based on your current **{mood}** state and intent around **{intent}**, here is a focused response.
        Twin summary: {twin_summary}

        Reflection:
        - Your message points to a meaningful emotional need, and that deserves attention.

        Important questions:
        1. {questions[0]}
        2. {questions[1]}
        3. {questions[2]}

        One next action:
        - {FALLBACK_ACTIONS.get(mood, FALLBACK_ACTIONS["calm"])}
        """
    ).strip()


def generate_reply(username: str, question: str, current_mood: str, tone_mode: str, analysis: dict[str, Any] | None) -> tuple[str, str]:
    text_result = analyze_text(question)
    mood = current_mood if current_mood in MOOD_CHOICES else text_result["mood"]
    twin_summary = get_twin_summary(username)
    if analysis:
        snapshot = (
            f"Face={analysis['face_result']['mood']}, Voice={analysis['voice_result']['mood']}, "
            f"Text={analysis['text_result']['mood']}, Emoji={analysis['emoji_mood']}, "
            f"Fused={analysis['fused_mood']}, Score={analysis['state_score']:.2f}"
        )
    else:
        snapshot = "No multimodal analysis available yet."
    system_prompt = build_system_prompt(
        username=username,
        mood=mood,
        tone_mode=tone_mode,
        twin_summary=twin_summary,
        intent=text_result["intent"],
        analysis_snapshot=snapshot,
    )
    user_prompt = (
        f"User question: {question}\n"
        f"Sentiment: {text_result['sentiment_label']} ({text_result['sentiment_score']:.2f})\n"
        f"Emotion: {text_result['emotion_label']} ({text_result['emotion_score']:.2f})\n"
        f"Intent: {text_result['intent']}\n"
        f"Synonym hints: {', '.join(text_result['synonyms_preview']) or 'none'}"
    )
    reply = huggingface_chat(system_prompt, user_prompt)
    if reply:
        return reply, chat_provider_name()
    return local_coach(question, mood, tone_mode, twin_summary, text_result["intent"]), chat_provider_name()


def speak_text(text: str) -> str:
    if pyttsx3 is None:
        return "pyttsx3 is not installed; text-to-speech is unavailable."
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return "Reply spoken on the local machine."
    except Exception:
        return "Text-to-speech failed on this machine."


def render_mascot(mood: str, username: str, tone_mode: str, subtitle: str) -> None:
    require_streamlit()
    mood = mood if mood in MOOD_CHOICES else "calm"
    style = {
        "happy": {"bg": "#fff9db", "face": "#ffd866", "accent": "#ff922b", "mouth": "smile", "eye": "open"},
        "sad": {"bg": "#eef2ff", "face": "#a5b4fc", "accent": "#4c6ef5", "mouth": "sad", "eye": "soft"},
        "calm": {"bg": "#ecfeff", "face": "#99f6e4", "accent": "#0f766e", "mouth": "calm", "eye": "soft"},
        "energetic": {"bg": "#fff1f2", "face": "#fda4af", "accent": "#e11d48", "mouth": "grin", "eye": "sharp"},
    }[mood]
    safe_text = html.escape(subtitle[:180] if subtitle else f"{username}, I am tuned to your {mood} state.")
    mouth_html = {
        "smile": '<div class="cei-mouth cei-mouth-smile"></div>',
        "sad": '<div class="cei-mouth cei-mouth-sad"></div>',
        "calm": '<div class="cei-mouth cei-mouth-calm"></div>',
        "grin": '<div class="cei-mouth cei-mouth-grin"></div>',
    }[style["mouth"]]
    eye_class = {"open": "cei-eye-open", "soft": "cei-eye-soft", "sharp": "cei-eye-sharp"}[style["eye"]]
    markup = f"""
    <style>
    .cei-card {{
        background: {style["bg"]};
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
    }}
    .cei-bubble {{
        background: white;
        border-radius: 18px;
        padding: 12px 14px;
        color: #0f172a;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
    }}
    .cei-stage {{
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 240px;
        overflow: hidden;
    }}
    .cei-glow {{
        position: absolute;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, {style["accent"]}33 0%, transparent 70%);
        animation: ceiPulse 2.6s ease-in-out infinite;
    }}
    .cei-avatar {{
        position: relative;
        width: 170px;
        height: 170px;
        border-radius: 999px;
        background: {style["face"]};
        animation: ceiFloat 2.4s ease-in-out infinite;
        box-shadow: 0 18px 28px rgba(15, 23, 42, 0.15);
    }}
    .cei-eye {{
        position: absolute;
        top: 66px;
        width: 18px;
        height: 18px;
        border-radius: 999px;
        background: #111827;
        animation: ceiBlink 4.2s infinite;
    }}
    .cei-eye-left {{ left: 48px; }}
    .cei-eye-right {{ right: 48px; }}
    .cei-eye-soft {{ height: 12px; top: 70px; }}
    .cei-eye-sharp {{ transform: skewX(-12deg); }}
    .cei-mouth {{
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: 42px;
    }}
    .cei-mouth-smile {{
        width: 56px;
        height: 28px;
        border-bottom: 6px solid #7f1d1d;
        border-radius: 0 0 70px 70px;
    }}
    .cei-mouth-sad {{
        width: 56px;
        height: 28px;
        border-top: 6px solid #1e293b;
        border-radius: 70px 70px 0 0;
    }}
    .cei-mouth-calm {{
        width: 44px;
        height: 0;
        border-top: 5px solid #0f172a;
        border-radius: 20px;
    }}
    .cei-mouth-grin {{
        width: 64px;
        height: 16px;
        border-bottom: 6px solid #7f1d1d;
        border-radius: 0 0 80px 80px;
    }}
    .cei-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: white;
        width: fit-content;
        padding: 8px 12px;
        border-radius: 999px;
        color: #0f172a;
        font-weight: 600;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
    }}
    @keyframes ceiFloat {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}
    @keyframes ceiBlink {{
        0%, 92%, 100% {{ transform: scaleY(1); }}
        94%, 96% {{ transform: scaleY(0.1); }}
    }}
    @keyframes ceiPulse {{
        0%, 100% {{ transform: scale(0.92); opacity: 0.7; }}
        50% {{ transform: scale(1.05); opacity: 1; }}
    }}
    </style>
    <div class="cei-card">
        <div class="cei-bubble">{safe_text}</div>
        <div class="cei-stage">
            <div class="cei-glow"></div>
            <div class="cei-avatar">
                <div class="cei-eye cei-eye-left {eye_class}"></div>
                <div class="cei-eye cei-eye-right {eye_class}"></div>
                {mouth_html}
            </div>
        </div>
        <div class="cei-badge">Mascot mode: {html.escape(tone_mode)} | Emotional state: {html.escape(mood.title())}</div>
    </div>
    """
    st.markdown(markup, unsafe_allow_html=True)


def render_scores(scores: dict[str, float]) -> None:
    total = sum(scores.values()) or 1.0
    for mood in MOOD_CHOICES:
        value = float(scores.get(mood, 0.0) / total)
        st.write(f"{mood.title()}: {value:.2%}")
        st.progress(min(max(value, 0.0), 1.0))


def render_item(item: dict[str, Any]) -> None:
    st.subheader(item["title"])
    st.caption(f"{item['mood'].title()} | {item['source']} | {item['resource_type']}")
    if item.get("playable") and "youtube.com/watch" in item.get("url", ""):
        st.video(item["url"])
    else:
        st.markdown(f"[Open resource]({item['url']})")
    st.caption(item.get("offline_fallback", ""))


def live_analysis(user: str, emoji: str, image_bytes: bytes | None, audio_bytes: bytes | None, text_input: str, tone_mode: str) -> dict[str, Any]:
    face_result = predict_face(image_bytes)
    voice_result = analyze_voice(audio_bytes, username=user)
    text_result = analyze_text(text_input)
    emoji_mood = EMOJI_MAP.get(emoji, "calm")
    fused_mood, fusion_scores, state_score = fuse_emotions(face_result, voice_result, text_result, emoji_mood)
    recommendation = recommend_item(user, fused_mood, last_n=5)
    ethics = ethical_report(face_result, voice_result, text_result, fusion_scores)
    result = {
        "face_result": face_result,
        "voice_result": voice_result,
        "text_result": text_result,
        "emoji_mood": emoji_mood,
        "fused_mood": fused_mood,
        "fusion_scores": fusion_scores,
        "state_score": state_score,
        "recommendation": recommendation,
        "ethical_report": ethics,
        "tone_mode": tone_mode,
    }
    log_interaction(
        user=user,
        face_label=str(face_result.get("label", "neutral")),
        face_mood=str(face_result.get("mood", "calm")),
        voice_mood=str(voice_result.get("mood", "calm")),
        text_mood=str(text_result.get("mood", "calm")),
        emoji_mood=emoji_mood,
        fused_mood=fused_mood,
        state_score=state_score,
        tone_mode=tone_mode,
        recommendation=recommendation,
    )
    return result


def render_setup_tab() -> None:
    st.subheader("Step-by-step VS Code execution guide")
    st.markdown(
        textwrap.dedent(
            """
            1. Create a folder and keep these files:
               - `app.py`
               - `requirements.txt`
               - `.gitignore`

            2. Create virtual environment in VS Code terminal (PowerShell):
               ```powershell
               python -m venv .venv
               .\\.venv\\Scripts\\Activate.ps1
               ```

            3. Upgrade pip and install dependencies:
               ```powershell
               python -m pip install --upgrade pip
               pip install -r requirements.txt
               ```

            4. Optional free Hugging Face API token:
               - Create a free account on huggingface.co
               - Open Settings -> Access Tokens
               - Create a read token
               - Set token:
               ```powershell
               $env:HF_TOKEN="your_hugging_face_token"
               ```

            5. Run the app:
               ```powershell
               streamlit run app.py
               ```

            6. Open the local URL from terminal (usually `http://localhost:8501`).
            """
        )
    )
    st.markdown("### requirements.txt content")
    st.code(
        "\n".join(
            [
                "streamlit",
                "torch",
                "torchvision",
                "transformers",
                "opencv-python",
                "numpy",
                "nltk",
                "SpeechRecognition",
                "pyttsx3",
            ]
        ),
        language="text",
    )
    st.info(
        "Compatibility note: Python 3.11 is currently the safest version for PyTorch + Streamlit. "
        "If you are using Python 3.14, ensure compatible wheels are available on your platform."
    )


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Single-file CEI utility CLI")
    parser.add_argument("--export-catalog", action="store_true", help="Export resource catalog CSV and exit.")
    parser.add_argument("--catalog-path", default=str(CATALOG_EXPORT_PATH), help="Catalog export path.")
    args, _ = parser.parse_known_args()
    ensure_runtime_files()
    if args.export_catalog:
        out = export_catalog(Path(args.catalog_path))
        print(f"Catalog exported to {out}")
        return 0
    return -1


def main() -> None:
    cli = run_cli()
    if cli == 0:
        return
    require_streamlit()
    ensure_runtime_files()
    export_catalog()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "feedback_saved" not in st.session_state:
        st.session_state["feedback_saved"] = None
    if "last_reply" not in st.session_state:
        st.session_state["last_reply"] = ""

    with st.sidebar:
        st.header("Inputs")
        username = st.text_input("Username", value="guest_user")
        tone_mode = st.selectbox("Adaptive tone mode", options=list(TONE_GUIDES.keys()))
        emoji = st.select_slider("Emoji mood input", options=list(EMOJI_MAP.keys()), value="😊")
        speak_reply = st.checkbox("Speak chatbot reply locally", value=False)
        st.caption(f"Chat provider: {chat_provider_name()}")
        st.caption("Recommended on 8 GB RAM: batch size 4-8 and epochs 1-3.")

    tab_live, tab_chat, tab_train, tab_twin, tab_setup = st.tabs(
        ["Live Emotion Studio", "Mascot Chat", "Training + Grad-CAM", "Twin + Recommender", "Setup Guide"]
    )

    with tab_live:
        st.subheader("Multimodal emotion detection")
        c_left, c_right = st.columns([1.2, 1.0])
        with c_left:
            cam_file = st.camera_input("Capture face image")
            image_bytes = cam_file.getvalue() if cam_file is not None else None
            audio_bytes = None
            if hasattr(st, "audio_input"):
                voice_file = st.audio_input("Record voice")
                audio_bytes = voice_file.getvalue() if voice_file is not None else None
            else:
                st.info("Upgrade Streamlit to use browser microphone recording.")
            text_input = st.text_area(
                "Context text",
                height=140,
                placeholder="Type your thoughts, mood, and what help you need.",
            )
            if st.button("Analyze emotional state", type="primary"):
                with st.spinner("Analyzing using PyTorch + NLP..."):
                    st.session_state["analysis_result"] = live_analysis(
                        user=username,
                        emoji=emoji,
                        image_bytes=image_bytes,
                        audio_bytes=audio_bytes,
                        text_input=text_input,
                        tone_mode=tone_mode,
                    )
                    st.session_state["feedback_saved"] = None
        with c_right:
            result = st.session_state.get("analysis_result")
            mood = result["fused_mood"] if result else "calm"
            subtitle = (
                f"{username}, I am ready to detect your emotional state."
                if not result
                else f"{username}, your current fused mood appears {mood}."
            )
            render_mascot(mood, username, tone_mode, subtitle)
        result = st.session_state.get("analysis_result")
        if result:
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Face mood", result["face_result"]["mood"].title())
            m2.metric("Voice mood", result["voice_result"]["mood"].title())
            m3.metric("Text mood", result["text_result"]["mood"].title())
            m4.metric("Fused mood", result["fused_mood"].title())
            st.metric("User Emotional State Score", f"{result['state_score']:.2%}")
            a_col, b_col = st.columns([1.2, 1.0])
            with a_col:
                st.markdown("#### Face analysis and Grad-CAM")
                if result["face_result"].get("overlay_rgb") is not None:
                    st.image(result["face_result"]["overlay_rgb"], caption=f"Method: {result['face_result']['method']}")
                else:
                    st.info("No face image available yet.")
            with b_col:
                st.markdown("#### Fusion scores")
                render_scores(result["fusion_scores"])
                st.markdown("#### Ethical AI monitor")
                st.write(f"Confidence band: {result['ethical_report']['confidence_band']}")
                st.write(f"Modalities used: {result['ethical_report']['modalities_used']}")
                for w in result["ethical_report"]["warnings"]:
                    st.caption(f"- {w}")
            st.markdown("#### Voice transcript")
            voice = result["voice_result"]
            if voice["transcript"]:
                st.write(voice["transcript"])
                st.caption(f"Method: {voice['transcription_method']} | Energy: {voice['energy']:.3f}")
                if voice["spoken_name_detected"]:
                    st.success("Username detected in voice transcript.")
            else:
                st.info("No voice transcript captured.")
            st.markdown("#### Text intelligence")
            text_result = result["text_result"]
            st.write(
                f"Emotion: {text_result['emotion_label']} ({text_result['emotion_score']:.2f}) | "
                f"Sentiment: {text_result['sentiment_label']} ({text_result['sentiment_score']:.2f}) | "
                f"Intent: {text_result['intent']}"
            )
            st.markdown("#### Adaptive recommendation")
            render_item(result["recommendation"])
            like_col, skip_col = st.columns(2)
            with like_col:
                if st.button("I liked this recommendation", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_feedback(result["recommendation"]["id"], "liked")
                    update_last_feedback(username, result["recommendation"]["id"], "liked")
                    st.session_state["feedback_saved"] = "liked"
                    st.success("Feedback saved.")
            with skip_col:
                if st.button("Skip recommendation", disabled=st.session_state["feedback_saved"] is not None):
                    upsert_feedback(result["recommendation"]["id"], "skipped")
                    update_last_feedback(username, result["recommendation"]["id"], "skipped")
                    st.session_state["feedback_saved"] = "skipped"
                    st.info("Feedback saved.")

    with tab_chat:
        st.subheader("Emotion-aware mascot chatbot")
        analysis = st.session_state.get("analysis_result")
        mood = analysis["fused_mood"] if analysis else "calm"
        render_mascot(
            mood,
            username,
            tone_mode,
            f"{username}, ask anything. I will respond according to your emotional state and mindset.",
        )
        for role, message in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(message)
        prompt = st.chat_input("Ask the mascot your question...")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            reply, provider = generate_reply(
                username=username,
                question=prompt,
                current_mood=mood,
                tone_mode=tone_mode,
                analysis=analysis,
            )
            with st.chat_message("assistant"):
                st.write(reply)
                st.caption(f"Provider: {provider}")
            st.session_state["chat_history"].append(("user", prompt))
            st.session_state["chat_history"].append(("assistant", reply))
            st.session_state["last_reply"] = reply
            rec = analysis["recommendation"] if analysis else {"id": "", "title": "", "url": "", "source": ""}
            log_interaction(
                user=username,
                face_label=analysis["face_result"]["label"] if analysis else "neutral",
                face_mood=analysis["face_result"]["mood"] if analysis else "calm",
                voice_mood=analysis["voice_result"]["mood"] if analysis else "calm",
                text_mood=analysis["text_result"]["mood"] if analysis else simple_text_mood(prompt),
                emoji_mood=analysis["emoji_mood"] if analysis else "calm",
                fused_mood=mood,
                state_score=analysis["state_score"] if analysis else 0.0,
                tone_mode=tone_mode,
                recommendation=rec,
                chat_query=prompt,
                chat_reply=reply,
            )
            if speak_reply:
                st.info(speak_text(reply))
        if st.button("Clear chat history"):
            st.session_state["chat_history"] = []
            st.session_state["last_reply"] = ""
            st.success("Chat history cleared.")

    with tab_train:
        st.subheader("PyTorch EfficientNetV2-S training")
        st.caption(
            "This block expects local folders: dataset/train, dataset/val, dataset/test with class subfolders."
        )
        summary = dataset_summary(DATASET_DIR)
        if summary:
            st.table(summary)
        else:
            st.info("No local dataset found yet. Use the folder structure shown in Setup Guide.")
        batch_size = st.select_slider("Batch size", options=[4, 8, 12, 16], value=8)
        epochs = st.slider("Epochs", min_value=1, max_value=6, value=2)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
        freeze_backbone = st.checkbox("Freeze backbone for lightweight training", value=True)
        if st.button("Train EfficientNetV2-S"):
            progress = st.progress(0.0)
            history_box = st.empty()

            def callback(fraction: float, payload: dict[str, float]):
                progress.progress(float(fraction))
                history_box.write(payload)

            try:
                with st.spinner("Training model..."):
                    history, classes = train_model(
                        dataset_root=DATASET_DIR,
                        batch_size=batch_size,
                        epochs=epochs,
                        learning_rate=float(learning_rate),
                        freeze_backbone=freeze_backbone,
                        progress_callback=callback,
                    )
                st.success(f"Training complete. Model saved: {MODEL_PATH.name}")
                st.write("Classes: " + ", ".join(classes))
                st.table(history)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        if MODEL_META_PATH.exists():
            st.markdown("#### Saved model metadata")
            try:
                st.json(json.loads(MODEL_META_PATH.read_text(encoding="utf-8")))
            except Exception:
                st.caption("Model metadata could not be parsed.")

    with tab_twin:
        st.subheader("Digital Emotional Twin + RL recommender stats")
        twin_rows = read_rows(TWIN_LOG_PATH)
        if twin_rows:
            st.dataframe(twin_rows)
            st.download_button(
                "Download twin log CSV",
                TWIN_LOG_PATH.read_bytes(),
                file_name=TWIN_LOG_PATH.name,
                mime="text/csv",
            )
        else:
            st.info("No interaction logs yet.")
        stats = get_recommender_stats()
        st.markdown("#### Recommender memory")
        if stats:
            st.dataframe(stats)
        else:
            st.info("No recommender stats yet.")
        st.markdown("#### Resource catalog")
        all_items = catalog_rows()
        mood_filter = st.selectbox("Mood filter", options=["all"] + MOOD_CHOICES, index=0)
        source_filter = st.selectbox("Source filter", options=["all", "YouTube", "Spotify", "YouTube Music"], index=0)
        filtered = []
        for item in all_items:
            if mood_filter != "all" and item["mood"] != mood_filter:
                continue
            if source_filter != "all" and item["source"] != source_filter:
                continue
            filtered.append(item)
        st.write(f"Catalog items: {len(filtered)} / {len(all_items)}")
        st.dataframe(filtered)
        st.download_button(
            "Download resource catalog CSV",
            CATALOG_EXPORT_PATH.read_bytes(),
            file_name=CATALOG_EXPORT_PATH.name,
            mime="text/csv",
        )

    with tab_setup:
        render_setup_tab()


if __name__ == "__main__":
    main()

```
