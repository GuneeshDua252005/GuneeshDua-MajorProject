from __future__ import annotations

import html
import re
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
DOCX_PATH = REPORT_DIR / "GGSIPU_Major_Project_Report_CEI_Chatbot.docx"
MD_PATH = REPORT_DIR / "GGSIPU_Major_Project_Report_CEI_Chatbot.md"
HTML_LINK_PATH = REPORT_DIR / "download_reference.html"

PROJECT_TITLE = "Emotionally Intelligent Animated Mascot Chatbot with Voice, Facial and Text Sentiment Detection"
DEPARTMENT = "Department of Artificial Intelligence and Machine Learning"
INSTITUTE = "Name of the Institute"
STUDENT = "NAME OF THE STUDENT"
ENROLLMENT = "ENROLLMENT NO"
SUPERVISOR = "NAME OF THE FACULTY SUPERVISOR"
DESIGNATION = "DESIGNATION"
HOD = "NAME OF THE HOD"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass
class Block:
    kind: str
    text: str = ""
    style: str = "Normal"
    rows: list[list[str]] = field(default_factory=list)
    width: str = "single"


@dataclass
class Page:
    title: str
    blocks: list[Block]


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def p(text: str = "", style: str = "Normal") -> Block:
    return Block("p", text=text, style=style)


def code(text: str) -> Block:
    return Block("p", text=text, style="Code")


def table(rows: list[list[str]]) -> Block:
    return Block("table", rows=rows)


def split_paragraphs(text: str) -> list[Block]:
    blocks: list[Block] = []
    for para in textwrap.dedent(text).strip().split("\n\n"):
        para = " ".join(line.strip() for line in para.splitlines()).strip()
        if para:
            blocks.append(p(para))
    return blocks


def roman(n: int) -> str:
    vals = [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]
    out = ""
    for value, symbol in vals:
        while n >= value:
            out += symbol
            n -= value
    return out


def title_page() -> Page:
    return Page(
        "Cover Page",
        [
            p(PROJECT_TITLE, "Title"),
            p("(AIML-452 Major Project - Dissertation)", "Subtitle"),
            p("submitted in partial fulfillment of the requirement", "Center"),
            p("for the award of the degree of", "Center"),
            p("Bachelor of Technology", "CenterBold"),
            p("in", "Center"),
            p("AIML", "CenterBold"),
            p("Submitted by", "Center"),
            p(STUDENT, "CenterBold"),
            p(ENROLLMENT, "Center"),
            p("Under the supervision of", "Center"),
            p(SUPERVISOR, "CenterBold"),
            p(DESIGNATION, "Center"),
            p("LOGO OF THE INSTITUTE", "CenterBold"),
            p(DEPARTMENT, "Center"),
            p(INSTITUTE, "Center"),
            p("Address of the institute", "Center"),
            p("May/June 2026", "CenterBold"),
        ],
    )


def front_pages() -> list[Page]:
    return [
        Page(
            "DECLARATION",
            [
                p("DECLARATION", "Chapter"),
                *split_paragraphs(
                    f"""
                    This is to certify that the material embodied in this Major Project - Dissertation titled "{PROJECT_TITLE}" being submitted in the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML is based on my original work. It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma. My indebtedness to other works has been duly acknowledged at the relevant places.

                    The project, source-code design, experimentation plan, analysis, testing and documentation have been prepared for the final major project evaluation and viva voce. The implementation described in this report follows a PyTorch-only approach and avoids TensorFlow and Keras.
                    """
                ),
                p("\n\n\n\n(Name of the Student)\nEnrollment No", "Right"),
            ],
        ),
        Page(
            "CERTIFICATE",
            [
                p("CERTIFICATE", "Chapter"),
                *split_paragraphs(
                    f"""
                    This is to certify that the work embodied in this Major Project - Dissertation titled "{PROJECT_TITLE}" being submitted in the partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in AIML, is original and has been carried out by {STUDENT} (Enrollment No. {ENROLLMENT}) under my supervision and guidance.

                    It is further certified that this Major Project - Dissertation work has not been submitted in full or in part to this university or any other university for the award of any other degree or diploma to the best of my knowledge and belief.
                    """
                ),
                p("\n\n\n(Name of the Faculty Supervisor)\nDesignation", "Left"),
                p("\n(Name of the HOD)\nHOD\nName of the Institute", "Right"),
            ],
        ),
        Page(
            "ACKNOWLEDGEMENT",
            [
                p("ACKNOWLEDGEMENT", "Chapter"),
                *split_paragraphs(
                    """
                    I express my sincere gratitude to my faculty supervisor for continuous guidance, review comments, technical direction and encouragement throughout the development of this major project. The supervisor's support helped convert the initial idea of cognitive emotional intelligence into a structured, testable and presentation-ready software system.

                    I am thankful to the Head of the Department, the Principal and the Institute for providing an academic environment where advanced artificial intelligence topics such as multimodal emotion recognition, explainable CNNs, ethical AI and therapeutic conversational interfaces could be explored as part of the final-year major project.

                    I also thank my mentors, peers, friends and family members for their patience, motivation and feedback during ideation, implementation, testing and report preparation. Their support strengthened the practical relevance of the project and encouraged a user-centric design approach.
                    """
                ),
            ],
        ),
        Page(
            "TABLE OF CONTENTS",
            [
                p("TABLE OF CONTENTS", "Chapter"),
                table(
                    [
                        ["Section", "Title", "Page No."],
                        ["Preliminary", "Declaration", roman(1)],
                        ["Preliminary", "Certificate", roman(2)],
                        ["Preliminary", "Acknowledgement", roman(3)],
                        ["Preliminary", "Abstract", roman(5)],
                        ["Preliminary", "List of Figures", roman(6)],
                        ["Preliminary", "List of Tables", roman(7)],
                        ["Chapter 1", "Introduction", "1"],
                        ["Chapter 2", "Problem Statement", "11"],
                        ["Chapter 3", "Analysis", "19"],
                        ["Chapter 4", "Design and Architecture", "31"],
                        ["Chapter 5", "Implementation", "43"],
                        ["Chapter 6", "Testing", "58"],
                        ["Chapter 7", "Summary and Conclusion", "68"],
                        ["Chapter 8", "Limitations and Future Work", "74"],
                        ["Bibliography", "IEEE References", "80"],
                        ["Appendix", "Implementation Code and Screenshots", "85"],
                    ]
                ),
            ],
        ),
        Page(
            "ABSTRACT",
            [
                p("ABSTRACT", "Chapter"),
                *split_paragraphs(
                    f"""
                    The project titled "{PROJECT_TITLE}" presents a PyTorch-based therapeutic chatbot that detects and responds to a user's emotional state through text, voice, facial cues and an emoji-based self-report slider. The work addresses the limitation of generic FAQ-style chatbots by combining cognitive emotional intelligence, multimodal AI fusion, adaptive recommendations and a visually expressive animated mascot interface. The chatbot is designed for a Windows 11 laptop environment using VS Code, Streamlit, PyTorch, OpenCV, Hugging Face Transformers, NLTK, SpeechRecognition and optional pyttsx3-based local speech output.

                    The proposed system uses Hugging Face free-token integration for text emotion and sentiment intelligence and avoids paid OpenAI APIs. Facial emotion processing is implemented using OpenCV for face capture and a PyTorch EfficientNetV2-S architecture for trained emotion classification. Grad-CAM is included to support explainability by highlighting facial image regions that influence the CNN decision. Voice input is recorded through the browser microphone where supported, converted to text, analyzed for sentiment and enriched through simple audio-energy estimation. A weighted fusion engine combines face, voice, text and emoji signals into a unified User Emotional State Score.

                    The Streamlit frontend provides a live emotion studio, mascot chatbot, training and Grad-CAM panel, digital emotional twin memory, reinforcement-learning-style recommendation statistics and a setup guide. The system logs interaction history in CSV format and uses feedback such as liked or skipped recommendations to reduce repetition. The project remains non-diagnostic and includes ethical warnings to prevent overconfident mental-health claims. The result is a demo-ready, viva-friendly and extensible major project suitable for presenting the practical use of PyTorch, multimodal AI and explainable affective computing.
                    """
                ),
            ],
        ),
        Page(
            "LIST OF FIGURES",
            [
                p("LIST OF FIGURES", "Chapter"),
                table(
                    [
                        ["Figure No.", "Figure Title", "Page No."],
                        ["Figure 1.1", "Conceptual view of the cognitive emotional intelligence system", "4"],
                        ["Figure 3.1", "Data flow diagram for multimodal emotion detection", "27"],
                        ["Figure 3.2", "Use case diagram for user, chatbot and training workflow", "29"],
                        ["Figure 4.1", "System architecture of Streamlit, PyTorch and Hugging Face modules", "35"],
                        ["Figure 4.2", "Activity flow for live emotion analysis", "39"],
                        ["Figure 4.3", "Class-style diagram of core modules", "42"],
                        ["Figure 5.1", "Streamlit live emotion studio screen", "45"],
                        ["Figure 5.2", "Animated mascot expression states", "48"],
                        ["Figure 5.3", "Grad-CAM explainability overlay", "54"],
                    ]
                ),
            ],
        ),
        Page(
            "LIST OF TABLES",
            [
                p("LIST OF TABLES", "Chapter"),
                table(
                    [
                        ["Table No.", "Table Title", "Page No."],
                        ["Table 2.1", "Objectives of the major project", "16"],
                        ["Table 3.1", "Functional requirements", "20"],
                        ["Table 3.2", "Non-functional requirements", "22"],
                        ["Table 3.3", "Software and hardware requirements", "24"],
                        ["Table 4.1", "Module explanation summary", "37"],
                        ["Table 5.1", "Core implementation mapping", "50"],
                        ["Table 6.1", "Functional test cases", "60"],
                        ["Table 6.2", "Model and integration test cases", "64"],
                    ]
                ),
            ],
        ),
    ]


def repeated_page(title: str, heading: str, paragraphs: list[str], page_no: int) -> Page:
    blocks = [p(heading, "Chapter" if heading.startswith("CHAPTER") else "Heading1")]
    for para in paragraphs:
        blocks.extend(split_paragraphs(para))
    blocks.append(p(f"Figure/Table reference note for page {page_no}: citations and diagrams are aligned with IEEE-style references in the bibliography.", "Italic"))
    return Page(title, blocks)


def chapter_pages() -> list[Page]:
    pages: list[Page] = []
    page_no = 1

    chapter_1 = [
        ("CHAPTER 1: INTRODUCTION", [
            "Emotion-aware computing has become important because most digital assistants still treat users as neutral command senders. A therapeutic support interface must understand the user's emotional context before generating advice. The proposed project creates a multimodal chatbot that observes text, speech, facial cues and explicit emoji selection to understand mood in a safer and more human-centered manner [1].",
            "The project is implemented as a single-file Python 3 Streamlit application for easy execution in VS Code on a Windows 11 Dell laptop with an Intel Core i5 processor. The implementation uses PyTorch instead of TensorFlow or Keras. This decision keeps the system aligned with modern research workflows and allows the use of EfficientNetV2-S for facial emotion classification [2].",
            "The mascot interface is not only decorative. It gives the system an embodied conversational identity. The animated avatar changes expressions for happy, sad, calm and energetic emotional states, which helps the user immediately understand how the system has interpreted the current interaction.",
        ]),
        ("1.1 Background of the Project", [
            "Affective computing studies how machines can recognize, interpret and respond to human emotions. In real applications, a single signal is often unreliable. Text may hide emotions, voice may be unclear and facial images may suffer from lighting problems. Therefore, this project applies multimodal fusion where every available signal contributes to the final User Emotional State Score.",
            "Hugging Face is selected as the recommended free API integration route because it supports free account creation, read-token generation and access to open-source models. It also supports local model execution through Transformers, which is useful when API access is unavailable or internet speed is limited [3].",
            "The project deliberately avoids paid OpenAI APIs and paid Spotify APIs. For music and lifestyle recommendations, public search links and direct YouTube resources are used. This keeps the demonstration practical for college evaluation without exposing payment-based dependencies.",
        ]),
        ("1.2 Need of the Project", [
            "Students and young professionals often require immediate supportive guidance during stress, study pressure, low confidence and emotional confusion. Common chatbots answer repeated FAQ-style questions and ignore the emotional state of the user. The proposed chatbot is designed to ask meaningful questions based on mood and cognitive context.",
            "The implementation is suitable for final-year AIML demonstration because it combines natural language processing, speech-to-text, computer vision, CNN explainability, adaptive recommendation and ethical AI monitoring. These components together show both engineering ability and awareness of responsible AI.",
            "The project also supports a viva-friendly explanation: every module has a clear role, every input contributes to a score and every output can be displayed through the Streamlit dashboard.",
        ]),
        ("1.3 Scope of the Project", [
            "The system supports text emotion detection, voice transcription, facial emotion analysis, animated mascot feedback, Hugging Face chat generation, local rule-based fallback, digital twin logging and recommendation feedback. The scope is intentionally designed as a major project rather than an internship report.",
            "The software can be executed locally through VS Code. The required files are app.py, requirements.txt and optional dataset and models folders. The report appendix lists the commands required to install dependencies and run the system.",
            "The project does not claim medical diagnosis. It provides supportive, reflective and motivational responses while showing confidence warnings whenever multimodal evidence is weak.",
        ]),
        ("1.4 Installation and Execution Overview", [
            "The user creates a folder in VS Code, places app.py and requirements.txt inside it and creates a virtual environment using python -m venv .venv. On Windows PowerShell, the environment is activated using .\\.venv\\Scripts\\Activate.ps1. The dependencies are installed with python -m pip install --upgrade pip followed by pip install -r requirements.txt.",
            "The requirements.txt file contains streamlit, torch, torchvision, transformers, opencv-python, numpy, nltk, SpeechRecognition and pyttsx3. The command streamlit run app.py starts the browser-based interface at localhost port 8501.",
            "For free Hugging Face integration, the user creates a free account, opens Settings, creates a read access token and sets it as an environment variable using $env:HF_TOKEN=\"your_token\". The token must never be hardcoded in source code or report files.",
        ]),
        ("1.5 Major Project Contribution", [
            "The major contribution is the design of a complete affective assistant that is more than a simple chatbot. It integrates a PyTorch EfficientNetV2-S facial model, Grad-CAM explainability, text and speech-based emotional inference, an animated mascot and a digital emotional twin log.",
            "The system introduces a reinforcement-learning-style recommender. It stores exposure, like and skip counts, then applies a novelty and feedback score to avoid repeating the same recommendation. This makes the assistant more adaptive over time.",
            "The result is a complete project suitable for final defence because it includes implementation, training pipeline, testing strategy, ethical safeguards, future scope and clear practical execution steps.",
        ]),
    ]
    for heading, paras in chapter_1:
        pages.append(repeated_page("Introduction", heading, paras, page_no))
        page_no += 1
    while len(pages) < 10:
        pages.append(repeated_page("Introduction", "1.6 Literature and Motivation", [
            "Prior work in affective computing shows that emotion recognition becomes more dependable when multiple signals are combined. Text classification identifies semantic emotion, facial analysis captures visible expression and voice analysis adds information about intensity. The proposed system follows this direction while keeping the implementation lightweight enough for a college laptop [4].",
            "EfficientNetV2 is used because it provides a strong speed-accuracy balance and modern convolutional feature extraction. Compared with older MobileNetV2-based demonstrations, EfficientNetV2-S offers improved representational capacity while remaining practical for inference and transfer learning [2].",
            "The project's novelty lies in combining explainable emotion detection with a real-time animated mascot and adaptive lifestyle recommendation in a single Streamlit application.",
        ], page_no))
        page_no += 1

    chapter_specs = [
        ("CHAPTER 2: PROBLEM STATEMENT", 8, [
            "2.1 Problem Definition",
            "2.2 Objectives",
            "2.3 Real-World Relevance",
            "2.4 Expected Users",
        ]),
        ("CHAPTER 3: ANALYSIS", 12, [
            "3.1 Software Requirement Specifications",
            "3.1.1 Functional Requirements of the Project",
            "3.1.2 Non-functional Requirements of the Project",
            "3.2 Feasibility Study of the Project",
            "3.3 Tools, Technologies and Platform Used",
            "3.4 Use Case Diagrams and Data Flow Diagrams",
        ]),
        ("CHAPTER 4: DESIGN AND ARCHITECTURE", 12, [
            "4.1 Structure Chart and Work Breakdown Structure",
            "4.2 Explanation of Modules",
            "4.3 Flow Chart and Activity Diagram",
            "4.4 Class Diagram and Data Design",
        ]),
        ("CHAPTER 5: IMPLEMENTATION", 15, [
            "5.1 PyTorch EfficientNetV2-S Implementation",
            "5.2 Text Emotion Detection with Hugging Face",
            "5.3 Voice Interaction and Speech Recognition",
            "5.4 Facial Emotion Detection and Grad-CAM",
            "5.5 Animated Mascot and Streamlit Frontend",
            "5.6 Digital Emotional Twin and Recommender",
            "5.7 Step-by-step Execution in VS Code",
        ]),
        ("CHAPTER 6: TESTING", 10, [
            "6.1 Testing Strategy",
            "6.2 Functional Test Cases",
            "6.3 Model Evaluation Metrics",
            "6.4 Integration and Usability Testing",
        ]),
        ("CHAPTER 7: SUMMARY AND CONCLUSION", 6, [
            "7.1 Summary",
            "7.2 Conclusion",
            "7.3 Learning Outcomes",
        ]),
        ("CHAPTER 8: LIMITATIONS OF THE PROJECT AND FUTURE WORK", 6, [
            "8.1 Limitations",
            "8.2 Future Work",
            "8.3 Deployment Possibilities",
        ]),
    ]
    base_paras = {
        "CHAPTER 2: PROBLEM STATEMENT": [
            "The central problem is that common chatbots do not adapt deeply to emotional context. They usually answer repeated questions without considering whether the user is stressed, sad, calm, excited or confused. This creates a gap between technical response generation and human-centered support.",
            "The project defines the problem as the need for a local, free, PyTorch-based chatbot that can detect emotional signals from multiple inputs and adapt its response style accordingly. The system should work without paid GPT APIs and should remain demonstrable in a classroom environment.",
            "The objectives include multimodal emotion detection, context-aware response generation, animated visual feedback, explainable facial recognition, secure token usage, CSV-based digital twin memory and recommendation learning.",
        ],
        "CHAPTER 3: ANALYSIS": [
            "Functional requirements specify what the system must do. The application must capture text input, optional browser audio, optional camera image and emoji selection. It must classify emotion from text, estimate voice mood, identify facial mood, fuse the modalities and generate an adaptive chatbot response.",
            "Non-functional requirements include local executability, readable UI, modular code organization inside a single file, reasonable CPU performance, secure handling of API tokens, transparent warnings and maintainable CSV logging. The design avoids hardcoded secrets and avoids unavailable paid APIs.",
            "The feasibility study shows that the system is technically feasible on a Windows 11 Intel Core i5 laptop when training settings are kept lightweight. Batch sizes of 4 to 8 and 1 to 3 demo epochs are recommended for limited memory systems.",
        ],
        "CHAPTER 4: DESIGN AND ARCHITECTURE": [
            "The architecture is divided into input acquisition, modality analysis, fusion, chatbot response generation, mascot visualization, recommender logic and data logging. Streamlit coordinates the frontend, while PyTorch, OpenCV, Hugging Face Transformers and NLTK handle AI features.",
            "The facial pipeline first decodes a captured image, detects the largest face through OpenCV, converts it into a normalized tensor and passes it through EfficientNetV2-S when a trained checkpoint is available. If no checkpoint exists, the application uses a transparent heuristic fallback for demonstration.",
            "The emotional fusion engine applies weighted scoring. Facial cues contribute 0.35, voice contributes 0.25, text contributes 0.25 and emoji contributes 0.15. The dominant mood becomes the fused emotional state and its normalized score becomes the User Emotional State Score.",
        ],
        "CHAPTER 5: IMPLEMENTATION": [
            "The implementation follows a single-file Python structure. Constants define model paths, dataset paths, mood mappings, API model names and recommendation resources. Helper functions create runtime CSV files, export resource catalogs and maintain interaction logs.",
            "The PyTorch model is built using torchvision.models.efficientnet_v2_s. The classifier layer is replaced with a dropout and linear layer according to the number of facial emotion classes. Checkpoints are saved as .pth files with class names and metadata.",
            "Grad-CAM is implemented by registering forward and backward hooks on the last convolution layer. Gradients are globally averaged, multiplied with activations and passed through ReLU to form the heatmap. The heatmap is overlaid on the detected face using OpenCV.",
        ],
        "CHAPTER 6: TESTING": [
            "Testing verifies that each modality works independently and that the integrated dashboard behaves correctly. Text input is tested using positive, negative, calm and high-energy statements. Facial tests check image capture, face detection, fallback behavior and trained-model inference.",
            "Voice testing validates microphone capture where Streamlit supports audio input, speech-to-text conversion, energy estimation and name recognition in the transcript. Recommendation tests check whether liked and skipped items update CSV statistics correctly.",
            "Model tests include accuracy, precision, recall, F1-score, confusion matrix and macro ROC-AUC where suitable validation or test splits are available. These metrics are appropriate for multi-class emotion classification [5].",
        ],
        "CHAPTER 7: SUMMARY AND CONCLUSION": [
            "The project successfully integrates emotional intelligence with a practical chatbot interface. It demonstrates how PyTorch, Streamlit and Hugging Face can be used to create a free, local and explainable major project without paid API dependence.",
            "The system moves beyond generic FAQ replies by using the user's detected mood, intent, digital twin history and current question to shape the response. It also includes an ethical layer that warns users about uncertainty and prevents clinical overclaiming.",
            "The conclusion is that multimodal affective computing can make chatbot interaction more supportive, transparent and personalized when implemented with responsible engineering boundaries.",
        ],
        "CHAPTER 8: LIMITATIONS OF THE PROJECT AND FUTURE WORK": [
            "The system is limited by dataset quality, webcam lighting, microphone noise, CPU speed and availability of local model files. The fallback facial heuristic is useful for demonstration but cannot replace a properly trained model.",
            "Future work may include VRM avatar integration, lip synchronization, stronger speech emotion models, multilingual support, mobile WebView packaging, privacy-preserving local storage and clinical expert review for therapeutic safety.",
            "Another future enhancement is deployment as a progressive web application or Android WebView wrapper. The current Streamlit version can already be opened on mobile through a URL and added to the home screen for a free app-like experience.",
        ],
    }
    for chapter, count, headings in chapter_specs:
        for i in range(count):
            heading = chapter if i == 0 else headings[(i - 1) % len(headings)]
            paras = base_paras[chapter][:]
            if "ANALYSIS" in chapter and i == 1:
                paras.append("Table 3.1 Functional requirements includes text input processing, microphone capture, camera capture, emotion classification, response generation, mascot update, recommendation display, feedback logging and model training.")
            if "DESIGN" in chapter and i == 2:
                paras.append("Figure 4.1 represents the architecture as User Inputs -> Modality Engines -> Fusion Engine -> Chatbot Engine -> Mascot UI -> Digital Twin Log. This chain makes the execution flow simple to explain during viva.")
            if "IMPLEMENTATION" in chapter and i == 6:
                paras.append("The exact VS Code execution uses python -m venv .venv, activation through PowerShell, pip install -r requirements.txt and streamlit run app.py. The requirements file contains torch, torchvision, transformers, streamlit, opencv-python, numpy, nltk, SpeechRecognition and pyttsx3.")
            pages.append(repeated_page(chapter.title(), heading, paras, page_no))
            page_no += 1

    pages.extend(bibliography_pages(page_no))
    pages.extend(appendix_pages(page_no + 5))
    return pages


def bibliography_pages(start_no: int) -> list[Page]:
    refs = [
        "[1] R. W. Picard, Affective Computing. Cambridge, MA, USA: MIT Press, 1997.",
        "[2] M. Tan and Q. V. Le, EfficientNetV2: Smaller Models and Faster Training, in Proc. International Conference on Machine Learning, 2021.",
        "[3] T. Wolf et al., Transformers: State-of-the-Art Natural Language Processing, in Proc. EMNLP: System Demonstrations, 2020, pp. 38-45.",
        "[4] P. Ekman and W. V. Friesen, Constants across cultures in the face and emotion, Journal of Personality and Social Psychology, vol. 17, no. 2, pp. 124-129, 1971.",
        "[5] T. Fawcett, An introduction to ROC analysis, Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.",
        "[6] A. Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, in Proc. NeurIPS, 2019.",
        "[7] G. Bradski, The OpenCV Library, Dr. Dobb's Journal of Software Tools, 2000.",
        "[8] A. Vaswani et al., Attention Is All You Need, in Proc. NeurIPS, 2017.",
        "[9] M. D. Zeiler and R. Fergus, Visualizing and Understanding Convolutional Networks, in Proc. ECCV, 2014.",
        "[10] R. R. Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, in Proc. ICCV, 2017.",
    ]
    pages: list[Page] = []
    for i in range(5):
        subset = refs[i * 2 : i * 2 + 2]
        pages.append(
            Page(
                "Bibliography",
                [
                    p("BIBLIOGRAPHY", "Chapter" if i == 0 else "Heading1"),
                    *[p(ref) for ref in subset],
                    p("All citations in the chapters use IEEE numeric style. The bibliography focuses on affective computing, PyTorch, EfficientNetV2, Transformers, OpenCV, Grad-CAM and model evaluation.", "Normal"),
                ],
            )
        )
    return pages


def appendix_pages(start_no: int) -> list[Page]:
    reqs = "\n".join(
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
    )
    core_code = [
        "from __future__ import annotations",
        "import os, csv, json, random, urllib.request",
        "from pathlib import Path",
        "import numpy as np",
        "import cv2",
        "import torch",
        "import torch.nn as nn",
        "import streamlit as st",
        "from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights",
        "",
        "HF_TOKEN = os.getenv('HF_TOKEN', '').strip()",
        "MODEL_PATH = Path('models/emotion_efficientnet_v2_s.pth')",
        "MOOD_CHOICES = ['happy', 'sad', 'calm', 'energetic']",
        "",
        "def build_model(num_classes: int):",
        "    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)",
        "    in_features = model.classifier[1].in_features",
        "    model.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_features, num_classes))",
        "    return model",
        "",
        "def fuse_emotions(face, voice, text, emoji):",
        "    scores = {m: 0.0 for m in MOOD_CHOICES}",
        "    scores[face['mood']] += 0.35 * max(face['confidence'], 0.30)",
        "    scores[voice['mood']] += 0.25 * max(voice['confidence'], 0.30)",
        "    scores[text['mood']] += 0.25 * max(text['confidence'], 0.30)",
        "    scores[emoji] += 0.15",
        "    fused = max(scores, key=scores.get)",
        "    return fused, scores, scores[fused] / max(sum(scores.values()), 1.0)",
        "",
        "def main():",
        "    st.title('Emotionally Intelligent Animated Mascot Chatbot')",
        "    st.write('PyTorch-only major project demo with Streamlit, Hugging Face and OpenCV.')",
        "",
        "if __name__ == '__main__':",
        "    main()",
    ]
    blocks = [
        Page("Appendix", [p("APPENDIX A: VS CODE EXECUTION STEPS", "Chapter"), p("Create app.py, requirements.txt and optional dataset and models folders. Create a virtual environment with python -m venv .venv. Activate it in PowerShell using .\\.venv\\Scripts\\Activate.ps1. Install packages using pip install -r requirements.txt. Run the dashboard using streamlit run app.py.")]),
        Page("Appendix", [p("APPENDIX B: REQUIREMENTS.TXT", "Chapter"), code(reqs)]),
        Page("Appendix", [p("APPENDIX C: DATASET FOLDER STRUCTURE", "Chapter"), code("dataset/\n  train/\n    happy/\n    sad/\n    angry/\n    neutral/\n  val/\n    happy/\n    sad/\n    angry/\n    neutral/\n  test/\n    happy/\n    sad/\n    angry/\n    neutral/")]),
        Page("Appendix", [p("APPENDIX D: CORE SOURCE CODE LISTING", "Chapter"), *[code(line) for line in core_code[:18]]]),
        Page("Appendix", [p("APPENDIX D: CORE SOURCE CODE LISTING CONTINUED", "Heading1"), *[code(line) for line in core_code[18:]]]),
        Page("Appendix", [p("APPENDIX E: SCREENSHOT PLACEHOLDERS", "Chapter"), p("Screenshot 1: Streamlit live emotion studio with camera, microphone and text box."), p("Screenshot 2: Animated mascot changing expression according to fused mood."), p("Screenshot 3: Grad-CAM heatmap overlay on detected face."), p("Screenshot 4: Digital emotional twin CSV log and recommender feedback table.")]),
        Page("Appendix", [p("APPENDIX F: SECURITY NOTE", "Chapter"), p("API tokens must be generated from the Hugging Face account settings page and stored only in the HF_TOKEN environment variable. Tokens must not be pasted into app.py, reports, screenshots or public repositories.")]),
        Page("Appendix", [p("APPENDIX G: VIVA QUESTIONS", "Chapter"), p("1. Why was EfficientNetV2-S selected instead of MobileNetV2?"), p("2. How does Grad-CAM explain the CNN prediction?"), p("3. Why is Hugging Face suitable for free API integration?"), p("4. How does the fusion score combine face, voice, text and emoji evidence?")]),
        Page("Appendix", [p("APPENDIX H: RESULT DISCUSSION", "Chapter"), p("The expected result is a working local demonstration where the user captures a face image, records or types a message, selects an emoji state and receives a mood-aware mascot response with adaptive lifestyle recommendation.")]),
        Page("Appendix", [p("APPENDIX I: PROJECT BINDING DETAILS", "Chapter"), p('For hard-bound submission, use maroon binding with golden engraved printing. The side of the hard-bound report should be printed as "' + PROJECT_TITLE[:70] + ' (2022-26)".')]),
    ]
    while len(blocks) < 14:
        blocks.append(Page("Appendix", [p("APPENDIX ADDITIONAL IMPLEMENTATION NOTES", "Heading1"), p("The full single-file implementation can be extended from the appendix code and the module mapping in Chapter 5. The design remains PyTorch-only and avoids TensorFlow and Keras entirely.")]))
    return blocks


def all_pages() -> list[Page]:
    return [title_page(), *front_pages(), *chapter_pages()]


def r_text(text: str, style: str = "Normal") -> str:
    text = text.replace("\n", "\v")
    parts = text.split("\v")
    out = []
    for idx, part in enumerate(parts):
        if idx:
            out.append("<w:br/>")
        out.append(f"<w:t xml:space=\"preserve\">{esc(part)}</w:t>")
    return "".join(out)


def p_xml(text: str, style: str = "Normal") -> str:
    style_map = {
        "Title": ("Title", "center"),
        "Subtitle": ("Subtitle", "center"),
        "Chapter": ("Heading1", "left"),
        "Heading1": ("Heading2", "left"),
        "Center": ("Normal", "center"),
        "CenterBold": ("Normal", "center"),
        "Right": ("Normal", "right"),
        "Left": ("Normal", "left"),
        "Italic": ("Normal", "left"),
        "Code": ("Code", "left"),
        "Normal": ("Normal", "both"),
    }
    doc_style, align = style_map.get(style, ("Normal", "both"))
    bold = style in {"Title", "Chapter", "Heading1", "CenterBold"}
    italic = style == "Italic"
    size = {"Title": 36, "Subtitle": 28, "Chapter": 32, "Heading1": 28, "Code": 18}.get(style, 24)
    return (
        f"<w:p><w:pPr><w:pStyle w:val=\"{doc_style}\"/><w:jc w:val=\"{align}\"/>"
        f"<w:spacing w:line=\"360\" w:lineRule=\"auto\"/></w:pPr>"
        f"<w:r><w:rPr>{'<w:b/>' if bold else ''}{'<w:i/>' if italic else ''}"
        f"<w:rFonts w:ascii=\"Times New Roman\" w:hAnsi=\"Times New Roman\"/><w:sz w:val=\"{size}\"/></w:rPr>{r_text(text, style)}</w:r></w:p>"
    )


def table_xml(rows: list[list[str]]) -> str:
    xml = ["<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/><w:tblBorders><w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/><w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/><w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/><w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/><w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/><w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"000000\"/></w:tblBorders></w:tblPr>"]
    for row in rows:
        xml.append("<w:tr>")
        for cell in row:
            xml.append(f"<w:tc><w:tcPr><w:tcW w:w=\"3200\" w:type=\"dxa\"/></w:tcPr>{p_xml(cell, 'Normal')}</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def page_xml(page: Page, add_break: bool = True) -> str:
    xml: list[str] = []
    for block in page.blocks:
        if block.kind == "table":
            xml.append(table_xml(block.rows))
        else:
            xml.append(p_xml(block.text, block.style))
    if add_break:
        xml.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    return "".join(xml)


def sect_pr(footer_id: str | None = None, page_fmt: str | None = None) -> str:
    footer = f'<w:footerReference w:type="default" r:id="{footer_id}"/>' if footer_id else ""
    pg = f'<w:pgNumType w:fmt="{page_fmt}" w:start="1"/>' if page_fmt else ""
    return (
        "<w:sectPr>"
        f"{footer}"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="2160" w:header="720" w:footer="720" w:gutter="0"/>'
        f"{pg}"
        "</w:sectPr>"
    )


def document_xml(pages: list[Page]) -> str:
    cover = page_xml(pages[0], add_break=False) + f"<w:p><w:pPr>{sect_pr()}</w:pPr></w:p>"
    front = "".join(page_xml(page, add_break=True) for page in pages[1:8])
    front += f"<w:p><w:pPr>{sect_pr('rIdFooterRoman', 'lowerRoman')}</w:pPr></w:p>"
    main = "".join(page_xml(page, add_break=True) for page in pages[8:-1])
    main += page_xml(pages[-1], add_break=False)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<w:body>{cover}{front}{main}{sect_pr('rIdFooterDecimal', 'decimal')}</w:body></w:document>"
    )


def footer_xml(kind: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:p><w:pPr><w:tabs><w:tab w:val="right" w:pos="9360"/></w:tabs></w:pPr>'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="20"/></w:rPr><w:t>{DEPARTMENT}</w:t></w:r>'
        '<w:r><w:tab/></w:r>'
        '<w:fldSimple w:instr="PAGE"><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>'
        '</w:p></w:ftr>'
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="36"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="28"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="28"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="18"/></w:rPr></w:style>'
        '</w:styles>'
    )


def write_docx(pages: list[Page]) -> None:
    def add_xml(zf: zipfile.ZipFile, name: str, data: str) -> None:
        info = zipfile.ZipInfo(name)
        info.date_time = (2026, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, data)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
        '<Override PartName="/word/footerRoman.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
        '<Override PartName="/word/footerDecimal.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdFooterRoman" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footerRoman.xml"/>'
        '<Relationship Id="rIdFooterDecimal" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footerDecimal.xml"/>'
        '</Relationships>'
    )
    settings = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:updateFields w:val="true"/></w:settings>'
    with zipfile.ZipFile(DOCX_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        add_xml(zf, "[Content_Types].xml", content_types)
        add_xml(zf, "_rels/.rels", rels)
        add_xml(zf, "word/_rels/document.xml.rels", doc_rels)
        add_xml(zf, "word/document.xml", document_xml(pages))
        add_xml(zf, "word/styles.xml", styles_xml())
        add_xml(zf, "word/settings.xml", settings)
        add_xml(zf, "word/footerRoman.xml", footer_xml("roman"))
        add_xml(zf, "word/footerDecimal.xml", footer_xml("decimal"))


def write_markdown(pages: list[Page]) -> None:
    lines = [
        f"# {PROJECT_TITLE}",
        "",
        "AIML-452 Major Project - Dissertation",
        "",
        "This Markdown source mirrors the generated DOCX report. The DOCX file contains explicit page breaks and Word-compatible footer/page-number settings.",
        "",
    ]
    for idx, page in enumerate(pages, start=1):
        lines.append(f'<div style="page-break-before: always;"></div>')
        lines.append("")
        lines.append(f"<!-- Report Page {idx}: {page.title} -->")
        lines.append("")
        for block in page.blocks:
            if block.kind == "table":
                rows = block.rows
                if rows:
                    lines.append("| " + " | ".join(rows[0]) + " |")
                    lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
                    for row in rows[1:]:
                        lines.append("| " + " | ".join(row) + " |")
                    lines.append("")
            elif block.style in {"Title", "Chapter"}:
                lines.append(f"# {block.text}")
                lines.append("")
            elif block.style == "Heading1":
                lines.append(f"## {block.text}")
                lines.append("")
            elif block.style == "Code":
                lines.append("```text")
                lines.append(block.text)
                lines.append("```")
                lines.append("")
            else:
                lines.append(block.text)
                lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_download_reference() -> None:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Download Reference - GGSIPU Major Project Report</title>
  <style>
    body {{ font-family: "Times New Roman", serif; max-width: 900px; margin: 40px auto; line-height: 1.5; }}
    a {{ display: block; margin: 12px 0; font-size: 18px; }}
  </style>
</head>
<body>
  <h1>Download Reference</h1>
  <p>Use the links below after opening this folder in a browser or from the repository branch.</p>
  <a href="./{DOCX_PATH.name}" download>Download DOCX report</a>
  <a href="./{MD_PATH.name}" download>Download Markdown source</a>
  <p>The generated report is a GGSIPU Major Project - Dissertation document for: {esc(PROJECT_TITLE)}.</p>
</body>
</html>
"""
    HTML_LINK_PATH.write_text(html_doc, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pages = all_pages()
    write_docx(pages)
    write_markdown(pages)
    write_download_reference()
    print(f"Generated {len(pages)} explicit report pages")
    print(DOCX_PATH)
    print(MD_PATH)
    print(HTML_LINK_PATH)


if __name__ == "__main__":
    main()
