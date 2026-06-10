# Cognitive Emotion Intelligence & Adaptive Lifestyle System

## IEEE-Style Research Paper Draft

Author: Student major-project draft  
Program: B.Tech Final Year Major Project  
Suggested format: Times New Roman, 10 pt, two-column IEEE conference layout when exported to Word or LaTeX.

---

## Title

**Cognitive Emotion Intelligence and Adaptive Lifestyle System Using Multimodal Emotion Fusion, Explainable CNNs, Digital Emotional Twin Logging, and Free AI Integration**

---

## Abstract

Emotion-aware intelligent systems are becoming increasingly important in digital wellbeing, adaptive interfaces, lifestyle assistance, and human-centered AI. Conventional recommendation and lifestyle systems usually rely on historical interactions, static preferences, and generic rule-based advice, which limits their ability to respond to the user’s real-time mindset and emotional context. This project presents a single-file Streamlit-based system titled **Cognitive Emotion Intelligence and Adaptive Lifestyle System**, designed to capture multimodal user emotion from facial image input, text context, emoji prior, and voice-to-text input, and then transform the inferred state into adaptive, explainable, and ethically monitored support. The proposed system integrates a lightweight transfer-learning convolutional neural network based on EfficientNetV2 with Global Average Pooling for facial emotion recognition, optional Grad-CAM visualization for explainable AI, a Digital Emotional Twin logging layer for longitudinal emotional state modeling, a history-aware no-repetition recommender for music and lifestyle resources, and a free AI integration pathway using Hugging Face for chatbot and speech services.

The system is specifically designed for practical execution on a Windows 11 laptop with Intel Core i5 processor and 8 GB RAM, making it suitable for final-year major-project deployment. To remain accessible and low-cost, all major features degrade gracefully when heavy frameworks or cloud tokens are unavailable. The architecture therefore supports both advanced mode with CNN training and cloud inference, and fallback mode using local heuristics and static knowledge assistance. Experimental design includes dataset sampling for low-memory training, evaluation through precision, recall, F1-score, confusion matrix, and ROC-AUC where possible, and interpretability using Grad-CAM overlays. The project addresses research gaps observed in 2023-2025 literature by combining multimodal emotion recognition, transparent AI, long-term user modeling, adaptive recommendation logic, and free conversational support in one executable program. The result is a practical, academically defensible, and deployment-oriented emotional intelligence prototype suitable for real-world wellness, study-support, and adaptive lifestyle scenarios.

**Keywords**: cognitive emotional intelligence, multimodal emotion recognition, explainable AI, Grad-CAM, EfficientNetV2, digital emotional twin, adaptive lifestyle system, emotion-aware recommendation, Streamlit, Hugging Face

---

## I. Introduction

Music and lifestyle recommendation systems have become an integral part of digital platforms, wellness interfaces, online education tools, and personalized productivity applications. Modern users interact with large digital ecosystems in which content selection, information retrieval, emotional support, and daily routine guidance are increasingly mediated by intelligent systems. Traditional recommendation mechanisms have been dominated by collaborative filtering, content-based filtering, and hybrid recommendation approaches. While these techniques have improved personalization, they generally remain dependent on historical interaction patterns such as clicks, ratings, and playback history. Such systems often ignore one of the most important contextual variables in human decision-making: the emotional and cognitive state of the user at the time of interaction.

Human behavior is strongly influenced by emotional context. A student preparing for an examination may require focused and calming support when stressed, motivational suggestions when fatigued, and exploratory content when curious and energized. Similarly, a user searching for music, guidance, or conversational assistance may not benefit from generic recommendations if the system does not understand whether the person is anxious, frustrated, lonely, happy, or emotionally neutral. Recent advances in artificial intelligence, natural language processing, computer vision, and speech analysis have opened the possibility of building emotion-aware systems that respond not only to historical preferences but also to present-time emotional conditions.

In parallel, research in facial emotion recognition has evolved from conventional machine learning models toward deep convolutional neural networks, transfer learning, and transformer-based methods. These advances have improved performance but have also introduced new concerns regarding computational expense, opacity, fairness, and deployment feasibility on commodity devices. At the same time, recommendation research has moved beyond static ranking toward sequential decision-making and reinforcement-learning-inspired personalization. However, these streams of research are still frequently developed in isolation. Emotion recognition work often ends at classification, recommender work often ignores affective context, and conversational AI systems often lack explicit multimodal grounding or user-governed historical memory.

This gap motivates the present work. The proposed project, **Cognitive Emotion Intelligence and Adaptive Lifestyle System**, is designed as an integrated prototype that combines multimodal emotion sensing, explainable deep learning, adaptive recommendation logic, long-term user-state tracking, and conversational interaction in a single executable Streamlit application. The system accepts a facial image, emoji mood prior, free-text context, and optional voice input. It fuses these sources into a unified emotional state representation, produces adaptive recommendations, logs longitudinal user patterns in a Digital Emotional Twin CSV store, and allows users to ask domain-relevant questions through a project-aware chatbot. A free API integration route is provided using Hugging Face tokens rather than relying exclusively on paid platforms such as OpenAI or Spotify.

The problem statement addressed by this work can be summarized as follows: **current emotion-aware lifestyle or recommendation systems often lack one or more of the following characteristics simultaneously: multimodal fusion, explainability, deployability on low-resource hardware, long-term user modeling, ethical override and uncertainty handling, and free or low-cost AI integration**. Many earlier systems are unimodal, computationally heavy, research-only, or disconnected from a realistic end-user interface. Others produce emotion classifications without actionable adaptive guidance. Some provide personalized responses but do not offer transparent explanations or user consent mechanisms.

This project aims to overcome these limitations by providing a unified implementation suitable for a Windows 11 laptop and VS Code-based execution workflow. Instead of building a separate frontend and backend, the entire system is implemented as a single Python source file that remains easy to understand, present, and demonstrate during project evaluation. The selected CNN architecture uses EfficientNetV2 with Global Average Pooling rather than older MobileNetV2 so that the model remains lightweight while improving feature quality and compatibility with Grad-CAM explainability.

The major contributions of this project are:

1. A single-file multimodal emotion-aware lifestyle support application in Python and Streamlit.
2. A facial emotion module based on EfficientNetV2 and Global Average Pooling, with optional local training and Grad-CAM explainability.
3. A multimodal fusion layer combining facial image, emoji, free text, and voice transcript.
4. A Digital Emotional Twin logging system for longitudinal emotional-state tracking.
5. A history-aware recommender with no-repetition bias and simple bandit-style reward update logic.
6. A free AI integration route using Hugging Face tokens for chat and speech.
7. Ethical AI monitoring through uncertainty warnings, user override, consent-based logging, and non-diagnostic framing.

The remainder of this paper is organized as follows. Section II presents a detailed literature review and analysis of research gaps. Section III describes the proposed methodology and system architecture. Section IV explains the implementation and experimental workflow. Section V presents results and discussion. Section VI concludes the work and identifies future scope. Section VII lists references in IEEE style.

---

## II. Literature Review

Recent studies have explored different dimensions of emotion-aware intelligent systems including multimodal affect recognition, explainable facial emotion recognition, personalized recommendation, reinforcement-learning-based adaptation, conversational intelligence, and ethical AI governance. However, integrated deployment-oriented solutions remain relatively rare.

Kim and Hong proposed an emotion-oriented recommender system for personalized indoor environmental quality control using a graph-attention-based architecture and emotional similarity estimation. Their results showed that affective information can significantly improve personalization compared with traditional static recommenders. However, the work was confined to a domain-specific environmental setting and relied on a limited private dataset rather than a general lifestyle support interface [1].

Abakarim, Qassimi, and Rakrak presented an emotion and sentiment enriched decision transformer for personalized recommendations. Their work is highly relevant because it demonstrates that emotional signals extracted from user text can improve long-horizon recommendation quality. The study reported gains in nDCG, hit ratio, and RMSE on Yelp and Google Local Review datasets. Even so, the approach depends heavily on review-text quality and does not integrate multimodal inputs such as facial expressions or voice [2].

Zhou et al. addressed distribution shift in offline reinforcement-learning-based recommender systems using a Q-learning regularized decision transformer. Their results indicate that sequential recommendation can be made more robust through regularization, particularly when only logged offline data is available. Although the study is important for adaptive recommendation, it does not explicitly include emotional state as a decision variable [3].

Messaoudi, Boughrara, and Lachiri studied multimodal emotion recognition by integrating speech and text for valence, arousal, and dominance prediction. Their work confirms that multimodal fusion outperforms unimodal analysis and supports the use of continuous affective dimensions instead of only fixed emotion labels. However, the system focused primarily on benchmark prediction rather than interactive lifestyle adaptation [4].

Fang et al. explored multimodal speech emotion recognition using large language model based text features combined with acoustic signals. The paper demonstrated that LLM-derived linguistic embeddings can enhance emotion recognition, particularly in speech-related tasks. Nevertheless, the work did not extend its outputs to downstream recommendation or personal assistance modules [5].

Zhang et al. investigated empathetic conversational recommender systems, arguing that emotional alignment and response generation should be treated jointly. This work is closely aligned with the conversational support goal of the present project. However, it is benchmark-centered and focused on recommendation dialogue rather than multimodal, wellbeing-oriented support on resource-limited devices [6].

Meyer and Elsweiler examined LLM-based conversational agents for behavior change support in a randomized controlled trial. Their findings are valuable for this project because they show that conversational agents can influence readiness to change when designed with supportive and safe prompting. Yet the model does not explicitly integrate multimodal sensed emotion or user-side visual explainability [7].

Gebele et al. analyzed CNN-based facial emotion recognition through Grad-CAM and discussed the interpretive implications of explainability maps. Their work highlights a major point relevant to this project: explainability is useful for transparency and debugging, but saliency alone should not be misrepresented as absolute evidence of "true" emotional understanding. This observation strongly motivates the present system's ethical AI warnings and user-override design [8].

Shah et al. combined Xception-based feature extraction with multiple explainability methods on the RAVDESS dataset and reported high classification accuracy along with rich interpretability. The contribution of this study lies in demonstrating that explainable emotion recognition can be both accurate and presentable. However, it uses an acted dataset and does not address low-cost deployment constraints or free API integration [9].

Punuri et al. proposed an EfficientNet-XGBoost facial emotion recognition pipeline using transfer learning. Their findings support the use of lightweight transfer-learning backbones for facial affect tasks. At the same time, the results reveal that strong benchmark performance on curated datasets does not automatically translate into robust in-the-wild deployment [10].

Khomidov and Lee combined EfficientNet-based facial recognition with heart-rate-variability signals for complex human emotion prediction. Their results are especially relevant because they show that multimodal fusion improves accuracy significantly compared with face-only recognition. This supports the central premise of the present project: cognitive emotional intelligence should emerge from multiple complementary signals rather than a single modality [11].

Ramirez-Quintana et al. designed a lightweight CNN with efficient channel attention for real-time embedded facial emotion recognition. Their work is important because it emphasizes deployability and reduced computational cost, which are critical for student projects running on everyday laptops. Still, the model remains primarily a classifier and does not expand into adaptive recommendation or long-term digital twin modeling [12].

Katirai reviewed ethical considerations in emotion recognition technologies and summarized the literature on privacy, fairness, interpretive harms, and governance concerns. This review provides a strong foundation for the present project’s ethical AI monitoring layer. It suggests that affective systems should avoid overclaiming, protect user data, and include meaningful human control mechanisms [13].

Kim formalized the concept of Affective Sovereignty and argued for sovereign-by-design architectures that enable users to contest, override, and govern affective inferences. This notion is highly aligned with the present project’s override selector, consent-based logging, and reflective rather than diagnostic positioning [14].

Zhang et al. proposed a digital twin framework for type 2 diabetes using machine learning, knowledge graphs, and longitudinal data. Although the domain differs, the architectural lesson is directly applicable: a meaningful intelligent system should maintain a persistent representation of the user across time rather than responding only to isolated events [15].

Kiran et al. further explored digital twins using retrospective lifestyle data and predictive modeling. Their work shows that lifestyle, psychosocial, and behavioral variables can be integrated into longitudinal simulation-oriented systems. This supports the Digital Emotional Twin concept used in the current project, where recurring emotional patterns and user corrections are stored for adaptive support [16].

Chen et al. introduced PersonaTwin, a prompt-conditioning framework for generating personalized digital twins using large language models. The study is important because it demonstrates that user-specific modeling can be integrated with conversational AI in a principled manner. However, it requires richer structured data and does not provide a low-cost student-project style deployment pathway [17].

The literature therefore reveals a consistent pattern: significant progress exists in individual components, but integration is lacking. Emotion recognition systems often stop at prediction. Recommender systems often ignore real-time affective state. Conversational systems often lack multimodal grounding. Ethical principles are frequently discussed but rarely translated into interface-level controls. This integrated gap directly motivates the proposed system.

---

## III. Problem Statement and Research Gaps

Based on the review of recent literature, the following research gaps are identified:

1. **Unimodal limitation**: Many projects rely only on text, only on face, or only on speech, reducing robustness in real-world use.
2. **Lack of explainability**: Several systems produce predictions without offering interpretable evidence to users or evaluators.
3. **Limited deployment practicality**: State-of-the-art models may be too heavy for low-resource devices common in student and educational settings.
4. **Weak longitudinal memory**: Many systems do not maintain historical emotional context or adaptive user-state tracking.
5. **Insufficient ethical controls**: Earlier systems often lack user override, consent-based logging, or confidence-aware warnings.
6. **High API-cost dependence**: Some prototypes rely on paid APIs, limiting reproducibility and practical student adoption.
7. **Disconnected pipelines**: Facial emotion recognition, recommendation, chatbot assistance, and dataset preparation are frequently implemented as separate research prototypes rather than one user-facing workflow.

Accordingly, the central research problem addressed in this project is:

> How can a low-cost, single-file Python application integrate multimodal affect recognition, explainable CNN inference, digital twin style user modeling, adaptive recommendation, and free AI services in a deployable manner suitable for a standard Windows 11 student laptop?

---

## IV. Proposed Work

### A. System Overview

The proposed system is a single-file Python Streamlit application that accepts multimodal user input and transforms it into adaptive lifestyle support. The design is intentionally implementation-oriented so that the system can be demonstrated in VS Code without separate backend and frontend services.

### B. Input Modalities

1. **Face image input** through file upload or webcam capture  
2. **Emoji prior** representing an explicit quick mood signal  
3. **Free-text context** written by the user  
4. **Voice input** converted to text using Hugging Face ASR when a token is available  

### C. Emotion Fusion Strategy

Each modality generates an emotion-probability distribution across unified labels:

- angry  
- disgust  
- fear  
- happy  
- neutral  
- sad  
- surprise  

The outputs are normalized and fused using weighted confidence-aware aggregation. Modalities with higher reliability and confidence contribute more strongly to the final decision. The final output includes:

- fused emotion  
- confidence  
- valence  
- arousal  
- stress estimate  
- ethical warning flags  

### D. CNN and Explainability Design

The facial emotion model uses **EfficientNetV2B0** as a transfer-learning backbone. This choice is motivated by:

- improved efficiency-quality trade-off over older lightweight models,
- compatibility with CPU-friendly deployment,
- suitability for transfer learning on small sampled datasets,
- strong compatibility with **Global Average Pooling** and **Grad-CAM**.

The classifier head contains:

1. input rescaling  
2. EfficientNetV2 backbone  
3. Global Average Pooling  
4. dropout  
5. dense softmax classification layer  

Grad-CAM is applied to the final convolutional feature representation to generate an interpretable heatmap overlay.

### E. Digital Emotional Twin

The Digital Emotional Twin stores interaction-level summaries in CSV format, including:

- user hash  
- modality outputs  
- fused emotion  
- final emotion after user override  
- valence, arousal, and stress  
- context excerpt  
- consent status  

This historical store supports personalized reflection and pattern tracking without requiring a database server.

### F. Adaptive Recommendation Logic

The recommender maps the current emotional state to mood buckets and resource candidates. It uses:

- a structured 100+ item catalog,
- simple reward updates based on helpful/skip feedback,
- repeat-avoidance penalties,
- exploration bonus for unseen items.

Although not a full online reinforcement learner, this design introduces practical RL-style adaptive logic appropriate for a student major project.

### G. Free AI Integration

Instead of requiring paid OpenAI or Spotify APIs, the system supports:

- free Hugging Face account token for chatbot responses,
- free Hugging Face ASR for speech transcription,
- optional free image-classification inference if a public face model is used.

This decision improves project accessibility and reproducibility.

### H. Ethical AI Monitoring

The proposed system includes a lightweight ethical governance layer:

- warns when only one modality is active,
- warns when confidence is low,
- warns when top emotion probabilities are too close,
- allows manual override,
- supports consent-based logging,
- avoids presenting predictions as clinical truth.

### I. Flowchart

```text
User Login / Session Key
        |
        v
Collect Inputs -> Face | Emoji | Text | Voice
        |
        v
Per-Modality Emotion Analysis
        |
        v
Confidence-Aware Fusion
        |
        +--> Ethical AI Monitoring
        |
        +--> Digital Emotional Twin Logging
        |
        +--> Adaptive Resource Recommendation
        |
        +--> Avatar Chatbot / Viva Assistant
        |
        v
Explainable Output + User Override + Feedback
```

---

## V. Implementation Details

### A. Software Environment

- Python 3.11 or 3.12  
- Streamlit  
- NumPy, Pandas, Matplotlib, Seaborn  
- Pillow  
- scikit-learn  
- huggingface_hub  
- datasets  
- optional TensorFlow CPU for local CNN training and Grad-CAM  

### B. Folder Structure

```text
project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- dataset/
|   |-- train/
|   |-- val/
|   |-- test/
|   `-- dataset_manifest.json
|-- models/
|   |-- efficientnetv2_emotion.keras
|   `-- efficientnetv2_emotion_metadata.json
|-- cei_twin_log.csv
|-- recommender_stats.csv
|-- resource_catalog.csv
`-- docs/
    |-- VS_CODE_WINDOWS_SETUP.md
    |-- VIVA_GUIDE.md
    |-- APK_CONVERSION_GUIDE.md
    |-- RESEARCH_PAPER_IEEE.md
    `-- LITERATURE_MATRIX.md
```

### C. Resource-Constrained Strategy

The target deployment device is a Windows 11 laptop with Intel Core i5 and 8 GB RAM. Therefore:

- dataset sample size is limited to 600-1200 images,
- batch size remains 4 or 8,
- epochs remain 1-3 for demo training,
- CPU mode is preferred,
- heavy cloud reliance is optional.

### D. Single-File Design Justification

The single-file design reduces integration complexity, simplifies presentation in viva, and makes execution easier for evaluators. While large production systems usually separate frontend and backend services, a major-project context benefits from readability, portability, and demonstrability.

---

## VI. Results and Discussion

### A. Expected Evaluation Metrics

When the local CNN is trained, the system reports:

- precision  
- recall  
- F1-score  
- confusion matrix  
- ROC-AUC where possible  

These metrics are generated on the test split prepared under `dataset/test`.

### B. Functional Outputs

The system demonstrates the following outputs:

1. multimodal emotion prediction,
2. per-modality confidence distribution,
3. ethical AI warnings,
4. Grad-CAM heatmap overlay,
5. digital emotional twin updates,
6. adaptive recommendations with feedback memory,
7. chatbot responses customized to the user's current mindset.

### C. Discussion

The key strength of the proposed system is integration. Instead of claiming state-of-the-art benchmark accuracy in isolation, it demonstrates how multimodal sensing, explainable CNNs, longitudinal memory, and adaptive assistance can operate within one executable interface. This makes it suitable for major-project evaluation, real demonstration, and future expansion.

The second important contribution is practical deployment awareness. The system is designed around CPU-friendly configuration, sample-based training, optional dependencies, and free API support. This directly addresses the gap between research prototypes and classroom-ready deployment.

The third contribution lies in responsible design. By including override, confidence warnings, and consent-based logging, the project avoids the common mistake of portraying emotion AI as fully objective or diagnostic. This is academically valuable because it shows technical ability together with ethical understanding.

### D. Limitations

1. Automatic dataset download depends on public dataset schema compatibility.  
2. Facial analysis quality without a trained model or remote inference falls back to heuristics.  
3. Real-time voice interaction depends on free API availability and network quality.  
4. The recommender uses simplified RL-style logic rather than a full policy-learning pipeline.  
5. Streamlit WebView wrapping is not equivalent to a fully native Android application.  

---

## VII. Conclusion and Future Scope

This paper presented a practical implementation of a **Cognitive Emotion Intelligence and Adaptive Lifestyle System** built as a single-file Streamlit application. The system integrates multimodal emotion fusion, an EfficientNetV2-based explainable facial emotion module, digital emotional twin logging, adaptive recommendation logic, and free Hugging Face integrations for conversational and speech support. The project addresses major research and implementation gaps observed in 2023-2025 emotion-aware systems by combining explainability, user override, low-resource deployment, and longitudinal user modeling in one executable workflow. Its design makes it suitable for academic demonstration, real student use, and future expansion toward more advanced assistive intelligence.

Future scope includes:

1. integrating physiological data such as heart rate, GSR, or wearable signals,  
2. replacing heuristic adaptation with full contextual reinforcement learning,  
3. adding multilingual support, richer avatars, and mobile-optimized on-device models.  

---

## VIII. References

[1] H. Kim and T. Hong, "Emotion-oriented recommender system for personalized control of indoor environmental quality," *Building and Environment*, vol. 254, p. 111396, 2024, doi: 10.1016/j.buildenv.2024.111396.

[2] S. Abakarim, S. Qassimi, and S. Rakrak, "Emotion and sentiment enriched decision transformer for personalized recommendations," *Scientific Reports*, vol. 15, p. 21020, 2025, doi: 10.1038/s41598-025-06386-y.

[3] Y. Zhou *et al.*, "Mitigating distribution shift in offline RL-based recommender systems with a Q-learning regularization decision transformer," *Information*, vol. 17, no. 4, p. 364, 2026, doi: 10.3390/info17040364.

[4] A. Messaoudi, H. Boughrara, and Z. Lachiri, "Multimodal emotion recognition: integrating speech and text for improved valence, arousal, and dominance prediction," *Annals of Telecommunications*, vol. 80, pp. 401-415, 2025, doi: 10.1007/s12243-025-01069-1.

[5] C. Fang *et al.*, "Multimodal speech emotion recognition based on large language model," *IEICE Transactions on Information and Systems*, vol. E107-D, no. 11, pp. 1463-1466, 2024, doi: 10.1587/transinf.2024EDL8034.

[6] X. Zhang *et al.*, "Towards empathetic conversational recommender systems," in *Proc. 18th ACM Conf. Recommender Systems (RecSys)*, 2024, pp. 84-93, doi: 10.1145/3640457.3688133.

[7] S. Meyer and D. Elsweiler, "LLM-based conversational agents for behaviour change support: A randomised controlled trial examining efficacy, safety, and the role of user behaviour," *International Journal of Human-Computer Studies*, vol. 200, p. 103514, 2025, doi: 10.1016/j.ijhcs.2025.103514.

[8] J. Gebele, P. Brune, F. Schwab, and S. von Mammen, "Interpreting emotions through the Grad-CAM lens: Insights and implications in CNN-based facial emotion recognition," in *Pattern Recognition (ICPR 2024)*, LNCS 15313, Springer, 2025, pp. 414-429, doi: 10.1007/978-3-031-78201-5_27.

[9] S. T. H. Shah *et al.*, "Explainable emotion recognition using Xception-based feature extraction and supervised machine learning on the RAVDESS dataset," in *IEEE Int. Symp. Medical Measurements and Applications (MeMeA)*, 2025, pp. 1-6, doi: 10.1109/MeMeA65319.2025.11068008.

[10] S. B. Punuri *et al.*, "Efficient Net-XGBoost: An implementation for facial emotion recognition using transfer learning," *Mathematics*, vol. 11, no. 3, p. 776, 2023, doi: 10.3390/math11030776.

[11] M. Khomidov and J.-H. Lee, "The novel EfficientNet architecture-based system and algorithm to predict complex human emotions," *Algorithms*, vol. 17, no. 7, p. 285, 2024, doi: 10.3390/a17070285.

[12] J. A. Ramirez-Quintana *et al.*, "Lightweight convolutional neural network with efficient channel attention mechanism for real-time facial emotion recognition in embedded systems," *Sensors*, vol. 25, no. 23, p. 7264, 2025, doi: 10.3390/s25237264.

[13] A. Katirai, "Ethical considerations in emotion recognition technologies: a review of the literature," *AI and Ethics*, vol. 4, pp. 927-948, 2024, doi: 10.1007/s43681-023-00307-3.

[14] R. S. Kim, "Formal and computational foundations for implementing affective sovereignty in emotion AI systems," *Discover Artificial Intelligence*, vol. 6, p. 235, 2026, doi: 10.1007/s44163-026-01000-0.

[15] Y. Zhang *et al.*, "A framework towards digital twins for type 2 diabetes," *Frontiers in Digital Health*, vol. 6, 2024, Art. no. 1336050, doi: 10.3389/fdgth.2024.1336050.

[16] M. Kiran *et al.*, "A digital twin framework for predicting and simulating type 2 diabetes onset using retrospective lifestyle data," *Frontiers in Digital Health*, vol. 8, 2026, Art. no. 1710829, doi: 10.3389/fdgth.2026.1710829.

[17] S. Chen, J. P. Lalor, Y. Yang, and A. Abbasi, "PersonaTwin: A multi-tier prompt conditioning framework for generating and evaluating personalized digital twins," in *Proc. Fourth Workshop on Generation, Evaluation and Metrics (GEM2)*, ACL, 2025, pp. 774-788.

---

## How to expand this draft into a 50-page submission

To turn this file into a longer vertical-format project report:

1. expand each subsection into 2-4 pages,
2. add screenshots from the Streamlit app,
3. add equations for weighted fusion and reward update,
4. include a detailed system architecture diagram,
5. include algorithm pseudocode for dataset preparation, emotion fusion, Grad-CAM, and recommendation,
6. append dataset screenshots, output screenshots, and CSV samples,
7. add a comparative table versus earlier 2023-2025 systems.

This repository also includes `docs/LITERATURE_MATRIX.md`, which contains the paper-by-paper extraction table requested for the literature review process.
