# Cognitive Emotion Intelligence and Adaptive Lifestyle System: A Multimodal, Explainable, and Ethical AI Framework for Emotion-Aware Major-Project Support

**Author:** Guneesh Dua  
**Program:** B.Tech / 8th Semester Major Project Draft  
**Suggested venue style:** IEEE conference paper format  

## Abstract

Emotion-aware intelligent systems are increasingly relevant in healthcare, education, assistive computing, human-computer interaction, and personalized digital wellness. However, many existing systems remain limited by single-modality inputs, insufficient explainability, weak deployment readiness for low-resource devices, and inadequate ethical monitoring for real-world use. This project presents a practical major-project framework titled *Cognitive Emotion Intelligence and Adaptive Lifestyle System*, designed as a single-file Python and Streamlit implementation that combines multimodal affect inference, adaptive lifestyle guidance, digital emotional twin logging, ethical AI monitoring, lightweight deployment considerations, and explainable deep learning. The proposed system fuses emotion cues from facial images, emoji selection, free-text context, and voice-to-text fallback. For model-based facial analysis, the system supports an EfficientNetV2 transfer-learning pipeline using Global Average Pooling (GAP) and Grad-CAM explainability. To remain compatible with low-resource academic environments, the system also includes heuristic fallbacks and lightweight data preparation modes. The recommendation layer generates no-download music and routine suggestions using structured resource catalogs and optional free or freemium APIs such as Hugging Face inference and Spotify search links. A digital emotional twin log stores structured interaction summaries rather than unnecessary raw personal media, thereby supporting transparency, auditability, and long-term adaptation. This paper surveys recent literature from IEEE, Springer, Elsevier, Wiley, Taylor & Francis, and MDPI; identifies research gaps in unimodal design, limited explainability, poor edge readiness, and weak ethical governance; and presents an integrated architecture intended for final-year engineering implementation. The work contributes a deployment-aware, explainable, and ethically monitored multimodal framework that is suitable for major-project demonstrations, viva presentation, and future extension toward assistive wellness platforms.

**Keywords:** cognitive emotional intelligence, multimodal emotion recognition, explainable AI, Grad-CAM, EfficientNetV2, global average pooling, ethical AI monitoring, digital emotional twin, adaptive lifestyle system, emotion-aware recommendation

---

## I. Introduction

Modern intelligent systems increasingly aim to understand human context rather than respond only to explicit commands. Among the most important context signals is emotion, because human emotional state influences productivity, stress tolerance, decision quality, learning behavior, communication style, and content preference. In digital systems that recommend music, assist with daily routines, support mental well-being, or personalize educational interaction, emotional context can significantly improve relevance and user satisfaction. This has made emotion-aware computing a major area of research across affective computing, multimodal AI, recommender systems, and assistive technology.

Conventional personalization systems often rely on historical user interactions such as clicks, ratings, consumption history, and session duration. In music recommendation, for example, collaborative filtering and content-based filtering remain dominant. While these methods are effective for large-scale personalization, they usually cannot capture real-time emotional variation. A user may want different support when stressed before a viva, calm music after an exhausting workday, or motivation during a coding sprint. Systems that ignore present emotional state often fail to provide context-aware adaptation, especially for new users or changing situations.

At the same time, emotion recognition has advanced through computer vision, speech processing, natural language processing, and multimodal fusion. Recent research has shown that combining multiple affect cues generally improves robustness compared with unimodal systems. Facial expression cues provide rich visual information; text captures explicit and implicit self-reported feeling; speech and vocal tone reflect intensity and urgency; and self-report signals such as emoji or quick user input can provide lightweight calibration. These advances have enabled a new class of emotionally adaptive systems that go beyond passive prediction and actively adjust recommendations, prompts, routines, and interfaces.

Despite this progress, important technical and practical gaps remain. First, many systems still depend on only one modality, which reduces robustness in ambiguous or noisy real-world conditions. Second, high-performing deep models are often difficult to interpret, creating trust, accountability, and deployment concerns in sensitive applications. Third, many research prototypes are not designed for low-resource hardware, although student projects and real-world users frequently depend on modest laptops or mobile devices. Fourth, the ethical consequences of continuous emotion inference are substantial: data privacy, bias, informed consent, low-confidence predictions, and misuse risk must be addressed rather than treated as afterthoughts.

These limitations are especially important in final-year major-project settings. A technically strong project must not only deliver code, but also demonstrate novelty, explainability, practical feasibility, academic grounding, and clarity during viva examination. The project presented in this paper therefore aims to bridge research and implementation by creating an integrated, single-source Python prototype that supports multimodal emotion fusion, adaptive lifestyle planning, explainable CNN-based image emotion inference, free or low-cost API paths, digital emotional twin logging, and ethical AI monitoring.

The proposed project is titled **Cognitive Emotion Intelligence and Adaptive Lifestyle System**. The central idea is that a system should not merely classify emotion, but use emotional state as one input among many for adaptive support. The term *cognitive emotion intelligence* in this work refers to the ability of the system to collect emotional indicators, reason over them, present interpretable results, and translate the outputs into actionable adaptive behavior. The term *adaptive lifestyle system* refers to the project’s role in recommending routines, reflection prompts, question generation, study support, and mood-aware content pathways rather than stopping at raw classification. The concept of a *digital emotional twin* refers to a structured, privacy-conscious behavioral memory that stores summarized user state, context, outputs, and feedback so that the system can become more consistent and auditable over time.

This work is motivated by several research and practical questions:

1. How can a final-year major project integrate multimodal emotion understanding and adaptive support without requiring a complex multi-service architecture?
2. How can explainability be included in a project that uses CNN-based visual emotion recognition?
3. How can the system remain usable on low-resource Windows hardware?
4. How can ethical AI concerns be surfaced at the interface level rather than buried in documentation?
5. How can the resulting system be made suitable for both viva presentation and future extension?

To answer these questions, this paper presents a system that combines:

- multimodal emotion estimation from face image, emoji, text, and voice transcript fallback,
- EfficientNetV2 transfer learning with Global Average Pooling,
- Grad-CAM explainability,
- free local fallback execution paths,
- optional Hugging Face and Spotify integrations,
- history-aware adaptive recommendation,
- digital emotional twin CSV logging,
- ethical AI monitoring checkpoints,
- dataset preparation and lightweight training support,
- and documentation for VS Code execution and academic presentation.

The contributions of this paper are summarized as follows:

1. A practical multimodal affective computing architecture suitable for a single-file Python major project.
2. A deployment-aware explainable CNN pipeline using EfficientNetV2 and Global Average Pooling in place of older MobileNetV2-centered baselines.
3. An ethically monitored adaptation framework using confidence checks, consent-aware logging, and dataset-balance warnings.
4. A structured digital emotional twin mechanism for personalized but auditable longitudinal adaptation.
5. A complete project package including code, setup guidance, viva preparation support, and an IEEE-style paper draft.

The remainder of the paper is organized as follows. Section II reviews the most relevant literature from 2022–2026 across major publishers. Section III identifies the research problem and gaps addressed by the proposed system. Section IV describes the complete methodology and architecture. Section V discusses implementation details and project execution flow. Section VI presents experimental and analytical result discussion suitable for a major-project report. Section VII gives conclusion and future scope. Finally, the reference section lists the cited works in IEEE style.

---

## II. Literature Review

Recent literature shows a strong movement from unimodal emotion recognition toward multimodal, explainable, and context-aware affective systems. However, these directions are often studied separately rather than in one integrated deployable prototype. The review below highlights representative papers across the publishers required for this project and connects them to the present system design.

Assuncao *et al.* [1] conducted a systematic literature review on music recommendation systems that incorporate emotions and contextual factors. Their study analyzed 64 publications and concluded that combining context and emotion generally improves listening experience, but open problems remain in user activity modeling, satisfaction feedback, cold-start handling, and cognitive load. This review is important because it establishes that emotion-aware recommendation is promising, yet still incomplete in terms of user-centered deployment and adaptive feedback loops.

Melchiorre *et al.* [2] proposed **Emotion-aware Music Tower Blocks (EmoMTB)**, an intelligent audiovisual interface for music discovery and recommendation. The work combined large-scale music exploration with emotion-aware recommendation and showed the value of nonlinear browsing and emotion-driven re-ranking. A major strength of the work is its integration of recommendation logic and interface design. However, its emotional input is more music-centric than person-centric, and it does not emphasize personal multimodal emotion fusion from face, text, and self-report in the way required for a student support or lifestyle platform.

Dong *et al.* [3] introduced **EmoAda**, a multimodal emotion interaction and psychological adaptation system. This work is relevant because it explicitly connects multimodal perception with personalized emotional support and adaptation. The paper demonstrates the practical relevance of systems that do not stop at recognition but attempt supportive conversational action. However, it is oriented toward broader emotional support dialogue and does not focus on lightweight student-project deployment, explainable CNN-based face modeling, or digital twin logging.

Salas-Caceres *et al.* [4] presented a multimodal emotion recognition system based on audiovisual fusion with temporal dynamics. Their architecture integrated audio and visual information and used LSTM-based temporal modeling, achieving strong results on RAVDESS, SAVEE, and CREMA-D. This work confirms the value of multimodal temporal modeling and public benchmark evaluation. Its limitation for the present project is that it assumes richer audiovisual pipelines and more specialized data preparation than what is practical for a low-resource, single-file major project.

Ramaswamy and Palaniswamy [5] reviewed multimodal emotion recognition trends and challenges in a Wiley survey. Their analysis emphasized feature extraction, fusion methods, and the movement toward more robust multimodal pipelines. The paper also highlighted persistent issues such as cross-domain robustness, data quality, and model complexity. This supports the argument that multimodality is valuable, but practical lightweight implementation remains an open challenge.

Kaur and Kumar [6] provided a comprehensive review of facial emotion recognition, covering traditional machine learning, deep learning, datasets, and applications. Their review showed the broad importance of FER across healthcare, HCI, and behavioral analysis, while also pointing to robustness issues under real-world conditions. This literature underlines the need for visual emotion recognition in the current project but also justifies adding fallback modalities and explainability.

Ammous *et al.* [7] designed an efficient CNN-based emotion recognition system and showed that careful preprocessing and improved training strategy can enhance FER performance. This is useful from a practical student-project perspective because it demonstrates that system quality depends not only on network selection but also on preprocessing, augmentation, and resource-aware design. However, the paper is centered on image classification rather than complete lifestyle adaptation or ethical monitoring.

Hamzah and Abdalla [8] surveyed EEG-based emotion recognition datasets for virtual environments. Although EEG is outside the minimal implementation of the present project, this review is important because it shows how rich affective signals can improve recognition quality while also increasing acquisition complexity. It supports a major design choice in this project: prioritizing low-friction modalities first, while leaving physiological inputs as future scope.

Chandraumakantham *et al.* [9] proposed an LLM-based feature-fusion approach for multimodal emotion recognition in IEEE Access. Their work reflects a newer direction in which large language model reasoning is brought into multimodal inference. The paper is relevant because it suggests that language-aware reasoning can improve fusion quality. Still, many such systems remain computationally heavy or depend on cloud-scale inference, which is not always suitable for low-end local deployment.

Tian *et al.* [10] proposed **MMREC**, an LLM-based multimodal recommender system. This work is relevant not because it is specifically an emotion-recognition paper, but because it shows how large language model reasoning is being integrated into recommendation systems that operate across modalities. It supports the idea that adaptive systems can use richer context than static collaborative filtering. However, it is not aimed at low-resource or student-major-project settings.

Belharbi *et al.* [11] proposed guided interpretable facial expression recognition using spatial action-unit cues. This paper is particularly important to the present work because it emphasizes interpretable FER rather than raw accuracy alone. It shows that meaningful localization of important facial regions can improve trust and analysis quality. The current project adopts the same spirit through Grad-CAM overlays for CNN predictions.

Yalcin and Alisawi [12] introduced a new FER dataset and demonstrated that preprocessing can substantially improve deep-learning performance. Their work supports the importance of dataset quality and preprocessing strategy in visual affect modeling. The present project reflects this by including a dataset preparation step, sample-mode operation, and class-distribution logging.

Kim and Hong [13] developed an emotion-oriented recommender system for personalized indoor environmental quality control. Although the domain is different, the work shows that emotion-aware recommendation is useful well beyond entertainment. This strengthens the justification for using emotional state as a control signal for adaptive action in the proposed lifestyle-support system.

Fan *et al.* [14] proposed a gradient-based lightweight network automated design method for facial expression recognition. This work is highly relevant to the present decision to move away from older backbones and toward stronger efficiency-oriented CNN design. It also shows that lightweight FER remains a major active research area rather than a solved problem.

Mattioli and Cabitza [15] reviewed the challenges and ethical considerations of automatic face emotion recognition technology. Their paper is central to this project because it argues that FER systems may suffer from questionable psychological assumptions, cultural bias, annotation unreliability, privacy harms, and misuse risk. The current project explicitly responds by treating emotion outputs as advisory, surfacing confidence warnings, and requiring consent-aware logging.

Barker *et al.* [16] further examined ethical considerations in emotion recognition research. Their work emphasizes privacy, bias, transparency, informed consent, and participant control. The present project operationalizes these concerns through an ethical AI monitoring panel rather than leaving them only in theoretical discussion.

Shehada *et al.* [17] proposed an explainable, privacy-preserving federated facial emotion recognition framework for mental health monitoring. Their work is particularly important because it combines lightweight CNN design, privacy preservation, and quantitative explainability evaluation. The reported results on RAF-DB, ExpW, and FER2013 show that trustworthy FER requires both strong performance and interpretable behavior. The current project does not implement full federated learning, but it is directly inspired by the idea of pairing lightweight CNNs with explainability and trust.

Shehada *et al.* [18] also introduced a human-centered quantitative explainability framework using SHAP and a Global Explanation Quality Score in a federated FER setting. This work underscores that explanation quality should be assessed seriously rather than visually admired. The present project uses Grad-CAM as a simpler student-project mechanism, but the paper’s insight motivates the inclusion of explainability as a first-class component.

Wu *et al.* [19] presented a comprehensive review of multimodal emotion recognition techniques, challenges, and future directions. Their review identifies lightweight models, cross-corpus generalizability, explainability, and richer modal fusion as key future research needs. This paper strongly supports the main research gap addressed in the present project: the need for integrated, explainable, practical MER prototypes.

Schmitz-Hubsch *et al.* [20] discussed emotional valence, arousal, and task demand in adaptive performance-oriented systems. Though not a classical FER system, the work is relevant because it links emotion and task state to adaptive system design. This informs the proposed project’s lifestyle-planning and stress-aware support layer.

Wyman and Zhang [21] provided a tutorial on AI tools for facial emotion recognition in R. While tool-centric, this paper is valuable because it shows the increasing demand for practical, reproducible emotion-analysis workflows. The current project similarly attempts to provide a usable, execution-ready academic prototype rather than only a conceptual model.

Varheenmaa *et al.* [22] studied facial emotion recognition in children with ADHD, showing that affect recognition can vary by application population and cognitive context. This reinforces the caution that emotion-recognition systems must not be treated as universally valid across populations or high-stakes settings.

Taken together, the literature indicates four dominant trends. First, multimodal emotion recognition is more robust than unimodal recognition. Second, explainability is now seen as essential rather than optional. Third, practical deployment on real hardware remains difficult despite algorithmic advances. Fourth, ethical concerns are no longer peripheral; they directly affect whether emotion-aware systems should be trusted or deployed at all. These observations motivate the integrated architecture proposed in this paper.

---

## III. Problem Statement and Research Gap

### A. Problem Statement

The problem addressed in this project is the lack of a practical, explainable, and ethically monitored multimodal emotional intelligence system that can be implemented as a student-major-project prototype on modest hardware while still demonstrating academic seriousness. Existing systems frequently suffer from one or more of the following limitations:

1. dependence on a single modality,
2. weak explainability,
3. lack of real-time adaptive support,
4. poor suitability for low-resource devices,
5. no structured emotional memory or user-state tracking,
6. inadequate ethical safeguards,
7. excessive implementation complexity for academic deployment.

As a result, students often face a gap between literature-level innovation and project-level feasibility. They may be able to reproduce a classifier, but not an integrated system that supports recommendation, adaptation, explainability, and responsible use.

### B. Research Gaps in 2023–2025 Work

From the reviewed literature, the main research gaps relevant to this project are:

**Gap 1: Unimodal dependence.**  
Many projects still rely only on text or only on facial cues. This reduces robustness when a user does not provide one signal clearly or when the signal is noisy.

**Gap 2: Explainability remains shallow.**  
Although many FER works mention explainability, student implementations often skip it entirely or show only isolated visualizations without integrating them into system trust and reporting.

**Gap 3: Deployment mismatch.**  
Published models may perform well on servers or benchmark setups, but major-project implementations often run on low-end laptops. This creates a gap between paper-level performance and actual usability.

**Gap 4: Missing adaptation loop.**  
Many systems stop at emotion classification and do not translate outputs into helpful actions such as study support, lifestyle adjustment, question prompting, or recommendation management.

**Gap 5: Weak ethical governance.**  
Literature increasingly highlights fairness, privacy, and reliability issues, yet many practical projects still log user data without clear consent handling or confidence warnings.

**Gap 6: No digital emotional memory.**  
Several adaptive systems personalize outputs, but few student projects include a simple, auditable longitudinal structure that can capture emotional state, context, outputs, and user feedback over time.

### C. How the Proposed Project Addresses the Gaps

The current project addresses these gaps through:

- multimodal fusion across face image, emoji, text, and voice transcript,
- EfficientNetV2 + GAP as a stronger yet practical CNN backbone,
- Grad-CAM visual explanation,
- local heuristic fallbacks when TensorFlow or large models are unavailable,
- recommendation and adaptive routine generation instead of raw classification alone,
- digital emotional twin CSV logging,
- consent-aware storage and ethical AI reporting,
- free API pathways and non-mandatory external integrations,
- a single-file architecture compatible with VS Code and classroom demonstration.

In this sense, the proposed work is not just a classifier. Its protagonist is the **integrated adaptive loop**: perceive -> fuse -> explain -> adapt -> log -> learn from feedback.

---

## IV. Proposed Methodology

### A. System Overview

The proposed system is organized into six major modules:

1. **Multimodal input collection**
2. **Emotion estimation and fusion**
3. **Adaptive support and recommendation**
4. **Explainability**
5. **Digital Emotional Twin logging**
6. **Ethical AI monitoring**

The system is implemented in a single Python source file (`app.py`) using Streamlit for the interface. This design simplifies setup and presentation while still keeping the internal logic modular through functions.

### B. Input Modalities

The project supports the following inputs:

1. **Face image**  
   A user uploads a face image. If a trained TensorFlow model is available, the system uses a CNN-based prediction. Otherwise, a lightweight heuristic image analysis fallback is used.

2. **Emoji input**  
   Emoji selection acts as explicit low-friction self-report. This is useful in low-resource or quick-interaction contexts.

3. **Free-text context**  
   The user describes the current situation, stress, mood, and goals in natural language. A lexical heuristic estimates emotional orientation.

4. **Voice-to-text fallback**  
   Instead of live speech recognition, a transcript can be pasted. This avoids unnecessary API requirements while preserving the idea of speech-context input.

These signals are not treated equally by default. The user can adjust modality weights in the Streamlit sidebar, allowing experimentation and viva discussion around fusion behavior.

### C. Emotion Estimation

#### 1) Text and voice estimation

The text and voice transcript branches use a keyword-based lexical scoring mechanism. Words associated with happiness, sadness, anger, neutrality, calmness, surprise, and fear are assigned base contributions. The overlap between input tokens and emotion lexicons increases the score for corresponding categories. The resulting scores are normalized into a probability-like distribution.

This approach is intentionally lightweight. It is not intended to replace a state-of-the-art transformer classifier, but to provide a free, local, easily explainable path that runs on low-end systems.

#### 2) Emoji estimation

Emoji-based estimation is a structured mapping from selected emoji to dominant emotion categories. This branch is not sophisticated, but it has two important advantages:

- it supports rapid self-report,
- and it can correct or calibrate uncertain model-based inference.

#### 3) Face-image estimation

The visual branch has two modes:

**Mode A: Trained model available**  
If TensorFlow is installed and a saved model exists, the system loads an EfficientNetV2-based classifier and produces class probabilities.

**Mode B: Fallback mode**  
If the trained model is unavailable, the system estimates emotion heuristically from grayscale brightness, contrast, regional means, and left-right symmetry. While this is not a true FER model, it ensures that the project remains runnable in demonstration environments even without a trained network.

### D. EfficientNetV2 with Global Average Pooling

The user specifically requested that the project move away from MobileNetV2 and use a newer CNN approach. The present implementation therefore uses **EfficientNetV2B0** with **Global Average Pooling (GAP)**.

The reasons are:

1. EfficientNetV2 is newer and generally offers stronger efficiency-accuracy trade-offs than older MobileNetV2 baselines.
2. GAP reduces the number of trainable parameters after convolutional feature extraction.
3. GAP reduces overfitting risk compared with very large fully connected classifier heads.
4. GAP helps preserve spatial relevance for visual explanation methods such as Grad-CAM.
5. The architecture remains compact enough for sampled educational datasets and low-resource experiments.

The training head in the current project is:

- EfficientNetV2B0 backbone without top layer,
- preprocessing using EfficientNetV2 input normalization,
- GlobalAveragePooling2D,
- dropout,
- dense hidden layer,
- softmax output over emotion classes.

### E. Dataset Preparation

To keep the project free and reproducible, the app includes a dataset-preparation mode using public Hugging Face datasets. The dataset pipeline attempts to:

- download a public emotion-image dataset,
- detect image and label columns,
- sample a manageable number of images,
- normalize label names,
- split the data into `train`, `val`, and `test`,
- save a `dataset_manifest.json` file with summary metadata.

This design reflects an important practical reality of student projects: full-scale training on large public datasets is often infeasible on 4 GB RAM systems. Therefore, the project explicitly recommends **sample mode** with about 600–1200 images, batch sizes of 4 or 8, and 1–3 epochs for proof-of-concept training.

### F. Multimodal Fusion

Each modality produces a normalized score distribution. Let the modality outputs be:

- \(S_f\) for face image,
- \(S_e\) for emoji,
- \(S_t\) for free text,
- \(S_v\) for voice transcript.

Let the corresponding user-controlled weights be:

- \(w_f, w_e, w_t, w_v\).

The combined score for emotion \(k\) is:

\[
F_k = \frac{w_f S_{f,k} + w_e S_{e,k} + w_t S_{t,k} + w_v S_{v,k}}{\sum w}
\]

followed by normalization over all emotion classes.

This weighted fusion has several advantages for a major project:

- it is easy to explain in viva,
- it exposes the effect of user trust in each modality,
- it supports ablation-style demonstrations,
- and it remains computationally cheap.

### G. Adaptive Lifestyle and Recommendation Layer

The project extends beyond recognition by generating:

1. **Adaptive lifestyle plans**  
   These are short actionable suggestions based on detected emotion, stress, energy, and sleep.

2. **Mood-aware recommendation candidates**  
   The system includes a structured catalog of more than 100 entries that link mood and goals to YouTube, Spotify, and YouTube Music search routes, plus offline fallback actions.

3. **Mindset-aware question generation**  
   The system can generate important questions based on user mood and project stage. Local templates are provided, and optional Hugging Face inference may be used to create richer coaching prompts.

4. **History-aware no-repetition logic**  
   Previously recommended items are penalized, reducing repetition and simulating lightweight reinforcement-learning logic.

### H. Reinforcement-Learning Logic

The user requested reinforcement-learning logic. In a strict research sense, the current project does not implement a heavy deep RL agent. Instead, it uses a **reinforcement-inspired adaptive reward loop**:

- recommendations are generated,
- user feedback is stored as helpful / neutral / needs-improvement,
- prior items are penalized in future ranking,
- novelty and relevance are balanced over time.

This is appropriate for a student major project because it preserves the conceptual idea of learning from feedback without introducing unnecessary training instability or infrastructure complexity.

### I. Digital Emotional Twin

The digital emotional twin is one of the strongest differentiators of the project. Instead of storing all raw personal media, the system records structured summaries such as:

- timestamp,
- user ID,
- fused emotion,
- confidence,
- current goal,
- energy,
- stress,
- sleep,
- active modalities,
- score distribution.

Recommendation feedback is logged separately. This creates a lightweight but auditable emotional memory. Over time, this memory can support personalization, trend analysis, transparency, and safer adaptation.

### J. Ethical AI Monitoring

The project includes an explicit ethical AI monitoring panel. It checks:

- whether user consent for logging is active,
- whether only one modality is being used,
- whether prediction confidence is low,
- whether the dataset manifest suggests severe class imbalance,
- whether explainability is available,
- and whether outputs should be treated as advisory rather than authoritative.

This design is directly inspired by the ethical FER literature and turns abstract concerns into visible system behavior.

### K. Explainable AI with Grad-CAM

For trained CNN-based face inference, the system implements Grad-CAM. The last convolutional layer is identified, gradients are computed with respect to the predicted class, and a heatmap is created to highlight the regions that influenced the decision. The heatmap is resized and blended with the original image.

Explainability has three roles in this project:

1. it helps users and evaluators understand what the model attended to,
2. it supports debugging and model trust,
3. it strengthens the viva narrative around responsible AI.

### L. Free API Strategy

The user requested free execution and asked about OpenAI and Spotify alternatives. The project therefore follows this strategy:

- **No OpenAI dependency** is required.
- **Hugging Face** is used as the primary optional free inference path because users can create free accounts and access hosted inference depending on availability and quotas.
- **Spotify** integration is optional and uses free developer credentials where available; if not provided, the app falls back to direct search links.
- All core features remain available without any paid API.

This is why Hugging Face is a suitable recommendation: it offers a practical free entry point for experimentation, model hosting, and token-based inference, whereas fully unrestricted large-model API access is rarely free at production scale.

---

## V. Implementation Flow and Execution

### A. Folder Structure

The project uses a minimal structure:

- `app.py` - single compatible code file
- `requirements.txt` - essential dependencies
- `README.md` - project overview
- `docs/` - execution guide and viva support
- `paper/` - research paper source and literature matrix
- `dataset/` - generated sampled dataset
- `models/` - saved trained model and metadata

### B. VS Code Execution Steps

1. Install Python 3.11 or 3.12.
2. Open the folder in VS Code.
3. Create a virtual environment.
4. Activate it.
5. Run `pip install -r requirements.txt`.
6. Run `streamlit run app.py`.
7. Use sampled dataset mode before training.
8. Export the catalog if required using `python app.py --export-catalog`.

### C. Why the Project Uses a Single-File Design

The single-file design satisfies the user’s requirement for a compact, VS Code-compatible project. It is especially helpful in viva settings because:

- there is one primary execution entry,
- reviewers can inspect the entire logic in one file,
- setup complexity is reduced,
- deployment and demonstration are easier.

### D. APK Conversion Trick

Because Streamlit is web-based, the most practical free route to an installable mobile experience is:

1. deploy the app,
2. open it in Chrome,
3. use “Add to Home Screen” or PWABuilder packaging.

This gives a PWA-like installable experience without maintaining a separate native front end.

---

## VI. Results and Discussion

### A. Expected Outputs

The project is designed to produce the following outputs:

1. multimodal emotion score distribution,
2. dominant emotion label and confidence,
3. adaptive lifestyle plan,
4. mood-aware recommendation list,
5. digital emotional twin log entry,
6. training metrics after CNN training,
7. confusion matrix,
8. weighted precision, recall, and F1,
9. ROC-AUC where class count and prediction format allow,
10. Grad-CAM overlay for visual explainability.

### B. Evaluation Metrics

The project computes:

- **accuracy** for overall correctness,
- **precision** for predictive purity,
- **recall** for sensitivity,
- **F1-score** for balanced performance,
- **confusion matrix** for class-wise analysis,
- **ROC-AUC (one-vs-rest)** when multiclass probabilities are available.

These metrics align with common FER evaluation practice and are appropriate for major-project reporting.

### C. Discussion of the Proposed System

The proposed system’s most important strength is not a single benchmark number but the integration of multiple academically relevant components. Compared with a plain FER classifier, it provides a richer demonstration environment. A student can explain:

- how emotion is inferred from multiple signals,
- how the system adapts outputs,
- how explainability is generated,
- how data is logged ethically,
- how free APIs and local fallbacks coexist,
- and how the project addresses real-world usability.

This is a stronger academic story than a narrow classifier demo.

### D. Comparison with Earlier Methods

Compared with older single-modality projects:

- the current system uses multiple modalities,
- includes a recommendation and adaptation layer,
- surfaces ethics and explainability,
- replaces MobileNetV2 with EfficientNetV2 + GAP,
- and provides deployment guidance for low-resource hardware.

Compared with larger research systems:

- it is lighter,
- simpler to run,
- easier to explain in viva,
- and more suitable for a final-year engineering project.

### E. Limitations

No project should overclaim its capabilities. The present system has important limitations:

1. Local text emotion analysis is heuristic, not state-of-the-art transformer classification.
2. The fallback visual path is not a true FER model.
3. Dataset quality depends on public mirrors and sampled subsets.
4. The recommendation engine uses structured ranking logic rather than a fully trained RL recommender.
5. The system is not a medical diagnostic platform and must not be presented as one.
6. Patentability and publication cannot be guaranteed.

These limitations should be openly acknowledged during viva and in the final report.

---

## VII. Conclusion and Future Scope

This paper presented a complete major-project framework for a **Cognitive Emotion Intelligence and Adaptive Lifestyle System** implemented as a single-file Python application. The project combines multimodal emotion fusion, EfficientNetV2-based explainable visual inference, adaptive lifestyle planning, optional free API integration, digital emotional twin logging, and ethical AI monitoring. The literature review showed that recent research has advanced multimodal affective computing, explainability, and adaptive recommendation, but gaps remain in integration, edge readiness, transparent deployment, and practical academic implementation. The proposed work addresses these gaps by offering a balanced architecture that is both academically grounded and practically executable on student hardware. The project’s strongest distinguishing feature is its protagonist architecture: a fused loop of perception, explanation, adaptation, and structured emotional memory. Rather than acting as only a classifier, the system behaves as a lightweight emotionally adaptive assistant for major-project support. This makes it suitable for demonstration, report writing, viva discussion, and future research extension.

### Future Scope

1. Add real speech emotion recognition and on-device audio feature extraction.
2. Extend the recommendation layer into a formal reinforcement-learning policy trained from longitudinal feedback.
3. Introduce federated learning and fairness-aware evaluation for stronger privacy and trustworthiness.

---

## References

[1] W. G. Assuncao, L. S. G. Piccolo, and L. A. M. Zaina, “Considering emotions and contextual factors in music recommendation: a systematic literature review,” *Multimedia Tools and Applications*, vol. 81, pp. 8367–8407, 2022.

[2] A. B. Melchiorre, D. Penz, C. Ganhor, O. Lesota, V. Fragoso, F. Fritzl, E. Parada-Cabaleiro, F. Schubert, and M. Schedl, “Emotion-aware music tower blocks (EmoMTB): an intelligent audiovisual interface for music discovery and recommendation,” *International Journal of Multimedia Information Retrieval*, vol. 12, art. 13, 2023.

[3] T. Dong, F. Liu, X. Wang, Y. Jiang, X. Zhang, and X. Sun, “EmoAda: A multimodal emotion interaction and psychological adaptation system,” in *Proc. MultiMedia Modeling (MMM)*, 2024, pp. 301–307.

[4] J. Salas-Caceres, J. Lorenzo-Navarro, D. Freire-Obregon, and M. Castrillon-Santana, “Multimodal emotion recognition based on a fusion of audiovisual information with temporal dynamics,” *Multimedia Tools and Applications*, vol. 84, pp. 27327–27343, 2025.

[5] M. P. A. Ramaswamy and S. Palaniswamy, “Multimodal emotion recognition: A comprehensive review, trends, and challenges,” *WIREs Data Mining and Knowledge Discovery*, vol. 14, no. 6, 2024.

[6] M. Kaur and M. Kumar, “Facial emotion recognition: A comprehensive review,” *Expert Systems*, vol. 41, no. 10, 2024.

[7] D. Ammous, A. Chabbouh, A. Edhib, A. Chaari, F. Kammoun, and N. Masmoudi, “Designing an efficient system for emotion recognition using CNN,” *Journal of Electrical and Computer Engineering*, 2023, Art. no. 9351345.

[8] H. A. Hamzah and K. K. Abdalla, “EEG-based emotion recognition datasets for virtual environments: A survey,” *Applied Computational Intelligence and Soft Computing*, 2024, Art. no. 6091523.

[9] O. Chandraumakantham, N. Gowtham, M. Zakariah, and A. Almazyad, “Multimodal emotion recognition using feature fusion: An LLM-based approach,” *IEEE Access*, vol. 12, pp. 108052–108067, 2024.

[10] J. Tian, J. Zhao, Z. Wang, and Z. Ding, “MMREC: LLM based multi-modal recommender system,” in *Proc. 19th Int. Workshop Semantic and Social Media Adaptation & Personalization (SMAP)*, 2024.

[11] S. Belharbi, M. Pedersoli, A. L. Koerich, S. Bacon, and E. Granger, “Guided interpretable facial expression recognition via spatial action unit cues,” in *Proc. IEEE Int. Conf. Automatic Face and Gesture Recognition (FG)*, 2024.

[12] N. Yalcin and M. Alisawi, “Introducing a novel dataset for facial emotion recognition and demonstrating significant enhancements in deep learning performance through pre-processing techniques,” *Heliyon*, vol. 10, no. 20, 2024, Art. no. e38913.

[13] H. Kim and T. Hong, “Emotion-oriented recommender system for personalized control of indoor environmental quality,” *Building and Environment*, vol. 254, 2024.

[14] J. Fan, S. Deng, X. Song, J. Liu, and Y. Sun, “A gradient-based lightweight network automated design method for facial expression recognition,” *Expert Systems with Applications*, vol. 296, 2026.

[15] M. Mattioli and F. Cabitza, “Not in my face: Challenges and ethical considerations in automatic face emotion recognition technology,” *Machine Learning and Knowledge Extraction*, vol. 6, no. 4, pp. 2201–2231, 2024.

[16] D. Barker, M. K. R. Tippireddy, A. Farhan, and B. Ahmed, “Ethical considerations in emotion recognition research,” *Psychology International*, vol. 7, no. 2, 2025, Art. no. 43.

[17] D. Shehada, H. Tawfik, A. Bouridane, and A. Hussain, “An explainable framework for mental health monitoring using lightweight and privacy-preserving federated facial emotion recognition,” *Sensors*, vol. 25, no. 23, 2025, Art. no. 7320.

[18] D. Shehada, H. Tawfik, A. Bouridane, and A. Hussain, “Human-centered and quantitative explainability evaluation of facial emotion recognition for trustworthy mental health monitoring,” *Computers*, vol. 15, no. 3, 2026, Art. no. 139.

[19] Y. Wu, Q. Mi, and T. Gao, “A comprehensive review of multimodal emotion recognition: Techniques, challenges, and future directions,” *Biomimetics*, vol. 10, no. 7, 2025, Art. no. 418.

[20] A. Schmitz-Hubsch, M. Gruber, Y. Diaz, M. Wirzberger, and P. Hancock, “Towards enhanced performance: an integrated framework of emotional valence, arousal, and task demand,” *Ergonomics*, vol. 67, no. 12, pp. 2082–2095, 2024.

[21] A. Wyman and Z. Zhang, “A tutorial on the use of artificial intelligence tools for facial emotion recognition in R,” *Multivariate Behavioral Research*, vol. 60, pp. 641–655, 2025.

[22] M. Varheenmaa, S. M. Lehto, P. Rizzo, H.-C. Steinhausen, R. Drechsler, and A.-K. Brem, “Facial emotion recognition in children with attention deficit hyperactivity disorder,” *Nordic Journal of Psychiatry*, 2024.
