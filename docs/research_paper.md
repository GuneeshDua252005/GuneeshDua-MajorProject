# Cognitive Emotion Intelligence and Adaptive Lifestyle System: A Multimodal, Explainable, and Ethical Framework for Adaptive Lifestyle Support

## Abstract

Emotion-aware digital systems are increasingly expected to move beyond static prediction toward adaptive, transparent, and human-centered support. This paper presents a single-source-code Python framework titled **Cognitive Emotion Intelligence and Adaptive Lifestyle System**, designed as a major project that integrates multimodal mood understanding, explainable AI, adaptive recommendation logic, and Digital Emotional Twin logging within a deployable Streamlit application. The proposed system accepts free-text context, emoji self-report, optional voice-note transcription, and face-image input, then fuses these modalities through a confidence-aware engine to infer the user's dominant emotional state. A lightweight custom convolutional neural network with Global Average Pooling is adopted for image-based emotion recognition so that Grad-CAM explanations can be generated with lower parameter complexity than heavier transfer-learning pipelines. The application also includes ethical AI monitoring prompts, CSV-based longitudinal mood logging, and a reward-driven recommendation loop that personalizes supportive resources over time without relying on a complex backend. A review of eighteen recent papers from IEEE, Springer, Elsevier, Wiley, Taylor & Francis, and MDPI is conducted to identify research gaps in multimodality, explainability, deployment simplicity, and ethical governance. Prototype-level results show stable emotion inference in representative anxious, focused, and tired user scenarios, while the generated resource catalog exceeds one hundred structured recommendations. The system is suitable for low-cost academic deployment and demonstrates how emotion intelligence, explainable vision models, and adaptive lifestyle guidance can be combined into a practical and extensible research artifact.

**Keywords:** affective computing, emotion-aware recommendation, Grad-CAM, Global Average Pooling, Digital Emotional Twin, ethical AI, multimodal fusion, Streamlit

## I. Introduction

### A. Overview

Music recommendation systems and emotionally adaptive digital assistants have become central to modern human-computer interaction. However, most deployed recommendation pipelines still rely on historical behavior, click logs, static preferences, or shallow content matching. These systems offer personalization, but they often fail to respond to a user's present emotional condition, short-term stress, academic workload, or need for immediate self-regulation. The rise of affective computing, multimodal emotion recognition, and explainable AI has made it possible to design systems that are not merely reactive to past data, but adaptive to current human context.

The present major-project work is motivated by a practical academic need: to design a single compatible Python program that goes beyond a simple classifier and behaves like an adaptive support engine. Instead of recommending only music or only motivational content, the system is intended to interpret the user's mindset and mood and then produce reflective questions, emotionally aligned action suggestions, and context-aware digital resources through a deployable interface.

### B. Problem Statement

The major problem addressed in this work is the absence of a simple, single-application system that can infer a user's emotional state from multiple low-cost modalities and translate that state into explainable, adaptive lifestyle guidance. Existing emotion-aware systems often suffer from one or more of the following limitations: dependence on only one modality, weak explainability, lack of ethical safeguards, non-portable implementations, and overreliance on expensive or paid external APIs. For academic major projects, an additional problem is that many published ideas are difficult to reproduce on low-resource student hardware or require separate frontend-backend architectures that complicate execution and evaluation.

Traditional recommendation systems based on collaborative filtering or content-based filtering are effective for preference modeling, but they do not fully capture a user's current emotional state. A student who usually prefers energetic playlists may still need calming or focus-oriented support during examination stress, sleep deprivation, or emotional fatigue. Therefore, there is a need for a system that treats emotion as a live context signal rather than a secondary or optional feature.

### C. Applications

The proposed work has applications in student well-being support, adaptive study guidance, emotion-aware music or resource recommendation, lightweight therapy-support interfaces, digital wellness dashboards, and human-centric assistive systems. It is especially relevant for academic settings where users may experience stress, anxiety, fatigue, distraction, or fluctuating motivation and need a low-cost decision-support interface rather than a full clinical system.

In practical terms, the system can be used during a major-project submission cycle, viva preparation, extended coding sessions, or emotional burnout periods. It can also serve as a prototype for future use in employee support dashboards, mental wellness advisory interfaces, or smart study companions.

### D. Existing Challenges

Real-time emotion recognition is difficult because emotional signals are noisy, subjective, and context-dependent. Text can be ambiguous or sarcastic, facial expressions may be subtle or partially visible, speech input may be unavailable, and multimodal fusion can increase computational cost. Moreover, emotion AI systems raise concerns regarding privacy, fairness, consent, and overclaiming. For low-end devices, the challenge is to achieve acceptable intelligence without large compute requirements.

Another difficulty is deployment realism. Many academic papers present strong accuracy values under controlled datasets but do not translate those architectures into tools that ordinary students can install, run, and explain in VS Code on modest hardware. The project therefore must solve both algorithmic and engineering problems.

### E. Existing Solutions

Recent literature shows progress through CNNs for facial emotion recognition, transformer models for contextual text emotion analysis, graph-attention fusion for multimodal conversation understanding, and affect-aware recommenders that integrate emotional semantics into ranking. At the same time, explainability techniques such as Grad-CAM and human-centered XAI studies have emphasized the need to show why a prediction was made. These developments motivate a practical design that borrows the strengths of modern research while remaining executable on an ordinary student machine.

Existing emotion-aware music systems have mostly used one of three strategies: image-based emotion mapping, text sentiment analysis, or hybrid recommendation logic. However, these solutions often stop at one-step recommendation and do not close the loop with reflective questioning, ethical risk alerts, Digital Emotional Twin logging, or feedback-based adaptation.

### F. Paper Structure

The remainder of this paper is organized as follows. Section II presents the literature review and research-gap analysis based on eighteen recent publications. Section III describes the proposed methodology, multimodal pipeline, CNN plus GAP architecture, explainability strategy, digital twin logging, and free API integration path. Section IV discusses prototype results and comparative analysis. Finally, Section V concludes the study and outlines future research directions.

## II. Literature Review

Srivastava *et al.* [1] presented **Emotify: An AI-Powered Emotion-Based Music Recommendation System**. The study used a CNN-based facial emotion detection stage connected to a music recommendation layer. Its main strength lies in demonstrating a practical end-to-end image-driven recommendation pipeline. However, the system is primarily image-centric and does not deeply integrate other low-cost user modalities such as text or emoji self-report.

Chikaraddi *et al.* [2] proposed an **Emotion-Driven Music Recommender System with Deep Learning and Streamlit Integration**. The paper is particularly relevant because it shows that emotion-aware recommendation can be wrapped inside a lightweight Python dashboard rather than a complex web stack. Still, the work remains closely tied to a narrow input path and is focused more on music output than broader lifestyle support.

Talaghat *et al.* [3] studied a **content-based music recommender system based on music emotion using deep learning**. Their contribution is important because it models emotional characteristics of content items rather than relying purely on user history. Yet the study concentrates on music-side content understanding and not on a wider human-state adaptation loop.

Pan *et al.* [4] published a major survey on **multimodal emotion recognition from datasets, preprocessing, features, and fusion methods**. The paper provides a strong taxonomy of datasets, preprocessing strategies, unimodal feature extraction approaches, and fusion mechanisms. It is especially valuable for system designers because it explains why multimodality generally outperforms unimodal emotion analysis. The limitation is that it is a survey and does not present a directly deployable end-user application.

Zhu *et al.* [5] reviewed **key technologies for emotion analysis using multimodal information**. The authors highlighted challenges involving modality synchronization, missing modalities, computational efficiency, and cross-domain adaptation. Their findings justify the use of a confidence-aware fusion pipeline in the present work. However, the paper does not directly solve the low-resource deployment problem faced in academic projects.

Ramaswamy and Palaniswamy [6] offered a comprehensive review of **multimodal emotion recognition, trends, and challenges**. The review strongly emphasizes the long-term importance of lightweight models, generalization, and explainability. These observations are closely aligned with the philosophy of the present system, which prefers an interpretable compact CNN plus user-facing explanation over an opaque but heavier architecture.

Gou and Li [7] explored **emotion analysis of dialogue by integrating BERT embeddings and BiLSTM**. Their approach improves contextual emotion understanding compared with shallow linguistic pipelines. It provides a strong theoretical basis for using transformer-based language models as an optional upgrade for text emotion inference. The main drawback is the higher computational and inference overhead relative to simpler on-device heuristics.

Du and Zhu [8] proposed **self-learning multimodal emotion recognition based on multi-scale dilated attention**. The study reported strong performance on FER2013, CK+, and DEAP benchmarks and demonstrated the advantage of adaptive fusion between facial and physiological signals. Although technically impressive, the architecture is considerably heavier than what many student laptops can train or even execute smoothly.

Wu *et al.* [9] introduced **MLGAT**, a multi-layer graph attention network for multimodal emotion recognition in conversations. The method improves robustness by modeling relations across modalities and utterances rather than using flat fusion alone. This work highlights an advanced future direction for the current project. Its complexity, however, makes it less suitable for a single-file, low-cost prototype.

Liu *et al.* [10] examined **fine-grained interpretability for EEG emotion recognition** using Grad-CAM and a systematic brain functional network. This paper is highly relevant to explainability. It shows that affective computing systems benefit when model evidence can be visualized and audited. Even though the domain is EEG rather than face images, the underlying explainability principle directly motivates the use of Grad-CAM in the present project.

Kaur and Kumar [11] offered a comprehensive review of **facial emotion recognition**. The paper summarizes the evolution from classical handcrafted features to modern CNN-based pipelines and discusses challenges in real-world uncontrolled settings. This review supports image-based emotion analysis as one modality of the proposed system, but also makes clear that face analysis alone is insufficient for robust human-state understanding.

Katirai [12] reviewed the **ethical considerations in emotion recognition technologies**. The study identified concerns related to bias, privacy, data sensitivity, and harmful misuse in consequential settings. This paper is one of the strongest motivations for embedding ethical monitoring warnings inside the current system. A key takeaway is that emotionally intelligent systems must be accompanied by governance logic rather than only model outputs.

Davila-Gonzalez and Martin [13] discussed a **Human Digital Twin in Industry 5.0** supported by emotional analytics. The concept is valuable because it frames emotional data not as isolated snapshots but as part of a longitudinal human-state representation. The present project adapts that idea into a lightweight, privacy-conscious CSV-based **Digital Emotional Twin** suitable for academic and personal-support use.

Liu *et al.* [14] proposed a **sentimentally enhanced conversation recommender system**. Their work showed that emotional semantics can improve recommendation quality beyond plain conversational matching. This supports the project's choice to place emotional context inside the recommendation pipeline rather than treating it as a cosmetic output label.

Yang *et al.* [15] explored **emotion-enhanced dual-agent recommendation** through cognitive conflict understanding. The paper argues that personalization becomes stronger when the system models emotional and cognitive patterns together. Although the implementation is more complex than needed for a student project, the conceptual relevance is high because the present work also aims to turn emotion understanding into better support decisions.

Li *et al.* [16] introduced **an explanation framework for AI-based text emotion analysis and visualisation**. Their paper expands explainability beyond image saliency and demonstrates that emotion AI should show reasoning cues and interpretable evidence. This complements the visual explainability path of Grad-CAM by motivating per-modality reporting and confidence summaries in the final application.

Shulner-Tal *et al.* [17] studied user perceptions of **human versus AI decision-making and the power of explainable AI**. The results indicate that explanation quality can reduce fairness and trust gaps between AI and human decision agents. This finding is important because an emotion-aware support system should not merely be accurate; it should also be understandable to its user.

Muller [18] systematically reviewed **how explainable AI affects human performance through saliency maps**. The paper showed that explanation tools can positively or negatively influence users depending on their quality and presentation. This is relevant to the proposed system because Grad-CAM and confidence displays must be treated as support mechanisms, not as infallible proof of correctness.

### Research Gaps Identified

The literature reveals four major gaps. First, many systems still depend on a single dominant modality such as face or text. Second, advanced models often prioritize benchmark performance while neglecting reproducible low-resource deployment. Third, ethical safeguards and privacy-aware logging are frequently treated as discussion points rather than implemented modules. Fourth, recommendation systems and emotion-recognition models are often disconnected, meaning that the emotional label is produced but not operationalized into a wider lifestyle support loop. These gaps motivate the proposed work.

## III. Proposed Work

### A. System Objective

The proposed system is designed as a single-file Python application implemented in Streamlit so that both demonstration and experimentation can be performed without separating frontend and backend services. The project is not limited to one narrow task such as music selection. Instead, it acts as an adaptive emotional-support interface that receives multimodal input, estimates the dominant mood, explains the reasoning pathway, and returns safe and useful responses.

### B. Input Modalities

The system accepts four primary inputs:

1. **Free-text context** entered by the user.
2. **Emoji self-report** selected by the user as a lightweight mood prior.
3. **Optional voice-note input**, processed through a speech-to-text fallback when available.
4. **Optional face-image input**, processed by either a trained custom CNN or a heuristic fallback.

These modalities were selected because they are practical in a VS Code or browser-based academic environment. They do not require dedicated physiological sensors, special cameras, or expensive platforms.

### C. Per-Modality Emotion Inference

Each input modality is converted into an emotion-score vector over the project classes: **happy, sad, angry, anxious, calm, focused, tired, and neutral**.

- **Text pathway:** local lexicon-based scoring is always available. When a free Hugging Face token is supplied, the app can optionally upgrade to a transformer-based text emotion model.
- **Emoji pathway:** the self-reported emoji acts as a prior belief and stabilizing signal.
- **Voice pathway:** audio is transcribed first, then passed through the same text emotion logic.
- **Image pathway:** if a trained CNN exists, the image is classified using the custom model; otherwise, a heuristic visual fallback estimates mood-like cues.

### D. Confidence-Aware Multimodal Fusion

The project uses a confidence-aware fusion engine. Each modality contributes:

- an emotion score distribution,
- a dominant local label,
- a confidence estimate,
- and a descriptive note about the inference source.

The fused output is generated by weighted aggregation. User-adjustable sliders allow emphasis or de-emphasis of text, emoji, audio, or image input. This makes the system practical in real use, because some sessions may include only text and emoji, while others may rely more heavily on image evidence.

### E. Custom CNN with Global Average Pooling

The visual module intentionally uses a **custom CNN with Global Average Pooling (GAP)** instead of depending solely on MobileNetV2. The architecture contains stacked convolution, batch normalization, ReLU, pooling, and dropout blocks, followed by a final convolution layer named `last_conv`. Instead of flattening the last feature maps into a large dense block, GAP is applied to reduce each map into a single representative value.

This design choice has several advantages:

- lower parameter count,
- lower overfitting risk,
- easier academic explanation,
- improved suitability for smaller datasets,
- direct compatibility with Grad-CAM.

In a viva context, the model is also easier to defend because every major block of the network is visible and under the student's control.

### F. Explainable AI with Grad-CAM

Grad-CAM is used to generate a heatmap over the uploaded face image. The heatmap highlights which visual regions most influenced the predicted emotion. This step is important not only for debugging but also for trust, presentation, and accountability. A black-box emotion classifier may be questioned by evaluators. A Grad-CAM overlay helps demonstrate that the model is focusing on semantically meaningful regions rather than arbitrary background pixels.

### G. Digital Emotional Twin

The system writes interaction records into a **Digital Emotional Twin** log stored in CSV format. Each record includes:

- timestamp,
- user name,
- free-text context,
- transcript when available,
- selected emoji,
- dominant fused emotion,
- confidence value,
- active modalities,
- score distribution,
- ethical warning flags.

This approach creates a lightweight longitudinal state model without requiring a heavy database. It is useful for trend analysis, later personalization, and project demonstration.

### H. Ethical AI Monitoring

The ethical AI monitor performs three practical checks:

1. It warns when only one modality is active.
2. It warns when the fused confidence is low.
3. It reminds the user that the system is supportive rather than diagnostic.

This design is directly inspired by recent literature on emotion-recognition ethics and by the practical need to avoid overclaiming in sensitive domains.

### I. Adaptive Recommendation and RL-Inspired Feedback Logic

The application generates mood-aligned resource links and action suggestions. A structured catalog with more than one hundred items is generated across YouTube, YouTube Music, and Spotify search sources. Instead of downloading media, the system offers safe link-based recommendations and offline fallback guidance.

Feedback is collected through a simple usefulness rating interface. The ratings are stored and used to bias future ranking by mood and source. This is a lightweight **reinforcement-learning style adaptation loop**. It is simpler than a full deep RL formulation, but it is practical, explainable, and fully compatible with the academic scope of the project.

### J. Free API Integration

The project does not depend on paid APIs for its main workflow. Instead, it uses **optional Hugging Face inference** as the recommended free integration path. If a token is not provided, the app continues to function with local emotion scoring and template-based reflective questions. This design makes the project demonstrable even in limited-resource or offline conditions.

### K. Proposed Method Flowchart

```text
User Input
    |
    +--> Free text context
    +--> Emoji self-check
    +--> Voice note -> transcription fallback
    +--> Face image
            |
Preprocessing + per-modality scoring
            |
Confidence-aware multimodal fusion
            |
Ethical AI monitor
            |
Digital Emotional Twin logger
            |
Adaptive output engine
    +--> reflective questions
    +--> action plan
    +--> link-based recommendations
    +--> feedback reward update
```

### L. Method Pipeline

1. Collect user input from text, emoji, optional voice note, and optional face image.
2. Preprocess each modality and generate per-modality emotion scores.
3. Normalize and fuse scores using confidence-aware weighted aggregation.
4. Run ethical AI checks and expose confidence plus modality contributions.
5. Generate reflective questions and an adaptive action plan.
6. Rank and display recommendation resources with no direct MP3 download dependency.
7. Log the interaction as a Digital Emotional Twin record and update reward signals from feedback.

## IV. Results and Discussion

### A. Prototype Functional Results

A prototype execution of the delivered application produced stable behavior in three representative student scenarios. In an **anxious viva scenario** using text and emoji inputs, the system classified the fused state as **anxious** with confidence **0.8077**, while still preserving a secondary **focused** component of **0.1923**. This is important because it shows the model can retain mixed emotional structure instead of collapsing every stressful academic state into a single rigid label.

In a **focused project-completion scenario**, the text and emoji signals aligned perfectly and the fused result became **focused** with confidence **1.0**. This behavior is desirable because when modality agreement is strong, the system becomes more decisive and can safely shift its response toward productivity support, planning prompts, and concentration-oriented resources.

In a **tired recovery scenario**, the fused output became **tired** with confidence **1.0**, leading the system to produce rest-oriented guidance rather than forcing additional high-energy tasks. This is one of the practical strengths of the project: it does not merely optimize engagement, but attempts to align output with the user's likely immediate capacity.

### B. Structured Resource Output

The generated recommendation catalog contained **120 structured resource entries**, satisfying the requirement of a 100-plus resource base. These entries are organized by mood and source, and each item contains offline fallback guidance. This is operationally significant because the project can provide rich recommendations without storing copyrighted music or requiring direct MP3 downloads.

### C. Explainability and Transparency Outcomes

The addition of Grad-CAM makes the system stronger from an academic evaluation standpoint. Many student projects stop at prediction. In contrast, the present application can show the likely visual evidence that influenced a model decision whenever a trained CNN is available. Together with confidence reporting and per-modality summaries, this produces a more transparent and defensible user experience.

### D. Comparison with Previous Methods

Compared with earlier single-modality systems such as [1], [2], and [7], the proposed project offers broader contextual coverage by combining text, emoji, audio transcript, and image input. Compared with advanced multimodal research models such as [8] and [9], the present work does not claim state-of-the-art benchmark dominance. Instead, its main contribution lies in implementation integration: multimodal affect sensing, ethical monitoring, explainability, digital twin logging, and adaptive recommendation exist together inside one executable Python artifact.

Compared with MobileNetV2-style transfer-learning demos commonly used in academic projects, the custom CNN plus GAP design offers a more transparent learning path, lower architectural opacity, and a cleaner route to Grad-CAM explanation. MobileNetV2 is efficient, but the proposed architecture is easier to justify during project defense because the student can explain the exact role of each layer, why GAP reduces parameters, and how the final convolutional maps interact with Grad-CAM.

### E. Discussion of Research Gaps Addressed

The system addresses the earlier literature gaps in the following way:

- **Gap 1: unimodal dependence.** The project introduces practical multimodal fusion.
- **Gap 2: low explainability.** The project integrates Grad-CAM, confidence reporting, and per-modality interpretation.
- **Gap 3: missing ethical controls.** The project includes warnings, scope limitation, and minimal logging principles.
- **Gap 4: weak operationalization of emotion.** The project converts emotion into reflection prompts, action guidance, and adaptive recommendation behavior.
- **Gap 5: difficult deployment.** The project is delivered as a single-file Python application with Windows/VS Code setup guidance and a free API pathway.

### F. Limitations

Although the system is practically useful, several limitations remain. The quality of image-based emotion inference depends on dataset quality and training conditions. Free APIs can be rate-limited. The feedback adaptation loop is lightweight and not a full reinforcement-learning environment. Low-resource hardware limits dataset scale and model complexity. Finally, emotional states remain subjective, which means that no automated interpretation should be treated as a definitive psychological truth.

## V. Conclusion and Future Scope

This paper presented a complete major-project framework titled **Cognitive Emotion Intelligence and Adaptive Lifestyle System**. The work addressed key research gaps identified in recent emotion-aware recommendation and multimodal affective computing studies, particularly the lack of practical single-source-code deployment, explainability, ethical monitoring, and adaptive logging in low-cost academic systems. The delivered Python application integrates text, emoji, optional voice transcription, and image analysis into a confidence-aware fusion engine, then converts emotional understanding into reflective questions, lifestyle support, and adaptive resource recommendations. By combining a custom CNN with Global Average Pooling, Grad-CAM explainability, CSV-based Digital Emotional Twin logging, and an optional free Hugging Face inference path, the proposed system demonstrates a realistic balance between technical innovation and reproducibility. Future work can expand the system through federated privacy-preserving personalization, longer-term digital twin analytics, multilingual support, and transformer-based multimodal fusion on larger datasets.

### Future Scope Points

1. Integrate federated or on-device personalization so that sensitive emotional interaction data does not require centralized storage.
2. Extend the Digital Emotional Twin with dashboards, trend analytics, anomaly alerts, and longer-term lifestyle adaptation.
3. Upgrade from confidence-weighted late fusion to transformer-based multimodal fusion when larger labeled datasets and stronger hardware are available.

## Major Viva Questions and Explanatory Notes

### 1. What is the protagonist of the project?

The protagonist is not merely the CNN or the text classifier. The true protagonist is the **adaptive decision-support loop** that senses the user's state, explains it, logs it, and converts it into practical support.

### 2. How is this project different from earlier 2023-2025 emotion systems?

Most earlier systems emphasized one of the following: classification, recommendation, or deployment. The present project combines all three, and adds explainability, ethical monitoring, and Digital Emotional Twin logging.

### 3. What is the significance of Explainable AI, Grad-CAM, MobileNetV2 alternatives, and GAP?

Explainable AI helps justify the prediction process. Grad-CAM provides localized visual evidence. The decision to use a compact custom CNN instead of relying fully on MobileNetV2 improves academic transparency. GAP reduces parameter count, supports regularization, and preserves better explainability structure.

### 4. How does the project solve a real-world problem?

It helps users receive emotionally aligned digital support rather than generic or history-only recommendations. In real life, students and professionals often need different forms of assistance depending on whether they are stressed, focused, calm, or exhausted.

## References

[1] D. Srivastava, S. Puri, S. D. Bhagat, and M. Verma, "Emotify: An AI-Powered Emotion-Based Music Recommendation System," 2023 4th International Conference on Intelligent Technologies (CONIT), 2024.

[2] A. Chikaraddi, S. G. Janakki, S. G. Kanakaraddi, and P. S. M., "Emotion-Driven Music Recommender System with Deep Learning and Streamlit Integration," 2025 International Conference on Multi-Agent Systems for Collaborative Intelligence (ICMSCI), 2025.

[3] M. A. Talaghat, E. Parvinnia, and R. Boostani, "Content-based music recommender system based on music emotion using deep learning," *Iran Journal of Computer Science*, 2025.

[4] B. Pan, K. Hirota, Z. Jia, and Y. Dai, "A review of multimodal emotion recognition from datasets, preprocessing, features, and fusion methods," *Neurocomputing*, 2023.

[5] X. Zhu, C. Guo, H. Feng, and Y. Huang, "A Review of Key Technologies for Emotion Analysis Using Multimodal Information," *Cognitive Computation*, 2024.

[6] M. P. A. Ramaswamy and S. Palaniswamy, "Multimodal emotion recognition: A comprehensive review, trends, and challenges," *WIREs Data Mining and Knowledge Discovery*, 2024.

[7] Z. Gou and Y. Li, "Integrating BERT Embeddings and BiLSTM for Emotion Analysis of Dialogue," *Computational Intelligence and Neuroscience*, 2023.

[8] X. Du and L. Zhu, "Self-Learning Multimodal Emotion Recognition Based on Multi-Scale Dilated Attention," *Brain Sciences*, 2026.

[9] J. Wu, J. Wu, Y. Zheng, P. Zhan, and M. Han, "MLGAT: multi-layer graph attention networks for multimodal emotion recognition in conversations," *Journal of Intelligent Information Systems*, 2025.

[10] B. Liu, J. Guo, C. L. P. Chen, X. Wu, and T. Zhang, "Fine-Grained Interpretability for EEG Emotion Recognition: Concat-Aided Grad-CAM and Systematic Brain Functional Network," *IEEE Transactions on Affective Computing*, 2024.

[11] M. Kaur and M. Kumar, "Facial emotion recognition: A comprehensive review," *Expert Systems*, 2024.

[12] A. Katirai, "Ethical considerations in emotion recognition technologies: a review of the literature," *AI and Ethics*, 2024.

[13] S. Davila-Gonzalez and S. Martin, "Human Digital Twin in Industry 5.0: A Holistic Approach to Worker Safety and Well-Being through Advanced AI and Emotional Analytics," *Sensors*, 2024.

[14] F. Liu, Q. Cao, X. Huang, and H. Liu, "Sentimentally enhanced conversation recommender system," *Complex & Intelligent Systems*, 2025.

[15] Y. Yang, Z. Wang, L. Li, and D. Zeng, "Emotion-Enhanced Dual-Agent Recommendation: Understanding and Leveraging Cognitive Conflicts for Better Personalization," *Applied Sciences*, 2025.

[16] Y. Li, J. Chan, G. Peko, and D. Sundaram, "An explanation framework and method for AI-based text emotion analysis and visualisation," *Decision Support Systems*, 2024.

[17] A. Shulner-Tal, T. Kuflik, D. Kliger, and A. Mancini, "Who Made That Decision and Why? Users' Perceptions of Human Versus AI Decision-Making and the Power of Explainable-AI," *International Journal of Human-Computer Interaction*, 2024.

[18] R. Muller, "How Explainable AI Affects Human Performance: A Systematic Review of the Behavioural Consequences of Saliency Maps," *International Journal of Human-Computer Interaction*, 2025.
