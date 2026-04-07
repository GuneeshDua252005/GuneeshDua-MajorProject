# Cognitive Emotion Intelligence and Adaptive Lifestyle System

Author block for submission:
- Student Name: __________________
- Department: ____________________
- Institution: ____________________
- Guide / Supervisor: ____________

## Abstract

Modern intelligent systems increasingly require the ability to understand not only explicit user commands but also emotional context, cognitive pressure, and behavioral intent. Conventional recommender and lifestyle-support systems usually rely on historical usage logs, static questionnaires, or single-modality prediction. Such approaches are often unable to react to real-time changes in mood, stress, academic workload, and personal routine. This paper presents a project titled *Cognitive Emotion Intelligence and Adaptive Lifestyle System*, a practical Python-based framework that combines text emotion analysis, adaptive reflective questioning, lifestyle recommendation, optional music metadata recommendation, reinforcement-learning style feedback adaptation, digital emotional twin memory, ethical AI monitoring, and an explainable visual emotion branch using EfficientNetV2-S with Grad-CAM. The proposed system is designed as a single-source Python application so that it can be demonstrated, extended, and executed easily within a final-year major-project environment using VS Code and standard academic hardware. The system replaces paid proprietary text-generation dependencies with a free-tier Hugging Face inference option and also provides an offline fallback workflow so that the prototype remains functional even without external API access. The paper reviews representative research in affective recommender systems, multimodal emotion recognition, explainable AI, and adaptive recommendation. It then describes the proposed architecture, problem statement, research gap, implementation flow, and a pilot discussion of system outputs. The overall contribution of this work is not a narrow music recommender alone, but a broader emotion-aware lifestyle intelligence assistant that aims to improve personalization, interpretability, accessibility, and continuity of user support in educational and day-to-day productivity settings.

## Keywords

Cognitive emotion intelligence, adaptive lifestyle system, affective computing, explainable AI, Grad-CAM, EfficientNetV2, digital emotional twin, reinforcement learning, Hugging Face API, emotion-aware recommendation

## I. Introduction

Emotion-sensitive intelligent systems have gained significant relevance in recent years because digital interactions now influence education, healthcare, productivity, entertainment, and daily decision making. Music recommendation, student support, productivity planning, mental well-being tracking, and personalized advisory systems increasingly require models that go beyond static preference analysis. Traditional recommendation methods such as collaborative filtering, content-based filtering, and hybrid ranking have demonstrated strong utility in large-scale online systems, but they generally depend on historical behavior rather than the present emotional condition of the user. As a result, these systems may recommend technically relevant content while still failing to match the user's immediate mindset, stress level, or psychological readiness.

This limitation is especially visible in academic and lifestyle-oriented settings. A student preparing for a viva examination, a hackathon, or a final-semester project may be anxious, mentally fatigued, or highly motivated at different times of the day. A recommendation engine that only remembers earlier preferences cannot fully respond to this dynamic context. Psychological and affective-computing studies show that human decisions, music selection patterns, and productivity behaviors are strongly shaped by emotion. Therefore, systems that can recognize emotional state and adapt their responses accordingly are more suitable for real-world personalized assistance.

Recent advances in artificial intelligence, deep learning, natural language processing, and multimodal sensing have improved the feasibility of such systems. Text-based emotion recognition, facial expression recognition, speech emotion recognition, transformer-based contextual modeling, and reinforcement learning have each contributed to the emergence of emotion-aware recommender frameworks. Several studies have explored emotion-aware music recommendation [1]-[8], while others have proposed broader affective recommender models [9]-[16]. At the same time, explainable AI methods such as Grad-CAM [19] have become important because academic, healthcare, and human-centered systems must justify predictions in a way that users, evaluators, and stakeholders can understand.

Despite this progress, multiple gaps remain. First, many existing systems are limited to one application domain, especially music recommendation, without extending emotional intelligence to broader adaptive lifestyle support. Second, a large number of systems depend on only one modality, such as text or audio, which reduces robustness when user inputs are noisy or incomplete. Third, many proposed solutions emphasize accuracy but neglect transparency, ethical safeguards, and user-continuity mechanisms. Fourth, several project prototypes are difficult to reproduce in ordinary academic environments because they depend on multiple codebases, paid APIs, or heavy deployment stacks. Fifth, historical recommender models often do not update their action policy from immediate user feedback, which limits practical personalization.

The present work addresses these gaps by proposing a single-file Python framework called *Cognitive Emotion Intelligence and Adaptive Lifestyle System*. The protagonist of this project is not merely emotion detection; instead, the central concept is a continuously adapting, explainable, and ethically monitored emotional-intelligence assistant. The system analyzes user text, optionally incorporates visual emotion predictions, stores recent patterns in a digital emotional twin, selects an adaptive action through reinforcement-learning style logic, asks important reflective questions, and suggests lifestyle or music-support interventions without downloading media files. The program is compatible with a standard VS Code environment and is intentionally designed to be modular while remaining executable as a single source file.

### A. Problem Statement

Existing emotional intelligence and lifestyle-assistance systems from the 2023-2025 period often suffer from one or more of the following issues:

1. dependence on historical preference data instead of real-time emotional context;
2. reliance on a single modality such as text, music metadata, or facial images;
3. absence of explainable AI for interpreting visual predictions;
4. weak continuity across sessions due to lack of persistent emotional memory;
5. minimal ethical checking related to consent, uncertainty, and stress-sensitive usage;
6. static recommendation logic that does not learn from user feedback;
7. dependence on proprietary APIs or fragmented multi-file architectures that are difficult to demonstrate in a major-project setting.

Accordingly, the research problem addressed in this paper is: *How can a low-cost, single-file, emotion-aware Python system combine multimodal affect recognition, explainable AI, adaptive questioning, digital emotional memory, and reinforcement-style personalization in a way that is practical for academic major-project deployment?*

### B. Applications

The proposed system has applications in:

- student productivity and exam-pressure support,
- hackathon and project-preparation assistance,
- emotion-aware music or content guidance,
- reflective journaling and well-being support,
- smart assistants and adaptive human-computer interaction,
- wellness-oriented lifestyle recommendation,
- AI teaching demonstrations involving explainability and ethical AI.

### C. Existing Challenges

Emotion understanding remains difficult because human language is ambiguous, context-dependent, and often non-literal. Anxiety, sadness, frustration, calmness, and motivation may co-exist in the same message. Facial-expression models also face issues such as illumination changes, class imbalance, and dataset bias. Large transformer models offer better contextual understanding, but their computational requirements can be restrictive in a normal laptop setting. User privacy and ethical concerns further complicate deployment because emotional data is highly sensitive. These challenges motivate the need for a practical architecture that remains light enough for academic use while still supporting strong conceptual depth.

### D. Existing Solutions

Prior research has offered hybrid recommender systems, LSTM- and CNN-based emotion classification, multimodal fusion, contextual emotion tagging, and transformer-based recommendation strategies [1]-[18]. However, most previous works focus on one slice of the problem. Some improve emotion classification, some improve recommendation ranking, and others discuss theoretical affective recommender design. Fewer works combine all of the following in a practical student-project system: free API use, ethical AI monitoring, digital emotional memory, RL-style action adaptation, explainable visual inference, and a single-source implementation model.

### E. Paper Organization

The remainder of this paper is organized as follows. Section II reviews representative literature and identifies research gaps. Section III explains the proposed methodology, architecture, and implementation flow. Section IV discusses prototype outputs, comparison with previous work, and implementation significance. Section V concludes the paper and outlines future scope.

## II. Literature Review

### A. Review of Representative Papers

**[1] Shan et al. (2009)** presented one of the earlier emotion-based music recommendation approaches by discovering affinity relations from film music. The work demonstrated that emotion can serve as an effective recommendation cue rather than a decorative label. Its strength lies in showing how affective similarity can guide recommendation, but the method predates modern multimodal deep learning and does not support real-time personal state tracking.

**[2] Deng et al. (2015)** explored user emotion in microblogs for music recommendation and showed that social text can be mapped to music preference changes. The study is important because it connects user-generated emotional expression with recommendation logic. However, the framework depends on social-media signal quality and does not include explainable AI or adaptive lifestyle assistance.

**[3] Polignano et al. (2021)** proposed an affective coherence model for emotion-aware recommender systems. Their contribution is conceptually important because it formalizes the relationship between emotion-driven behaviors and recommendation outcomes. The work is useful for system design thinking, yet it is more focused on recommender coherence modeling than on practical single-file deployment or multimodal explainability.

**[4] Wang et al. (2021)** introduced a novel emotion-aware hybrid music recommendation method using a deep neural network. The study combined content-based and collaborative reasoning with an emotion model, improving recommendation relevance over standard baselines. A limitation is that the implementation remains centered on music recommendation rather than broader lifestyle adaptation and ethical decision support.

**[5] De Prisco et al. (2022)** examined induced emotion-based music recommendation through reinforcement learning. This is particularly relevant to the present work because it demonstrates that emotional recommendation can benefit from adaptive policy updates rather than fixed ranking rules. However, the paper concentrates on music-induction objectives and does not integrate digital emotional memory or explainable visual recognition.

**[6] Kim et al. (2021)** modeled recommendation using emotional information and collaborative filtering. The study highlighted that emotional metadata can help recommendation systems better represent user preference variation. The approach remains valuable as a hybrid baseline, though it does not fully address multimodal fusion, RL-style personalization, or interpretability.

**[7] Krupa et al. (2020)** proposed an emotion-aware smart music recommender using a two-level CNN architecture. Their work is useful because it demonstrates the role of CNNs in extracting mood-related features automatically. Nevertheless, the scope remains domain-specific and does not include a broader adaptive lifestyle system or ethical monitoring layer.

**[8] Jagadeesh et al. (2024)** employed Inception-ResNet V2 in an emotion-infused music recommender. The paper shows the growing shift toward stronger CNN backbones for affective applications. Even so, the emphasis is still largely on recommendation performance rather than explainable AI, persistent user state memory, or a free API-based question-generation assistant.

**[9] Srivastava et al. (2024)** proposed Emotify, an AI-powered emotion-based music recommendation system. The work is aligned with the modern trend of combining user mood recognition with recommendation pipelines. The limitation, from the perspective of this project, is that it does not extend the model into an integrated lifestyle, reflection, and feedback-learning ecosystem.

**[10] Xiao et al. (2025)** presented an efficient bi-modal fusion framework for music emotion recognition. This contribution is relevant because multimodal fusion is central to robust affective inference. The paper focuses on recognition quality, while the present project extends the same spirit into a user-facing adaptive-support system with downstream action selection.

**[11] Ayadi and Lachiri (2022)** combined CNN and LSTM models for audio emotion recognition from speech and song attributes. This work confirms the benefit of combining local feature extraction and sequential context modeling. However, audio-only emotion pipelines still face domain constraints and do not directly solve lifestyle personalization or explainability concerns.

**[12] Ara and Rekha (2024)** studied emotion classification of music lyrics using LSTM networks. The paper shows that text-based emotion signals remain highly useful even without audio or image input. The limitation is that lyric-based analysis is narrower than the broader open-text journaling and reflective input used in the proposed system.

**[13] Kalsum et al. (2018)** addressed facial emotion recognition using hybrid feature descriptors. Their findings remain relevant because facial emotion remains one of the strongest visual affect modalities. Yet the method predates recent backbone improvements and does not provide the interpretability benefits now expected in explainable AI settings.

**[14] Almomani et al. (2021)** discussed rational, emotional, and attentional models for recommender systems. The paper is significant at a conceptual level because it acknowledges that recommendation cannot be fully explained through rational choice alone. This supports the philosophical foundation of the current project, which treats mood, cognition, and adaptation as joint drivers of assistance.

**[15] Salazar et al. (2024)** proposed a generic architecture for an affective recommender system in e-learning environments. This is highly relevant to academic productivity and personalized assistance. The present work extends this line by adding digital emotional twin memory, explainable facial analysis, and a free API-based questioning interface within a single executable application.

**[16] Tkalcic et al. (2013)** introduced an early framework and case study on emotion-aware recommender systems. Their work is frequently cited because it formalized the design space of affective recommendation. The proposed project builds on this foundation but shifts toward contemporary implementation concerns such as explainability, portability, and adaptive feedback loops.

**[17] Yang et al. (2021)** leveraged sentence-level emotion analysis for recommendation. The paper is important because it captures fine-grained semantic affect instead of relying only on coarse sentiment labels. This supports the use of contextual language understanding in the proposed system, although the present implementation additionally focuses on user reflection, lifestyle response, and low-cost deployment.

**[18] Katarya and Verma (2016)** reviewed recent developments in affective recommender systems and showed the broad relevance of emotion in personalized recommendation. Their analysis helps identify persistent challenges such as context handling, sparse data, and emotional variability. The review also highlights why more integrated and adaptive systems are still needed.

**[19] Selvaraju et al. (2020)** presented Grad-CAM, a widely used visual explanation method for deep networks. Although not a recommender paper, it is essential for this project because it addresses the transparency problem in image-based emotion recognition. By highlighting salient facial regions that influence the classifier, Grad-CAM increases interpretability during demonstrations and viva evaluation.

**[20] Tan and Le (2021)** introduced EfficientNetV2, a more modern CNN family with strong accuracy-efficiency trade-offs. This architecture is chosen in the proposed system instead of MobileNetV2 because it offers a better balance between representational power and computational practicality. It is especially useful when paired with Global Average Pooling and Grad-CAM-based explanation.

### B. Literature Gap Summary

The literature reveals consistent progress in affective recommendation and emotion recognition, but several research gaps remain:

1. Most systems remain domain-specific and do not expand into broader lifestyle adaptation.
2. Only a subset of works combine multiple modalities and downstream action personalization.
3. Explainable AI is rarely integrated directly into student-friendly affective systems.
4. Ethical AI monitoring is usually discussed abstractly rather than implemented.
5. Persistent emotional memory and reinforcement-style adaptation are underrepresented.
6. Reproducible, single-file academic prototypes are uncommon despite their pedagogical value.

These gaps justify the need for the present system.

## III. Proposed Methodology

### A. Proposed System Objective

The proposed system is designed to interpret a user's present emotional state, ask meaningful reflective questions, recommend adaptive lifestyle actions, optionally suggest mood-aligned music metadata, and preserve emotional continuity across sessions. Unlike a narrow music recommender, the system acts as a lightweight cognitive-emotional assistant. Its purpose is to support student well-being, productivity, and emotional self-awareness in a single, demonstration-ready Python implementation.

### B. System Architecture

The system contains six major layers:

1. **Input Layer**  
   Accepts journal text, user context, study pressure, sleep, energy level, and optional facial image.

2. **Ethical AI Monitoring Layer**  
   Checks consent, short-input uncertainty, low-confidence situations, and high-stress conditions.

3. **Emotion Intelligence Layer**  
   Performs text emotion analysis, optional visual emotion prediction through EfficientNetV2-S, and multimodal fusion.

4. **Adaptive Reasoning Layer**  
   Generates important questions, selects lifestyle actions, and applies reinforcement-learning style policy updates from user feedback.

5. **Digital Emotional Twin Layer**  
   Stores recent emotional states, actions, and goals to preserve continuity across sessions.

6. **Recommendation and Explanation Layer**  
   Produces adaptive lifestyle guidance, optional Spotify metadata recommendations, and Grad-CAM explanations for visual predictions.

### C. Textual Flowchart

```text
User profile + journal text + optional face image
                 |
                 v
        Ethical consent and risk checks
                 |
        -------------------------------
        |                             |
        v                             v
  Text emotion analysis      EfficientNetV2-S image branch
        |                             |
        |                       Grad-CAM explanation
        ----------- Multimodal fusion -------------
                           |
                           v
                Digital emotional twin update
                           |
                           v
            RL-style adaptive action selection
                           |
                           v
    Important questions + lifestyle plan + music metadata
```

### D. Method Pipeline

The proposed methodology follows the steps below:

1. Collect user context such as role, study pressure, sleep hours, energy level, current goal, and a short journal-style text.
2. Perform ethical AI checks before heavy inference or recommendation.
3. Analyze the journal text through an offline-first emotion scoring engine.
4. Optionally obtain image-based emotion probabilities through EfficientNetV2-S.
5. Fuse text and image evidence to obtain a final emotional state.
6. Query a free-tier Hugging Face model for reflective questions, or use a built-in fallback question bank.
7. Use a reinforcement-learning style policy table to choose the most suitable next action.
8. Recommend lifestyle adjustments and optional Spotify track metadata according to the predicted mood.
9. Save the interaction in a digital emotional twin memory file for future personalization.
10. If the visual branch is active, use Grad-CAM to explain the model's focus region.

### E. Why EfficientNetV2-S Instead of MobileNetV2

MobileNetV2 is lightweight and historically important, but it is now a comparatively older baseline for modern explainable vision workflows. EfficientNetV2-S provides stronger feature extraction quality while still remaining efficient enough for standard systems. It also works naturally with Global Average Pooling, which reduces parameter count in the classification head and aligns well with class activation and Grad-CAM style explanation. Therefore, EfficientNetV2-S is a more suitable modern backbone for this project.

### F. Significance of Global Average Pooling

Global Average Pooling (GAP) replaces large fully connected blocks with a compact aggregation mechanism that averages spatial feature maps. This reduces the number of trainable parameters, lowers overfitting risk, and improves portability. For explainable AI, GAP also keeps stronger correspondence between convolutional feature maps and final class decisions. Hence, GAP is valuable both computationally and interpretively.

### G. Explainable AI and Ethical AI Monitoring

Explainable AI is critical in emotional intelligence systems because users and evaluators should understand why a model predicted stress, sadness, or happiness. In this project, Grad-CAM is used to highlight salient visual regions affecting the facial emotion prediction. Ethical AI monitoring complements this by warning about low-confidence cases, checking consent-sensitive usage, and treating the system as supportive rather than diagnostic. Together, these features improve trustworthiness and academic defensibility.

### H. Digital Emotional Twin

The digital emotional twin is the continuity engine of the system. Instead of producing one-time recommendations and forgetting them, the program stores recent emotional states, goals, and selected actions in a local artifact file. Over time, this makes it possible to recognize trends such as recurring anxiety before presentations, sustained productivity during specific schedules, or repeated low-energy patterns after sleep loss. This protagonist idea differentiates the project from ordinary emotion classifiers.

### I. Reinforcement Learning Logic

The system uses a small Q-table style adaptive policy rather than a full-scale deep RL environment. This design choice keeps the implementation simple and executable on standard hardware. User feedback on the usefulness of the recommended action is converted into a reward signal, which updates future action scores for similar states. This mechanism enables low-cost personalization without requiring a large online learning infrastructure.

### J. Free API Integration

The text question-generation component supports the Hugging Face Inference API as a free alternative to paid proprietary APIs. The system reads the `HF_TOKEN` environment variable, calls an open-source text model, and returns practical reflective questions. When the token is unavailable or the call fails, the system uses an internal question bank. Optional Spotify integration is handled through developer credentials, but the system remains fully operational in offline mode with built-in search suggestions. This makes the implementation financially accessible and suitable for academic demonstration.

## IV. Results and Discussion

### A. Prototype Execution Discussion

The present repository includes a fully executable text workflow and a training-ready vision workflow. The text path can run immediately after installing lightweight dependencies, while the image path requires FER-style data and deep-learning packages. Because this paper accompanies a reproducible repository rather than a fabricated benchmark sheet, quantitative image-branch accuracy is intentionally left to local training runs. This is academically preferable to inventing results that cannot be verified.

### B. Sample Output Scenarios

**Scenario 1: Viva and deadline stress**  
Input text: "I feel pressure because my project report and viva are close. I want to finish everything but I am worried I may not manage on time."  
Observed output: The system labels the state as *anxious*, recommends a breathing reset and task decomposition, asks priority-oriented reflective questions, and suggests calm study music search terms.  
Interpretation: The model links academic pressure phrases to anxious and motivated cues simultaneously, which is appropriate for student project settings.

**Scenario 2: Emotional fatigue**  
Input text: "I am tired, overwhelmed, and mentally drained. I have too many tasks and do not know where to begin."  
Observed output: The system labels the state as *overwhelmed*, prioritizes recovery and simplification, and recommends a walk-and-reset style action.  
Interpretation: This demonstrates the advantage of using action adaptation rather than only classification.

**Scenario 3: Productive momentum**  
Input text: "I finally feel motivated and ready to complete the hard part of my project today."  
Observed output: The system predicts *motivated*, recommends a focused study block, and generates questions around converting momentum into routine.  
Interpretation: The system supports positive-state optimization instead of assuming recommendation is only for negative moods.

### C. Comparison with Earlier Research Gaps

The proposed system addresses earlier research gaps from 2023-2025 emotional-intelligence project trends in the following way:

1. **From narrow recommendation to broader adaptation**  
   Earlier systems often stopped at music suggestion or emotion label generation. The proposed work converts emotional understanding into practical lifestyle and productivity actions.

2. **From one-time prediction to continuity**  
   Many earlier prototypes lacked persistent personalization. The digital emotional twin records patterns across sessions and provides continuity.

3. **From opaque vision models to explainability**  
   Earlier student projects often used CNN classifiers without explanation. This project adds Grad-CAM to justify visual predictions.

4. **From static logic to adaptive feedback**  
   Traditional student projects rarely updated recommendations from user feedback. The Q-table style policy allows action personalization over time.

5. **From costly dependence to accessible deployment**  
   Instead of relying on only paid APIs, the system uses a free Hugging Face path plus an offline fallback strategy.

### D. Feature-Level Comparison with Prior Systems

- **Emotion-only projects**: Usually classify mood but do not store emotional memory.
- **Music-only recommenders**: Usually improve playlist relevance but do not support reflective questioning or lifestyle adaptation.
- **Single-modality prototypes**: Often fail when one signal is noisy or absent.
- **Black-box image classifiers**: Provide labels but not explanations.
- **Current proposed system**: Combines emotion understanding, digital memory, adaptive questioning, RL-style personalization, optional multimodal fusion, and explainability in one practical implementation.

### E. Real-World Problem Solving Value

The system addresses real-world problems in student and youth digital life:

- managing exam and viva stress,
- reducing productivity breakdown caused by emotional overload,
- supporting structured self-reflection,
- improving personalized non-clinical well-being support,
- giving users understandable AI outputs instead of black-box labels,
- offering a low-cost prototype path for educational institutions and hackathons.

### F. Why the Project is Different from Others

The differentiating protagonist of this project is the combination of *cognitive emotion intelligence* with *adaptive operational lifestyle support*. Most projects focus on recognizing feelings; this project attempts to interpret emotion, preserve its context, explain it, and respond to it with adaptive action. In other words, the system does not merely say "the user is anxious." It tries to answer, "Given that the user is anxious, what should the system ask, suggest, remember, and explain next?"

## V. Conclusion and Future Scope

This paper presented a complete concept and implementation package for a *Cognitive Emotion Intelligence and Adaptive Lifestyle System* developed as a single-file Python major project. The study reviewed relevant work in affective recommendation, multimodal emotion recognition, explainable AI, and adaptive learning. Based on the identified gaps, a practical architecture was proposed that integrates text-based emotion analysis, optional EfficientNetV2-S facial recognition, Grad-CAM explanation, digital emotional twin memory, reinforcement-style action adaptation, ethical AI monitoring, and free-tier API support through Hugging Face. The resulting system is suitable for academic demonstration because it is technically meaningful, reproducible, and extendable.

### Future Scope

1. Integrate real speech emotion recognition and physiological sensor signals for stronger multimodal fusion.
2. Replace the small adaptive policy table with contextual bandits or a deeper reinforcement-learning framework.
3. Develop a lightweight Android interface and package the system as an APK using Kivy and Buildozer.
4. Add bias-analysis and fairness dashboards for demographic robustness evaluation.
5. Conduct controlled user studies to measure behavioral usefulness, trust, and long-term personalization quality.

## References

[1] M.-K. Shan, F.-F. Kuo, M.-F. Chiang, and S.-Y. Lee, "Emotion-based music recommendation by affinity discovery from film music," *Expert Systems with Applications*, 2009, doi: 10.1016/j.eswa.2008.09.042.

[2] S. Deng, D. Wang, X. Li, and G. Xu, "Exploring user emotion in microblogs for music recommendation," *Expert Systems with Applications*, 2015, doi: 10.1016/j.eswa.2015.08.029.

[3] M. Polignano, F. Narducci, M. de Gemmis, and G. Semeraro, "Towards emotion-aware recommender systems: An affective coherence model based on emotion-driven behaviors," *Expert Systems with Applications*, 2021, doi: 10.1016/j.eswa.2020.114382.

[4] S. Wang, C. Xu, A. S. Ding, and Z. Tang, "A novel emotion-aware hybrid music recommendation method using deep neural network," *Electronics*, 2021, doi: 10.3390/electronics10151769.

[5] R. De Prisco, A. Guarino, D. Malandrino, and R. Zaccagnino, "Induced emotion-based music recommendation through reinforcement learning," *Applied Sciences*, 2022, doi: 10.3390/app122111209.

[6] T.-Y. Kim, H. Ko, S.-H. Kim, and H.-D. Kim, "Modeling of recommendation system based on emotional information and collaborative filtering," *Sensors*, 2021, doi: 10.3390/s21061997.

[7] K. S. Krupa, G. Ambara, K. Rai, and S. Choudhury, "Emotion aware smart music recommender system using two level CNN," in *Proc. 2020 3rd Int. Conf. Smart Systems and Inventive Technology*, 2020, doi: 10.1109/ICSSIT48917.2020.9214164.

[8] M. Jagadeesh, S. A. K, and S. K, "Emotion-infused music recommender system using Inception-ResNet V2 algorithm," in *Proc. 2024 5th Int. Conf. Intelligent Communication Technologies and Virtual Mobile Networks*, 2024, doi: 10.1109/ICICV62344.2024.00066.

[9] D. Srivastava, S. Puri, S. D. Bhagat, and M. Verma, "Emotify: An AI-powered emotion-based music recommendation system," in *Proc. 2023 4th Int. Conf. Intelligent Technologies*, 2024, doi: 10.1109/CONIT61985.2024.10626667.

[10] Y. Xiao, H. Ruan, X. Zhao, and P. Jin, "An efficient bi-modal fusion framework for music emotion recognition," *IEEE Transactions on Affective Computing*, 2025, doi: 10.1109/TAFFC.2024.3486340.

[11] S. Ayadi and Z. Lachiri, "A combined CNN-LSTM network for audio emotion recognition using speech and song attributes," in *Proc. 2022 6th Int. Conf. Advanced Technologies for Signal and Image Processing*, 2022, doi: 10.1109/ATSIP55956.2022.9805924.

[12] A. Ara and R. V, "A study of emotion classification of music lyrics using LSTM networks," in *Proc. 2024 5th Int. Conf. Mobile Computing and Sustainable Informatics*, 2024, doi: 10.1109/ICMCSI61536.2024.00026.

[13] T. Kalsum, S. M. Anwar, M. Majid, and B. Khan, "Emotion recognition from facial expressions using hybrid feature descriptors," *IET Image Processing*, 2018, doi: 10.1049/iet-ipr.2017.0499.

[14] A. Almomani, C. Monreal, J. Sieira, and J. Grana, "Rational, emotional, and attentional models for recommender systems," *Expert Systems*, 2021, doi: 10.1111/exsy.12594.

[15] J. C. Salazar, J. Aguilar, J. Monsalve-Pulido, and E. Montoya, "A generic architecture of an affective recommender system for e-learning environments," *Universal Access in the Information Society*, 2024, doi: 10.1007/s10209-023-01024-8.

[16] M. Tkalcic, U. Burnik, A. Odic, and A. Kosir, "Emotion-aware recommender systems: A framework and a case study," in *Advances in Intelligent Systems and Computing*, 2013, doi: 10.1007/978-3-642-37169-1_14.

[17] C. Yang, X. Chen, L. Liu, and P. Sweetser, "Leveraging semantic features for recommendation: Sentence-level emotion analysis," *Information Processing and Management*, 2021, doi: 10.1016/j.ipm.2021.102543.

[18] R. Katarya and O. P. Verma, "Recent developments in affective recommender systems," *Physica A: Statistical Mechanics and its Applications*, 2016, doi: 10.1016/j.physa.2016.05.046.

[19] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *International Journal of Computer Vision*, 2020, doi: 10.1007/s11263-019-01228-7.

[20] M. Tan and Q. V. Le, "EfficientNetV2: Smaller models and faster training," in *Proc. 38th Int. Conf. Machine Learning*, 2021.

[21] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, "GoEmotions: A dataset of fine-grained emotions," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020.

[22] I. J. Goodfellow et al., "Challenges in representation learning: A report on three machine learning contests," in *Proc. Neural Information Processing Systems Workshop*, 2013.

[23] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019.
