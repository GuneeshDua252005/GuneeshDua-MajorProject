# Literature Matrix for the Major Project

Use this matrix when writing the final research paper report, PPT literature review slide, and viva answers.

## 1. Emotion-aware recommendation

| No. | Citation | Year | Methodology | Dataset | Performance | Salient features | Limitations |
|---|---|---:|---|---|---|---|---|
| 1 | H. Kim and T. Hong, "Emotion-oriented recommender system for personalized control of indoor environmental quality," *Building and Environment*, Elsevier. | 2024 | Emotion ontology, fuzzy emotional similarity, graph attention recommender | Private IEQ-emotion dataset | Improved predictive performance over baseline recommenders | Uses emotion as a recommendation signal | Small and domain-specific dataset |
| 2 | S. Abakarim, S. Qassimi, and S. Rakrak, "Emotion and sentiment enriched decision transformer for personalized recommendations," *Scientific Reports*, Springer Nature. | 2025 | Decision Transformer with emotion/sentiment enrichment | Yelp, Google Local Reviews | Up to +11.76% nDCG@10 on Yelp | Strong bridge between affect mining and long-horizon personalization | Review-text dependence |
| 3 | Y. Zhou et al., "Mitigating Distribution Shift in Offline RL-Based Recommender Systems with a Q-Learning Regularization Decision Transformer," *Information*, MDPI. | 2026 | QRDT offline RL recommender | Amazon domains | Improves HR, NDCG, Recall, Precision | Strong sequential recommendation grounding | No direct emotion modality |

## 2. Multimodal emotion recognition

| No. | Citation | Year | Methodology | Dataset | Performance | Salient features | Limitations |
|---|---|---:|---|---|---|---|---|
| 4 | A. Messaoudi, H. Boughrara, and Z. Lachiri, "Multimodal emotion recognition: integrating speech and text for improved valence, arousal, and dominance prediction," *Annals of Telecommunications*, Springer. | 2025 | Acoustic + linguistic fusion for continuous VAD | IEMOCAP | Late fusion improved overall performance | Continuous affect instead of only discrete labels | Exact abstract metric uncertain |
| 5 | C. Fang et al., "Multimodal Speech Emotion Recognition Based on Large Language Model," *IEICE Transactions on Information and Systems*. | 2024 | GPT-derived text features + audio fusion | IEMOCAP | 79.62% WA, 80.38% UA | Shows LLM-assisted multimodal gains | Benchmark-only scope |
| 6 | M. Khomidov and J.-H. Lee, "The Novel EfficientNet Architecture-Based System and Algorithm to Predict Complex Human Emotions," *Algorithms*, MDPI. | 2024 | EfficientNet-B0 + HRV fusion | FER2013 + HRV data | 74% face-only, 88.2% with HRV | Validates multimodal fusion | Requires physiological setup |

## 3. Explainable AI and Grad-CAM

| No. | Citation | Year | Methodology | Dataset | Performance | Salient features | Limitations |
|---|---|---:|---|---|---|---|---|
| 7 | J. Gebele et al., "Interpreting Emotions Through the Grad-CAM Lens: Insights and Implications in CNN-Based Facial Emotion Recognition," in *Pattern Recognition*, Springer LNCS. | 2025 | Grad-CAM and guided backprop interpretation study | FER-2013, RAF-DB, AffectNet | Interpretability-focused | Strong evidence for explainable FER | Not a deployable end-user system |
| 8 | S. T. H. Shah et al., "Explainable Emotion Recognition Using Xception-Based Feature Extraction and Supervised Machine Learning on the RAVDESS Dataset," *IEEE MeMeA*. | 2025 | Xception features + SVM + Grad-CAM, SHAP, LIME | RAVDESS | 93.87% accuracy | Combines strong prediction and explanation | Acted emotion dataset |

## 4. Lightweight CNN and deployment efficiency

| No. | Citation | Year | Methodology | Dataset | Performance | Salient features | Limitations |
|---|---|---:|---|---|---|---|---|
| 9 | S. B. Punuri et al., "Efficient Net-XGBoost: An Implementation for Facial Emotion Recognition Using Transfer Learning," *Mathematics*, MDPI. | 2023 | EfficientNet transfer learning + XGBoost | CK+, KDEF, JAFFE, FER2013 | Very high on curated datasets | Lightweight transfer-learning baseline | Real-world generalization weaker |
| 10 | J. A. Ramirez-Quintana et al., "Lightweight Convolutional Neural Network with Efficient Channel Attention Mechanism for Real-Time Facial Emotion Recognition in Embedded Systems," *Sensors*, MDPI. | 2025 | Lightweight CNN + channel attention | CK+, KDEF, FER2013, EMOTION-ITCH | 79.2% on FER2013 | Strong real-time deployment story | Dataset realism still varies |

## 5. Conversational and empathetic AI

| No. | Citation | Year | Methodology | Dataset | Performance | Salient features | Limitations |
|---|---|---:|---|---|---|---|---|
| 11 | X. Zhang et al., "Towards Empathetic Conversational Recommender Systems," *RecSys '24*, ACM. | 2024 | Emotion-aware recommendation + empathetic response generation | ReDial | Improved recommendation quality and user satisfaction | Unifies recommendation and conversation | Movie-domain benchmark |
| 12 | S. Meyer and D. Elsweiler, "LLM-based conversational agents for behaviour change support: A randomised controlled trial examining efficacy, safety, and the role of user behaviour," *International Journal of Human-Computer Studies*, Elsevier. | 2025 | GPT-based MI chatbot with safety analysis | Human study | Positive effect on readiness to change | Strong safety and efficacy angle | Not a direct FER paper |
| 13 | Y.-P. Chen et al., "Recent Trends in Personalized Dialogue Generation: A Review of Datasets, Methodologies, and Evaluations," *LREC-COLING*. | 2024 | Review of personalized dialogue generation | Multiple datasets | Review paper | Useful taxonomy for profile-aware chatbots | Survey, not implementation |

## 6. Ethical AI and affective sovereignty

| No. | Citation | Year | Methodology | Dataset | Performance | Salient features | Limitations |
|---|---|---:|---|---|---|---|---|
| 14 | A. Katirai, "Ethical considerations in emotion recognition technologies: a review of the literature," *AI and Ethics*, Springer. | 2024 | Structured literature review | 43 reviewed papers | Not applicable | Strong ethics foundation for FER | No implementation |
| 15 | R. S. Kim, "Formal and computational foundations for implementing Affective Sovereignty in emotion AI systems," *Discover Artificial Intelligence*, Springer. | 2026 | Sovereign-by-design formal framework | Simulation | Lower interpretive override score | Turns ethics into concrete system controls | Human validation still limited |

## 7. Digital twin and user modeling

| No. | Citation | Year | Methodology | Dataset | Performance | Salient features | Limitations |
|---|---|---:|---|---|---|---|---|
| 16 | Y. Zhang et al., "A framework towards digital twins for type 2 diabetes," *Frontiers in Digital Health*. | 2024 | Multi-model digital twin architecture | Arivale longitudinal cohort | Predictive framework | Strong architecture reference for soft digital twin design | Healthcare-specific |
| 17 | M. Kiran et al., "A digital twin framework for predicting and simulating type 2 diabetes onset using retrospective lifestyle data," *Frontiers in Digital Health*. | 2026 | Lifestyle digital twin with causal simulation | UK Biobank | C-index 0.90 | Strong longitudinal lifestyle modeling | Disease-specific |
| 18 | S. Chen et al., "PersonaTwin: A Multi-Tier Prompt Conditioning Framework for Generating and Evaluating Personalized Digital Twins," *GEM2 Workshop*, ACL. | 2025 | LLM persona twin with psychometric conditioning | Healthcare user data | Near-oracle simulation fidelity | Strong foundation for conversation-linked user modeling | Workshop paper |

## How this matrix supports the project

1. It justifies replacing an older backbone such as MobileNetV2 with a stronger and still efficient CNN family such as EfficientNetV2.
2. It supports the use of Grad-CAM and explanation overlays because explainability is a major active concern in facial emotion recognition.
3. It validates multimodal fusion because modern systems outperform unimodal emotion pipelines.
4. It provides academic support for a history-aware recommendation layer with sequential learning logic.
5. It offers a defensible ethics narrative through user override, consent, uncertainty display, and non-clinical positioning.
