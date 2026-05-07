# Workflow Architecture Flowcharts (Chapter 4 Methodology)

## 1) End-to-End System Workflow

```mermaid
flowchart TD
    A[User Session Start] --> B[Multi-Modal Input Capture]
    B --> B1[Facial Frame / Webcam]
    B --> B2[Emoji or Text Input]
    B --> B3[Voice Transcript]
    B --> B4[Gesture Frame]
    B --> B5[Timestamp Context]

    B1 --> C1[Face Mood Detection<br/>Lightweight CV Heuristics]
    B2 --> C2[Emoji + Keyword Sentiment Mapping]
    B3 --> C3[Voice Sentiment Classification]
    B4 --> C4[Gesture Recognition<br/>Threshold + Pixel Density]
    B5 --> C5[Time Context Analyzer]

    C1 --> D[Emotion Fusion Engine]
    C2 --> D
    C3 --> D
    C5 --> D

    D --> E[Unified Emotion State + Confidence]
    E --> F[Recommendation Engine]
    F --> G[Personalized Activity / Music Suggestions]
    G --> H[User Feedback Signal]
    H --> I[Reinforcement-Inspired Update]
    I --> F

    E --> J[Emotional Digital Twin Update]
    H --> J
    J --> K[Excel-Based Storage .xlsx]
    K --> L[Preprocessing + Analytics]
    L --> M[Energy Score + Trends + Anomaly Detection]
    M --> N[AI Coach Lifestyle Guidance]
    N --> O[User-Facing Dashboard / Report]
```

## 2) Emotion Fusion Model (Weighted Aggregation)

```mermaid
flowchart LR
    A1[Face Emotion + Confidence] --> W[Weighted Scoring]
    A2[Emoji Emotion + Confidence] --> W
    A3[Voice Emotion + Confidence] --> W
    A4[Time Context Emotion + Confidence] --> W

    W --> W1[Face Weight = 0.40]
    W --> W2[Emoji Weight = 0.30]
    W --> W3[Voice Weight = 0.20]
    W --> W4[Time Weight = 0.10]

    W1 --> S[Aggregate Vote Scores]
    W2 --> S
    W3 --> S
    W4 --> S

    S --> F1[Final Emotion = argmax(weighted votes)]
    S --> F2[Confidence = winning_weight / total_weight]
    S --> F3[Energy Index = signed weighted score]
```

## 3) Adaptive Recommendation Feedback Loop

```mermaid
flowchart TD
    A[Detected Fused Emotion] --> B[Select Candidate Recommendations]
    B --> C[Rank by Q-Value + Prior Bias]
    C --> D[Deliver Top Recommendation]
    D --> E[Capture User Feedback]
    E --> F{Reward Type}
    F -->|Positive| G[Strengthen Association]
    F -->|Negative| H[Penalize Association / Explore Alternative]
    G --> I[Q-value Update: Q <- Q + alpha*(r - Q)]
    H --> I
    I --> J[Updated Policy for Next Session]
    J --> B
```

## 4) Data Collection and Analysis Pipeline

```mermaid
flowchart TD
    A[Session Interaction Data] --> B[Store Structured Record]
    B --> C[Excel Sheet interactions]
    C --> D[Preprocessing]
    D --> D1[Null Handling]
    D --> D2[Emotion Label Normalization]
    D --> D3[Timestamp Standardization]
    D --> D4[User Filtering]

    D1 --> E[Analytics Engine]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F1[Mood Frequency Distribution]
    E --> F2[Trend + Engagement Analysis]
    E --> F3[Emotional Energy Score]
    E --> F4[Anomaly Detection]

    F1 --> G[Weekly Report Generation]
    F2 --> G
    F3 --> G
    F4 --> G
    G --> H[Dashboard Visualization + AI Coach Insights]
```
