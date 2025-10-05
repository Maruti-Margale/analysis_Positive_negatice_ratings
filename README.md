# analysis_Positive_negatice_ratings

demo link : https://analysispositivenegaticeratings-maruti.streamlit.app/

```mermaid
graph TD
    A[Start Application: app.py] --> B(Initialize: Load NLTK VADER Lexicon);
    B --> C(Initialize: Load Hugging Face Pipeline @st.cache_resource);

    subgraph User Interface
        D1(Display Title, Tabs, and Input Areas);
        C --> D1;
        D1 --> E{User Selects Tab};
    end

    subgraph Tab 1: Single Text Analysis
        E -- Single Text --> F[Text Input: Paste or Type Review];
        F --> G1{Perform VADER Analysis};
        G1 --> H1[Display VADER Scores and Compound Score];
        F --> G2{Hugging Face Pipeline Ready?};
        G2 -- Yes --> H2[Perform HF Analysis (Label + Score)];
        G2 -- No --> I2[Display HF Error/Warning];
        H1 & H2 --> J1[Display Single Review Results];
    end

    subgraph Tab 2: Dataset Batch Analysis
        E -- Dataset Batch --> F2[File Uploader: Upload CSV];
        F2 --> G3{File Uploaded?};
        G3 -- Yes --> H3[Display DataFrame Head];
        H3 --> I3[User Selects Text Column and Rating Column];
        I3 --> J3{Review Column Selected?};
        J3 -- Yes --> K3(Apply VADER to all rows @st.cache_data);
        K3 --> L3{Rating Column Selected?};
        L3 -- Yes --> M3[Group Data: Calculate Avg. Compound Score per Rating];
        L3 -- No --> I4[Display Warning: Select Rating Column];
        M3 --> N3[Generate and Display Altair Bar Chart];
        K3 & N3 --> O3[Display Batch Analysis Results Sample];
        J3 -- No --> I5[Display Warning: Select Text Column];
    end

    J1 & O3 --> P[End/Wait for Next User Interaction];
```
