# Real-time Cyberattack Detection System using Artificial Intelligence

## Project Overview

This project proposes an advanced, real-time cyberattack detection system leveraging Artificial Intelligence (AI) to address the escalating threats in the digital era. Traditional signature-based detection systems often exhibit high false positive rates (18-25%) and struggle with evolving threats, as 60% of modern attacks utilize polymorphic techniques to evade conventional signatures[cite: 4, 5].

Our solution integrates a hybrid approach, combining a comprehensive dataset, explainable AI models, an edge-cloud architecture, and adversarial defense mechanisms to provide transparent, scalable, and effective threat detection.

## Key Features

* **Comprehensive Dataset:** Utilizes the CIC-IDS2017 dataset, known for its diversity and representation of 14 types of attacks and benign traffic, with 2.8 million records and 80 features per flow[cite: 18, 19].
* **Explainable AI (XAI):** Incorporates XAI techniques like SHAP and LIME to enhance trust in model decisions, identify data biases, and facilitate compliance with regulations like GDPR[cite: 7, 10, 11, 12, 13].
* **Hybrid Architecture:** Employs a three-layered system:
    * **Acquisition Layer:** Uses Suricata and Zeek for flow generation, supporting NetFlow v9 and IPFIX protocols at a 1:1 sampling rate[cite: 24, 25].
    * **Processing Layer:**
        * **Edge Computing:** Utilizes an optimized Random Forest model (100 trees) on NVIDIA Jetson Nano hardware for low-latency (<50 ms) processing of flows[cite: 17, 26].
        * **Cloud Computing:** Deploys a bidirectional LSTM model on AWS EC2 (p3.2xlarge with Tesla V100 GPU) for temporal analysis[cite: 27].
    * **Visualization Layer:** Presents real-time alerts, geographical attack distribution, and historical trends via a Kibana dashboard, integrated with Jira Service Management for incident escalation[cite: 28, 29].
* **Robust Preprocessing:** Includes min-max scaling, one-hot encoding, SMOTE-Tomek for class balancing, and ANOVA for feature selection[cite: 21, 22].
* **Advanced Model Evaluation:** Employs various metrics such as accuracy, precision, recall, F1-score, classification reports, and confusion matrices. Cross-validation (10-fold) is also used to assess model stability[cite: 23].

## Dataset

The project utilizes the **CIC-IDS2017** dataset. This dataset is highly validated, having been used in over 120 academic investigations since 2017[cite: 20]. It includes diverse attack types such as DDoS, SQLi, and port scans, alongside benign traffic.

## Methodology

### Data Preprocessing

The raw dataset undergoes several preprocessing steps:
1.  **Column Cleaning:** Stripping whitespace from column names.
2.  **Null Value Handling:** Identifying and addressing missing values.
3.  **Categorical Encoding:** One-hot encoding for categorical features.
4.  **Infinite Values:** Replacing infinite values with NaN and then dropping rows with NaN.
5.  **Data Scaling:** Applying `StandardScaler` to numerical features for optimal model performance.

### Model Training and Evaluation

Two primary machine learning models are used:

#### Random Forest Classifier

* **Configuration:** 100 estimators, `random_state=42`.
* **Training:** Trained on `X_train` and `y_train` subsets.
* **Evaluation:** Performance is assessed using:
    * Accuracy
    * Classification Report (Precision, Recall, F1-Score)
    * Confusion Matrix
    * 10-fold Cross-validation accuracy scores.

#### LSTM (Long Short-Term Memory) Neural Network

* **Data Preparation:** Reshaping input data for LSTM compatibility (`X_train_lstm`, `X_test_lstm`).
* **Architecture:**
    * An LSTM layer with 64 units.
    * Dropout layer (0.3).
    * Dense layer with 32 units and 'relu' activation.
    * Output Dense layer with 1 unit and 'sigmoid' activation for binary classification.
* **Compilation:** `loss='binary_crossentropy'`, `optimizer='adam'`, `metrics=['accuracy']`.
* **Training:** Trained with `epochs=10`, `batch_size=64`, `validation_split=0.2`, and `EarlyStopping` callback.
* **Evaluation:** Performance is assessed using:
    * Accuracy
    * Classification Report (Precision, Recall, F1-Score)
    * Confusion Matrix

## Performance Comparison

A comparative analysis of the Random Forest and LSTM models is performed based on:

* Accuracy
* Precision
* Recall
* F1-Score

The results are visualized using a bar chart to highlight the performance differences between the two models.

## Installation and Usage

To run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (A `requirements.txt` file would typically contain `kagglehub`, `pandas`, `scikit-learn`, `numpy`, `tensorflow`, `matplotlib`).

3.  **Download the dataset:** The script automatically downloads the "chethuhn/network-intrusion-dataset" from Kaggle Hub.

4.  **Run the script:**
    ```bash
    python proyectotitulo1.py
    ```

## Limitations and Future Work

### Technical Limitations

* **Data Dependence:** The system's effectiveness relies heavily on the quality and timeliness of training data[cite: 30].
* **Hardware Requirements:** Complex model processing necessitates significant hardware resources[cite: 30].
* **Privacy Challenges:** Comprehensive network flow analysis presents privacy concerns[cite: 30].

### Potential Impact

* **Reduced Detection/Response Times:** Significant decrease in the time required to detect and respond to cyber threats[cite: 31].
* **Enhanced Adaptability:** Greater flexibility in addressing emerging threats[cite: 31].
* **Algorithmic Transparency Compliance:** Adherence to regulations concerning algorithmic transparency[cite: 31].

## References

[1] Universidad Andrés Bello. Portafolio De Proyectos. [cite: 1]
[2] "Desarrollo de un Sistema Avanzado de Detección de Ataques Cibernéticos en Tiempo Real mediante Inteligencia Artificial: Un Enfoque Integral basado en el Dataset CIC-IDS 2017". [cite: 2]
[3] IBM Security. "Informe Global de Amenazas 2025". [cite: 3]
[4] Nagaraju, B. (2025). "Next-Generation Cybersecurity Systems: Integrating AI, IoT and Cloud Computing". *International Journal of Multidisciplinary Engineering*, 12(3), 112-130. [cite: 4, 36, 37]
[5] Lazim, S., & Ali, Q. I. (2025). "Machine Learning-Based IDPS for Critical IIoT Infrastructure: A Performance Analysis". *IEEE Transactions on Industrial Informatics*, 21(3), 1456-1472. [cite: 5, 15, 33, 34]
[6] This work proposes a hybrid real-time detection system. [cite: 6]
[7] Dataset CIC-IDS2017: For its representativeness and diversity of attacks. [cite: 7]
[8] Explainable AI models (XAI): For transparency in decision making. [cite: 8]
[9] Mechanisms of defense against adversarial attacks: Based on adversarial training. [cite: 9]
[10] Mohale, V. Z., & Obagbuwa, I. C. (2025). "Explainable Artificial Intelligence in Cybersecurity: A Meta-Analysis of 150 Case Studies". *Frontiers in Artificial Intelligence*, 8(2), 1-18. [cite: 10, 35]
[11] Allows identifying biases in training data. [cite: 11]
[12] Facilitates compliance with regulations like GDPR (Article 22: "Right to explanation"). [cite: 12]
[13] SHAP (Shapley Additive Explanations): Quantifies the contribution of each feature in the prediction. LIME (Local Interpretable Model-agnostic Explanations): Generates local explanations for specific instances. [cite: 13]
[14] Challenges in IoT/IIoT Environments. [cite: 14]
[15] Limited resource devices: Require light models (<100 MB of memory). [cite: 15]
[16] Proposed solution: Use of optimized Random Forest (maximum depth: 15 nodes) for balance between precision and consumption. [cite: 16]
[17] Implementation in edge layer with NVIDIA Jetson Nano devices. [cite: 17]
[18] Selection of Dataset: CIC-IDS2017. [cite: 18]
[19] Structure: 2.8 million records with 80 features per flow (e.g., duration, protocol, bytes sent). [cite: 19]
[20] Validation: Used in 120+ academic investigations since 2017 (Canadian Institute for Cybersecurity, 2017). [cite: 20]
[21] Normalization: Min-max scaling (0-1) for numerical features. Encoding: One-hot for categorical variables (e.g., TCP/UDP protocols). [cite: 21]
[22] Balancing: SMOTE-Tomek technique to balance minority classes (1:1 ratio). Feature selection: ANOVA (p-value <0.05) to eliminate noise. [cite: 22]
[23] From sklearn.ensemble import RandomForestClassifier, from sklearn.model_selection import train_test_split, from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, from sklearn.model_selection import cross_val_score. [cite: 23]
[24] Tools: Suricata (for flow generation) + Zeek (protocol analysis). Protocols supported: NetFlow v9, IPFIX. [cite: 24]
[25] Sampling rate: 1:1 (no packet loss). [cite: 25]
[26] Model: Random Forest (Scikit-learn) with 100 trees. Hardware: NVIDIA Jetson Nano (4GB RAM). Latency: <50 ms per flow. [cite: 26]
[27] Model: Bidirectional LSTM (TensorFlow) for temporal analysis. Environment: AWS EC2 (p3.2xlarge instance with Tesla V100 GPU). [cite: 27]
[28] Dashboard: Kibana with panels for: Real-time alerts, Geographical distribution of attacks (heatmap), Historical trends (time series). [cite: 28]
[29] Integration: Ticket system (Jira Service Management) to escalate incidents. [cite: 29]
[30] Technical Limitations: Dependence on the quality and updating of training data, Hardware requirements for processing complex models, Privacy challenges in the complete analysis of network flows. [cite: 30]
[31] Potential Impact: Significant reduction in detection and response times, Greater adaptability to emerging threats, Compliance with algorithmic transparency regulations. [cite: 31]
[32] Canadian Institute for Cybersecurity. (2017). CICIDS2017 Dataset: A Comprehensive Resource for Network Intrusion Detection Research. University of New Brunswick. https://www.unb.ca/cic/datasets/ids-2017.html. [cite: 32]
[33] Lazim, S., & Ali, Q. I. (2025). Machine Learning-Based IDPS for Critical IIoT Infrastructure: A Performance Analysis. IEEE Transactions on Industrial Informatics, 21(3), 1456-1472. https://doi.org/10.1109/TII.2025.123456. [cite: 33]
[34] IEEE Transactions on Industrial Informatics, 21(3), 1456-1472. [cite: 34]
[35] Mohale, V. Z., & Obagbuwa, I. C. (2025). Explainable Artificial Intelligence in Cybersecurity: A Meta-Analysis of 150 Case Studies. Frontiers in Artificial Intelligence, 8(2), 1-18. https://doi.org/10.3389/frai.2025.00123. [cite: 35]
[36] Nagaraju, B. (2025). Next-Generation Cybersecurity Systems: Integrating AI, IoT and Cloud Computing. International Journal of Multidisciplinary Engineering, 12(3), 112-130. [cite: 36]
[37] https://doi.org/10.1080/12345678.2025.1234567. [cite: 37]
[38] National Institute of Standards and Technology. (2024). Framework for Improving Critical Infrastructure Cybersecurity, Version 2.0. NIST Special Publication 800-82. https://doi.org/10.6028/NIST.SP.800-82r2. [cite: 38]
[39] Xiaofeng Chen et al. - Springer, 2022. [cite: 39]
[40] Applied Cyber Security and the Smart Grid Eric D. Knapp & Raj Samani - Elsevier, 2013. [cite: 40]
[41] Cybersecurity and Artificial Intelligence: Challenges and Opportunities Hossein Hassani et al. [cite: 41]
[42] - Springer, 2022. [cite: 42]
[43] Tang, T., et al. - IEEE Transactions on Industrial Informatics, 2020. [cite: 43]
[44] Q. Duan et al. - Journal of Cybersecurity and Privacy, 2021. [cite: 44]
[45] C. Modi et al. - Computer Communications, 2019. [cite: 45]
[46] Kim, G. et al. - Computers & Security, 2021. [cite: 46]
[47] Yin, C. et al. - IEEE Access, 2017. [cite: 47]
[48] Thakkar, A. & Lohiya, R. - Computer Science Review, 2020. [cite: 48]
