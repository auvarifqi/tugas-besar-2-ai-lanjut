# Data Science - Tugas Besar 2 AI Lanjut

<p align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=747B2E&center=true&vCenter=true&width=700&lines=Data-Science;Tugas+Besar+IF5150" alt="Typing SVG" />
  </a>
</p>

## **Author**
<table align="center">
    <tr>
        <td colspan=4 align="center">Kelompok 8</td>
    </tr>
    <tr>
        <td>No.</td>
        <td>Nama</td>
        <td>NIM</td>
        <td>Email</td>
    </tr>
    <tr>
        <td>1.</td>
        <td>Ahmad Rizki</td>
        <td>18221071</td>
        <td><a href="mailto:18221071@std.stei.itb.ac.id">18221071@std.stei.itb.ac.id</a></td>
    </tr>
    <tr>
        <td>2.</td>
        <td>Auvarifqi</td>
        <td>18221060</td>
        <td><a href="mailto:18221060@std.stei.itb.ac.id">18221060@std.stei.itb.ac.id</a></td>
    </tr>
</table>

---

## **Project Description**

This project is a part of Tugas Besar 2 for the course **IF5150 - Inteligensi Artifisial Lanjut** at Institut Teknologi Bandung. The goal of this project is to utilize **Supervised** and **Unsupervised Learning** methods to analyze electric consumption data of households in Melbourne. The focus lies in detecting households that own **Electric Vehicles (EVs)** and clustering households based on consumption patterns.

Key features of this project include:
- Implementation of binary classification models (e.g., Logistic Regression, KNN, Decision Tree).
- Clustering techniques like K-Means and DBSCAN.
- Advanced preprocessing steps such as dimensionality reduction using PCA.
- Ensemble methods to improve classification performance.
- Comprehensive EDA and data visualization.

---

## **Table of Contents**
- [Data Science - Tugas Besar 2 AI Lanjut](#data-science---tugas-besar-2-ai-lanjut)
  - [**Author**](#author)
  - [**Project Description**](#project-description)
  - [**Table of Contents**](#table-of-contents)
  - [**Features**](#features)
  - [**Project Structure**](#project-structure)
  - [**How to Run**](#how-to-run)
  - [**Requirements**](#requirements)
  - [**Usage Guide**](#usage-guide)
  - [**Contributors**](#contributors)
  - [**License**](#license)

---

## **Features**

- **Exploratory Data Analysis (EDA)**: 
  Gain insights from data using statistical and visualization techniques.
- **Supervised Learning**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Ensemble Voting
- **Unsupervised Learning**:
  - K-Means Clustering
  - DBSCAN (Density-Based Spatial Clustering)
- **Clustering Visualization**:
  Interactive 3D cluster visualization for better pattern understanding.

---

## **Project Structure**

```
tugas-besar-2-ai-lanjut/
├── README.md
├── data/
│   └── EV_data.csv
├── main.ipynb
├── requirements.txt
└── utils/
    ├── classification_models/
    │   ├── logistic_regression.py
    │   ├── k_nearest_neighbor.py
    │   ├── decision_tree.py
    │   └── ensemble_voting.py
    ├── clustering_models/
    │   ├── k_means.py
    │   └── dbscan.py
    ├── eda_visualization.py
    ├── eda_statistics.py
    └── preprocessing.py
```

---

## **How to Run**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/auvarifqi/tugas-besar-2-ai-lanjut.git
   cd tugas-besar-2-ai-lanjut
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or later installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Launch the Jupyter Notebook for detailed workflows:
   ```bash
   jupyter notebook main.ipynb
   ```

---

## **Requirements**

- Python 3.8+
- Libraries:
  - Pandas
  - Numpy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Plotly

Install all dependencies using the provided `requirements.txt`.

---

## **Usage Guide**

1. **Exploratory Data Analysis (EDA)**:
   - Run `main.ipynb` to explore data distributions and correlations.
   - Visualize target balance, correlation matrices, and cluster distributions.

2. **Supervised Learning**:
   - Train and evaluate Logistic Regression, KNN, and Decision Tree models.
   - Use Ensemble Voting to improve predictions.

3. **Unsupervised Learning**:
   - Perform clustering using K-Means and DBSCAN.
   - Visualize clusters and analyze their characteristics.

4. **Interactive Visualization**:
   - View clustering results interactively with 3D plots.

5. **Model Evaluation**:
   - Compare models using accuracy, precision, recall, and F1-Score metrics.

---

## **Contributors**
- **Ahmad Rizki** (18221071) | [Email](mailto:18221071@std.stei.itb.ac.id)
- **Auvarifqi** (18221060) | [Email](mailto:18221060@std.stei.itb.ac.id)

---

## **License**

This project is developed as part of academic coursework and is not intended for commercial use.