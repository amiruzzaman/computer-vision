# Homework: Deep Learning - Dogs vs. Cats Classification

## 🎯 Objective
In this assignment, you will step into the shoes of a machine learning engineer. You will find an existing **Convolutional Neural Network (CNN)** architecture on GitHub, download a massive dataset from **Kaggle**, and train the model to distinguish between images of cats and dogs.

---

## 🛠 Step 1: Data Acquisition (Kaggle)
You will use the famous "Dogs vs. Cats" dataset. To download it:

1.  **Create a Kaggle Account:** If you don't have one, sign up at [Kaggle.com](https://www.kaggle.com).
2.  **Generate API Token:** Go to your account settings, scroll to the "API" section, and click **"Create New API Token"**. This downloads a `kaggle.json` file.
3.  **Download via Python:** Use the following commands in your notebook/script:
    ```bash
    pip install kaggle
    ```
    ```python
    import os
    os.environ['KAGGLE_USERNAME'] = "your_username"
    os.environ['KAGGLE_KEY'] = "your_api_key"
    !kaggle competitions download -c dogs-vs-cats
    ```
4.  **Unzip the Data:** Ensure you extract the `train.zip` and `test.zip` files to a folder in your project directory.

---

## 🚀 Step 2: Model Selection & Training
1.  **Find a Model:** Search GitHub for a "Cats and Dogs CNN Python" repository. You are encouraged to use established architectures (like a simple Custom CNN, VGG16, or ResNet). 
2.  **Training:** You do **not** need to train for 50+ epochs. Aim for **3–5 epochs** just to demonstrate that your training loop is functional.
3.  **Metrics:** Your code must calculate and print the following during or after training:
    * **Accuracy:** Overall correctness.
    * **Loss:** The "error" value the model is minimizing.
    * **Precision:** How many predicted dogs were actually dogs?
    * **Recall:** How many actual dogs did the model find?

---

## 📝 Deliverables (The Submission)
You must submit two things:

### 1. GitHub Repository
Upload all your code (`.py` or `.ipynb`) to a public GitHub repository. Include a `requirements.txt` file listing your dependencies (e.g., `tensorflow`, `torch`, `opencv-python`).

### 2. Written Report (PDF)
A short report (2–3 pages) containing:
* **Model Source:** A link to the GitHub repository you used as a base.
* **Code Snippets:** Show your data loading logic and your training loop.
* **Results Table:** A table showing your final Accuracy, Loss, Recall, and Precision. 
* **Discussion:** * Even if your accuracy is low (e.g., 50-60%), explain **why**. Was it the low number of epochs? Small dataset size?
    * Include one "prediction" image: Show an image the model got **correct** and one it got **wrong**.

---

## 💡 Pro-Tip
If you are using Google Colab, remember to enable the **GPU Accelerator** (Runtime > Change runtime type > T4 GPU) to speed up your training significantly!