# Homework: The Digital Alchemist (Light, Shadow, & Color)

## 🎯 Objective
In this assignment, you will explore how computers interpret visual data using the **HSV (Hue, Saturation, Value)** color space. You will write a Python script that identifies environmental lighting (shadows and highlights) and tracks a specific object based on its color.

---

## 🛠 Setup Instructions

1.  **Install Python:** Ensure you have Python 3.x installed.
2.  **Install Libraries:** Open your terminal or command prompt and run:
    ```bash
    pip install opencv-python numpy
    ```
3.  **Download the Starter Code:** Save the provided `explorer.py` script to your project folder.

---

## 📖 Background: Understanding HSV
Unlike RGB (Red, Green, Blue), the **HSV** model is much closer to how humans perceive color:
* **Hue (H):** The "type" of color (Red, Green, Blue, etc.).
* **Saturation (S):** The "vibrancy" (grayish vs. neon).
* **Value (V):** The "brightness" (shadow vs. light).



By looking at the **Value** channel, we can easily isolate shadows (low value) and highlights (high value) regardless of what color the objects actually are.

---

## 🚀 Your Tasks

### 1. Run the Starter Code
Run your script and observe the webcam feed. 
* **Shadows** should appear with a **Purple** overlay.
* **Highlights** should appear with a **Yellow** overlay.

### 2. Calibrate the Light
Modify the `upper_shadow` and `lower_light` variables in the code. 
* Try to make it so that only the truly dark corners of your room are purple.
* Try to make it so only a direct flashlight or a bright window is yellow.

### 3. Implement Color Tracking
Find a vibrant object (like a bright blue pen or a red cup). Your goal is to make the code "find" this object. 
* Define `lower_color` and `upper_color` using HSV ranges.
* **Tip:** Use the video resource below to learn how to find the exact HSV coordinates for your object.

---

## 📺 Helpful Resources
* **Video Guide:** [OpenCV Color Detection in Python](https://www.youtube.com/watch?v=Kg6Y6YIDZ-JpY) - This will help you understand how to define color ranges.
* **Documentation:** [OpenCV Thresholding Basics](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)

---

## 📝 Deliverables (The Report)
Submit a PDF report containing the following sections:

1.  **Code Snippet:** A copy of the code you wrote to define your custom color range.
2.  **Visual Results:** * **Screenshot A:** The webcam view showing shadows and highlights being detected.
    * **Screenshot B:** The webcam view successfully "masking" or highlighting your chosen colored object.
3.  **Discussion:**
    * What happened to the color detection when you moved your object into a dark shadow? 
    * Explain why using **HSV** is better for tracking a red ball than using **RGB**.