import os
import uuid
import cv2
from flask import Flask, request, send_from_directory, render_template_string

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Black Rectangle (Phone) Detector</title>
    <style>
        body {
            font-family: Arial;
            background: #f7f7f7;
            padding: 30px;
            text-align: center;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            max-width: 650px;
            margin: auto;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.15);
        }
        input[type=file] { margin: 20px; }
        button {
            padding: 10px 20px;
            border: none;
            background: #2196f3;
            color: white;
            cursor: pointer;
            border-radius: 6px;
        }
        img {
            width: 90%;
            max-width: 500px;
            margin-top: 20px;
            border-radius: 10px;
        }
        #preview { display: none; }
    </style>
</head>
<body>

<div class="card">
    <h2>Black Rectangular Object Detector</h2>
    <p>Select an image. It will detect a black rectangular shape (your phone) and draw a green box.</p>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" onchange="previewImage(event)" required>
        <br>
        <button type="submit">Upload & Detect</button>
    </form>

    <img id="preview">

    {% if original %}
        <h3>Original Image</h3>
        <img src="/uploads/{{ original }}">

        <h3>Detection Result</h3>
        <img src="/uploads/{{ result }}">
    {% endif %}
</div>

<script>
function previewImage(event) {
    var img = document.getElementById("preview");
    img.src = URL.createObjectURL(event.target.files[0]);
    img.style.display = "block";
}
</script>

</body>
</html>
"""


def detect_black_phone(image_path, output_path):
    """Detect a black rectangular shape (like a phone)."""

    img = cv2.imread(image_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold: detect dark regions
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:  # rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # heuristic: phones are tall rectangles
            if h > w * 1.1 and w > 40 and h > 100:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(img, "Phone", (x, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

    cv2.imwrite(output_path, img)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files.get("image")
        if not img_file:
            return render_template_string(HTML_PAGE)

        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(path)

        out_name = "out_" + filename
        out_path = os.path.join(UPLOAD_FOLDER, out_name)

        detect_black_phone(path, out_path)

        return render_template_string(
            HTML_PAGE,
            original=filename,
            result=out_name
        )

    return render_template_string(HTML_PAGE)


@app.route("/uploads/<filename>")
def uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)

# pip install flask opencv-python