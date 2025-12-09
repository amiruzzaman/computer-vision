# app.py
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rectangle Detector (Canvas → OpenCV)</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 20px; }
    h1 { font-size: 1.4rem; margin-bottom: 6px; }
    #canvasWrap { display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }
    #drawCanvas { border:2px solid #333; touch-action: none; background: #fff; }
    #controls { display:flex; gap:6px; margin-top:8px; }
    button { padding:8px 12px; border-radius:6px; border:1px solid #888; background:#f4f4f4; cursor:pointer; }
    button:active{ transform: translateY(1px); }
    #resultImg { max-width:480px; border:1px solid #ddd; display:block; margin-top:10px; }
    .math { background:#fbfbfb; border-left:3px solid #ccc; padding:8px; margin-top:10px; }
    #info { margin-top:10px; max-width:880px; }
    .coords { font-family: monospace; white-space:pre; background:#f7f7f7; padding:8px; border-radius:6px; }
    footer { margin-top:16px; font-size:0.9rem; color:#555; }
  </style>

  <!-- MathJax for rendering mathematics -->
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
  <h1>Rectangle Detector — Draw on the canvas and click <em>Detect</em></h1>

  <div id="canvasWrap">
    <div>
      <canvas id="drawCanvas" width="640" height="480"></canvas>
      <div id="controls">
        <button id="clearBtn">Clear</button>
        <button id="detectBtn">Detect</button>
        <button id="downloadBtn">Download PNG</button>
      </div>
      <div class="math">
        <strong>Mathematics (server-side, shown here):</strong>
        <div style="margin-top:6px">
          <p>
            We detect contours and approximate polygons using the Douglas–Peucker algorithm:
          </p>
          <p style="margin-left:12px">
            $$\text{approx} = \text{approxPolyDP}(\text{contour}, \varepsilon, \text{closed})$$
          </p>
          <p>
            Where we use \( \varepsilon = 0.02 \times \text{perimeter} \).
            For a candidate quadrilateral (4 vertices) we compute the angle between adjacent edges using the dot product:
          </p>
          <p style="margin-left:12px">
            For consecutive vertices \(A,B,C\) the angle at \(B\) is:
            $$\cos\theta = \dfrac{(A-B)\cdot(C-B)}{\|A-B\|\|C-B\|}, \quad \theta = \arccos(\cos\theta)$$
          </p>
          <p>
            We accept a rectangle if all four angles are close to \(90^\circ\) (by default within 15°).
          </p>
        </div>
      </div>
    </div>

    <div style="max-width:480px">
      <h3>Annotated result</h3>
      <img id="resultImg" src="" alt="Result will appear here" />
      <div id="info">
        <div><strong>Detected corners (image coordinates):</strong></div>
        <div id="coords" class="coords">No result yet.</div>
        <div id="stats" style="margin-top:8px;color:#333"></div>
      </div>
    </div>
  </div>

<script>
/* Drawing code */
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let drawing=false;
let lastX=0, lastY=0;
ctx.lineWidth = 6;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = '#000';

function getPointerPos(e){
  const rect = canvas.getBoundingClientRect();
  if (e.touches && e.touches.length) {
    return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
  } else {
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }
}
function start(e){
  e.preventDefault();
  drawing=true;
  const p = getPointerPos(e);
  lastX = p.x; lastY = p.y;
}
function move(e){
  if(!drawing) return;
  e.preventDefault();
  const p = getPointerPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x; lastY = p.y;
}
function stop(e){
  drawing=false;
}
canvas.addEventListener('mousedown', start);
canvas.addEventListener('touchstart', start);
canvas.addEventListener('mousemove', move);
canvas.addEventListener('touchmove', move,{passive:false});
canvas.addEventListener('mouseup', stop);
canvas.addEventListener('mouseout', stop);
canvas.addEventListener('touchend', stop);

/* buttons */
document.getElementById('clearBtn').addEventListener('click', ()=> {
  ctx.clearRect(0,0,canvas.width, canvas.height);
  document.getElementById('resultImg').src = '';
  document.getElementById('coords').textContent = 'No result yet.';
  document.getElementById('stats').textContent = '';
});
document.getElementById('downloadBtn').addEventListener('click', ()=> {
  const link = document.createElement('a');
  link.download = 'drawing.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
});

/* send to server */
document.getElementById('detectBtn').addEventListener('click', async ()=>{
  const dataURL = canvas.toDataURL('image/png');
  const resBox = document.getElementById('coords');
  resBox.textContent = 'Detecting...';
  try {
    const resp = await fetch('/detect', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({image: dataURL})
    });
    const data = await resp.json();
    if (data.success) {
      // show annotated image
      document.getElementById('resultImg').src = data.annotated_image;
      if (data.corners && data.corners.length) {
        const coordsText = data.corners.map((p,i)=>`P${i+1}: (${p[0].toFixed(1)}, ${p[1].toFixed(1)})`).join('\\n');
        resBox.textContent = coordsText;
        document.getElementById('stats').textContent = `Rectangularity score: ${data.rect_score.toFixed(3)} — angle deviations (deg): ${data.angle_deviations.map(a=>a.toFixed(1)).join(', ')}`;
      } else {
        resBox.textContent = 'No rectangle found.';
        document.getElementById('stats').textContent = data.message || '';
      }
    } else {
      resBox.textContent = 'Detection failed: ' + (data.message||'unknown error');
    }
  } catch (err) {
    resBox.textContent = 'Error: ' + err.message;
  }
});
</script>

<footer>
  Tip: draw a clear quadrilateral (four sides) — imperfections are tolerated. The server filters by contour area and angle closeness to 90°.
</footer>
</body>
</html>
"""

def data_uri_to_cv2_img(uri):
    """
    Convert a data URI (base64) to an OpenCV BGR image.
    """
    header, encoded = uri.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1].copy()  # RGB->BGR
    return arr

def cv2_img_to_data_uri(img_bgr):
    """
    Convert an OpenCV BGR image to a PNG data URI.
    """
    _, buffer = cv2.imencode('.png', img_bgr)
    b64 = base64.b64encode(buffer).decode('ascii')
    return 'data:image/png;base64,' + b64

def angle_between_vectors_deg(u, v):
    # returns angle in degrees between vectors u and v
    dot = np.dot(u, v)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu==0 or nv==0:
        return 0.0
    cos = np.clip(dot / (nu*nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify(success=False, message='No image provided'), 400
        img_bgr = data_uri_to_cv2_img(data['image'])
        h, w = img_bgr.shape[:2]

        # Preprocess: convert to gray, blur, threshold
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Because the user draws with black on white, invert for robustness
        # Use adaptive threshold to handle variable stroke thickness
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _,th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return jsonify(success=True, message='No contours found', corners=[],
                           annotated_image=cv2_img_to_data_uri(img_bgr))

        # Choose largest contour by area (but ignore tiny ones)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        chosen = None
        chosen_poly = None
        chosen_score = None
        chosen_angles = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # small noise -> skip; tweak threshold as needed
                continue
            peri = cv2.arcLength(cnt, True)
            eps = 0.02 * peri
            approx = cv2.approxPolyDP(cnt, eps, True)
            # If approx has 4 points, candidate quadrilateral
            if len(approx) == 4:
                # ensure polygon is in consistent order
                pts = approx.reshape(4,2).astype(np.float32)
                # compute angles at each vertex
                angles = []
                for i in range(4):
                    A = pts[(i-1)%4]; B = pts[i]; C = pts[(i+1)%4]
                    v1 = A - B
                    v2 = C - B
                    ang = angle_between_vectors_deg(v1, v2)
                    angles.append(ang)
                # score how close angles are to 90 deg
                deviations = [abs(a - 90.0) for a in angles]
                rect_score = sum(deviations) / 4.0  # lower is better
                # Accept candidate if average deviation within threshold
                if rect_score < 15.0:
                    chosen = cnt
                    chosen_poly = pts
                    chosen_score = rect_score
                    chosen_angles = deviations
                    break
        # If no approx4 found, optionally try convex hull and approx again
        if chosen is None:
            # fallback: try convex hull of the largest contour and approximate
            cnt = contours[0]
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            eps = 0.02 * peri
            approx = cv2.approxPolyDP(hull, eps, True)
            if len(approx) == 4:
                pts = approx.reshape(4,2).astype(np.float32)
                angles = []
                for i in range(4):
                    A = pts[(i-1)%4]; B = pts[i]; C = pts[(i+1)%4]
                    v1 = A - B
                    v2 = C - B
                    ang = angle_between_vectors_deg(v1, v2)
                    angles.append(ang)
                deviations = [abs(a - 90.0) for a in angles]
                rect_score = sum(deviations) / 4.0
                if rect_score < 20.0:
                    chosen = hull
                    chosen_poly = pts
                    chosen_score = rect_score
                    chosen_angles = deviations

        annotated = img_bgr.copy()
        response = dict(success=True, corners=[], annotated_image=None, rect_score=0.0, angle_deviations=[])
        if chosen_poly is not None:
            # Order points in consistent order (clockwise starting from top-left)
            # Use OpenCV's ordering by sum/diff trick
            pts = chosen_poly
            s = pts.sum(axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1).reshape(-1)
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            ordered = np.array([tl, tr, br, bl], dtype=np.float32)

            # Draw polygon and points
            pts_int = ordered.reshape(-1,1,2).astype(int)
            cv2.polylines(annotated, [pts_int], True, (0,255,0), 3)
            for i,(x,y) in enumerate(ordered):
                cv2.circle(annotated, (int(x),int(y)), 6, (0,0,255), -1)
                cv2.putText(annotated, f"P{i+1}", (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            response['corners'] = ordered.tolist()
            response['annotated_image'] = cv2_img_to_data_uri(annotated)
            response['rect_score'] = float(chosen_score)
            response['angle_deviations'] = [float(x) for x in chosen_angles]
            response['message'] = 'Quadrilateral found'
            return jsonify(response)
        else:
            # No rectangle found: return image of contours for debugging
            debug_img = img_bgr.copy()
            cv2.drawContours(debug_img, contours[:3], -1, (255,0,0), 2)
            response['annotated_image'] = cv2_img_to_data_uri(debug_img)
            response['message'] = 'No quadrilateral detected'
            return jsonify(response)

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

if __name__ == '__main__':
    # Run with: python app.py
    app.run(debug=True)
