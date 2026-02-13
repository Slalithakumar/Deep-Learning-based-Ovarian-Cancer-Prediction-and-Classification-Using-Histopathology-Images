# -------------------------------------------------------------------------
# FILE: app.py (Updated — auto-enhance + camera redirect + shared processing)
# -------------------------------------------------------------------------

import os
import io
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ---------- CONFIG ----------
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
VALIDATION_FOLDER = 'validation_samples'
MODEL_FOLDER = 'model'
MODEL_FILENAME = 'ovarian_cancer_best_inceptionv3.h5'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
# You can tune this back to 0.72 if you'd like stricter validation
THRESHOLD_COSINE = 0.50

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'change-me'

# ---------- LOAD MODELS ----------
print("Loading models...")
pred_model = None
model_path = os.path.join(MODEL_FOLDER, MODEL_FILENAME)
if os.path.exists(model_path):
    try:
        pred_model = tf.keras.models.load_model(model_path)
        print("Loaded prediction model")
    except Exception as e:
        print("Failed to load prediction model:", e)
else:
    print("Prediction model missing at", model_path)

# VGG16 for validator
try:
    base_vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
    print("Loaded VGG16 validator")
except Exception as e:
    base_vgg = None
    print("Failed to load VGG16 validator:", e)

# ---------- UTILITIES ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_inception(pil_img):
    """Return shape (1,299,299,3) float32 preprocessed tensor."""
    img = pil_img.resize((299, 299))
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    return np.expand_dims(arr, 0)

def compute_vgg_vector(pil_img):
    """Return 1D embedding from VGG16 pooling output."""
    if base_vgg is None:
        raise RuntimeError("VGG validator is not available")
    img = pil_img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    vec = base_vgg.predict(arr)
    return vec.reshape(-1)

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

# ---------- LOAD REFERENCE EMBEDDINGS ----------
ref_vectors, ref_paths = [], []
if os.path.isdir(VALIDATION_FOLDER) and base_vgg is not None:
    for fn in os.listdir(VALIDATION_FOLDER):
        if allowed_file(fn):
            path = os.path.join(VALIDATION_FOLDER, fn)
            try:
                img = Image.open(path).convert('RGB')
                ref_vectors.append(compute_vgg_vector(img))
                ref_paths.append(path)
                print("Loaded reference:", path)
            except Exception as e:
                print("Failed to load reference", path, ":", e)

if len(ref_vectors) == 0:
    print("No validation references found — validator will be permissive")

LABELS = ["Clear_Cell", "Endometri", "Mucinous", "Serous", "Non_Cancerous"]

# ---------- IMAGE ENHANCEMENT ----------
def apply_enhancements(pil_img, do_sharpen=True, brightness_factor=1.15, autocontrast=True):
    """
    Apply mild auto-contrast, brightness boost, and sharpening to improve
    camera-captured images. Returns a new PIL Image (RGB).
    """
    try:
        img = pil_img.convert("RGB")
        if autocontrast:
            img = ImageOps.autocontrast(img, cutoff=1)  # remove tiny extremes
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
        if do_sharpen:
            # Unsharp mask with mild parameters
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        return img
    except Exception as e:
        print("Enhancement failed:", e)
        return pil_img

# ---------- GRAD-CAM ----------
def generate_gradcam_overlay(img_pil, preprocessed_tensor, model, intensity=0.45):
    """Return a PIL RGB image overlay (or None on failure)."""
    try:
        # find a good conv layer
        layer_name = None
        for layer in reversed(model.layers):
            # prefer inception 'mixed' blocks, else any Conv2D
            if "mixed" in getattr(layer, "name", ""):
                layer_name = layer.name
                break
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        if layer_name is None:
            print("No conv layer found for Grad-CAM")
            return None

        last_conv = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_tensor)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print("Gradients are None — cannot build Grad-CAM")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()     # H x W x C
        pooled_grads = pooled_grads.numpy()        # C

        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i in range(pooled_grads.shape[-1]):
            cam += pooled_grads[i] * conv_outputs[:, :, i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)
        else:
            return None

        # create heatmap RGBA and resize
        heatmap = (cm.get_cmap('jet')(cam)[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap).resize(img_pil.size, resample=Image.BILINEAR)
        overlay = Image.blend(img_pil.convert("RGB"), heatmap_pil, alpha=float(intensity))
        return overlay

    except Exception as e:
        print("GradCAM error:", e)
        return None

# ---------- PDF ----------
def create_pdf_bytes(filename, pred_label, probs, overlay_img=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 60, "Ovarian Cancer Prediction Report")

    c.setFont("Helvetica", 11)
    c.drawString(40, h - 90, f"File: {filename}")
    c.drawString(40, h - 110, f"Prediction: {pred_label}")
    c.drawString(40, h - 130, f"Date: {datetime.now()}")

    y = h - 160
    for k, v in probs.items():
        c.drawString(40, y, f'{k}: {v:.6f}')
        y -= 15

    if overlay_img is not None:
        try:
            temp = os.path.join(RESULT_FOLDER, "_overlay.png")
            overlay_img.save(temp)
            c.drawImage(temp, 40, 40, width=220, height=220)
        except Exception as e:
            print("PDF overlay write failed:", e)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# ---------- IMAGE PROCESSING COMMON FUNCTION ----------
def process_image(pil_img, filename, enhance=True):
    """
    Centralized processing: enhancement (optional) -> validation -> prediction -> gradcam -> pdf.
    Returns dict with keys: valid(bool), message/score, label, probs, overlay_b64, pdf_b64.
    """
    result = {"valid": False, "distance": 0.0, "threshold": THRESHOLD_COSINE, "label": None, "probs": None,
              "overlay_b64": None, "pdf_b64": None, "error": None}

    try:
        # Enhancement for camera/low-quality images
        if enhance:
            pil_img = apply_enhancements(pil_img)

        # Validation
        if ref_vectors and base_vgg is not None:
            try:
                vec = compute_vgg_vector(pil_img)
                sims = [cosine_similarity(vec, rv) for rv in ref_vectors]
                sim_score = float(np.max(sims))
                result["distance"] = sim_score
                if sim_score < THRESHOLD_COSINE:
                    result["valid"] = False
                    return result
            except Exception as e:
                result["error"] = f"Validation error: {e}"
                result["valid"] = False
                return result

        # Prediction
        if pred_model is None:
            result["error"] = "Prediction model not loaded"
            return result

        preproc = preprocess_inception(pil_img)
        preds = pred_model.predict(preproc)[0]
        top_idx = int(np.argmax(preds))
        top_label = LABELS[top_idx]
        probs = {LABELS[i]: float(preds[i]) for i in range(len(LABELS))}

        # Grad-CAM overlay
        overlay = generate_gradcam_overlay(pil_img, preproc, pred_model)

        # PDF
        pdf_buf = create_pdf_bytes(filename, top_label, probs, overlay)
        pdf_b64 = base64.b64encode(pdf_buf.getvalue()).decode()

        overlay_b64 = None
        if overlay is not None:
            b = io.BytesIO()
            overlay.save(b, format="PNG")
            overlay_b64 = base64.b64encode(b.getvalue()).decode()

        result.update({
            "valid": True,
            "label": top_label,
            "probs": probs,
            "overlay_b64": overlay_b64,
            "pdf_b64": pdf_b64
        })
        return result

    except Exception as e:
        result["error"] = f"Processing exception: {e}"
        print("process_image exception:", e)
        return result

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if 'file' not in request.files:
            flash("No file uploaded")
            return redirect(url_for("index"))

        file = request.files['file']
        if file.filename == "":
            flash("No file selected")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Unsupported file type")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        pil_img = Image.open(save_path).convert("RGB")

        # Process (enhance=True by default)
        res = process_image(pil_img, filename, enhance=True)

        if not res.get("valid"):
            # show the 'not histopathology' result page
            return render_template("result.html", valid=False, distance=res.get("distance", 0.0), threshold=THRESHOLD_COSINE, error=res.get("error"))

        # Save overlay image (optional) so you can reference it statically
        if res.get("overlay_b64"):
            try:
                overlay_bytes = base64.b64decode(res["overlay_b64"])
                with open(os.path.join(RESULT_FOLDER, f"overlay_{filename}.png"), "wb") as f:
                    f.write(overlay_bytes)
            except Exception:
                pass

        # history append (simple)
        try:
            with open('history.csv', 'a') as fh:
                fh.write(f"{datetime.now().isoformat()},{filename},{res['label']},{res['probs'][res['label']]:.6f}\n")
        except Exception:
            pass

        return render_template("result.html",
                               valid=True,
                               filename=filename,
                               label=res["label"],
                               probs=res["probs"],
                               overlay=res["overlay_b64"],
                               pdf_b64=res["pdf_b64"])

    except Exception as e:
        print("Analyze route error:", e)
        flash("Error processing the image. See server logs.")
        return redirect(url_for("index"))

# -------------------------------------------------------------------------
# NEW LIVE CAMERA SCANNING ROUTE (returns redirect + JSON)
# -------------------------------------------------------------------------
@app.route("/scan_camera", methods=["POST"])
def scan_camera():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image received"}), 400

        base64_img = data["image"]
        # remove data URL prefix if present
        if "," in base64_img:
            base64_img = base64_img.split(",", 1)[1]

        # fix padding if necessary
        try:
            img_bytes = base64.b64decode(base64_img + "===")
        except Exception:
            return jsonify({"error": "Invalid base64 image data"}), 400

        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print("Failed to open image from camera data:", e)
            return jsonify({"error": "Could not decode image"}), 400

        # Save temp file
        filename = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            pil_img.save(save_path)
        except Exception:
            pass

        # Process (enhance=True for camera)
        res = process_image(pil_img, filename, enhance=True)

        if not res.get("valid"):
            # return JSON explaining invalid image — frontend can show alert
            return jsonify({"valid": False, "distance": res.get("distance", 0.0), "threshold": THRESHOLD_COSINE, "error": res.get("error")})

        # Save overlay as static file (optional)
        if res.get("overlay_b64"):
            try:
                overlay_bytes = base64.b64decode(res["overlay_b64"])
                with open(os.path.join(RESULT_FOLDER, f"overlay_{filename}.png"), "wb") as f:
                    f.write(overlay_bytes)
            except Exception:
                pass

        # Save PDF too (optional)
        if res.get("pdf_b64"):
            try:
                pdf_bytes = base64.b64decode(res["pdf_b64"])
                with open(os.path.join(RESULT_FOLDER, f"{filename}.pdf"), "wb") as f:
                    f.write(pdf_bytes)
            except Exception:
                pass

        # Return redirect URL so frontend can navigate to the same result page as upload flow
        redirect_url = url_for("camera_result", filename=filename)
        return jsonify({
            "valid": True,
            "filename": filename,
            "label": res.get("label"),
            "probs": res.get("probs"),
            "overlay": res.get("overlay_b64"),
            "redirect": redirect_url
        })

    except Exception as e:
        print("scan_camera unexpected error:", e)
        return jsonify({"error": "Unexpected server error"}), 500

# -------------------------------------------------------------------------
# Render camera result page (so camera -> next page)
# -------------------------------------------------------------------------
@app.route("/camera_result/<filename>")
def camera_result(filename):
    # load saved image and run the same processing again (or just read previously saved outputs)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(save_path):
        flash("Camera image not found")
        return redirect(url_for("index"))

    try:
        pil_img = Image.open(save_path).convert("RGB")
        res = process_image(pil_img, filename, enhance=True)

        if not res.get("valid"):
            return render_template("result.html", valid=False, distance=res.get("distance", 0.0), threshold=THRESHOLD_COSINE, error=res.get("error"))

        return render_template("result.html",
                               valid=True,
                               filename=filename,
                               label=res["label"],
                               probs=res["probs"],
                               overlay=res["overlay_b64"],
                               pdf_b64=res["pdf_b64"])
    except Exception as e:
        print("camera_result error:", e)
        flash("Error showing camera result")
        return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    return analyze()

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)

# -------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------
