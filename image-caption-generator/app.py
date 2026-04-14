import os
import uuid

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from model import generate_caption


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file) -> tuple[str, str]:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    safe_name = secure_filename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower() or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(saved_path)
    return unique_name, saved_path


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/caption")
def caption():
    file = request.files.get("image")
    if file is None or file.filename is None or file.filename.strip() == "":
        return render_template("index.html", error="Please choose an image file to upload.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload PNG/JPG/JPEG/WEBP/BMP.",
        )
    unique_name, saved_path = save_upload(file)

    try:
        cap = generate_caption(saved_path)
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Caption generation failed: {e}",
            image_url=url_for("static", filename=f"uploads/{unique_name}"),
        )

    return render_template(
        "index.html",
        caption=cap,
        image_url=url_for("static", filename=f"uploads/{unique_name}"),
    )


@app.post("/caption-api")
def caption_api():
    """
    AJAX endpoint used by live camera capture (returns JSON).
    Expects multipart/form-data with field name: image
    """
    file = request.files.get("image")
    if file is None or file.filename is None or file.filename.strip() == "":
        return jsonify({"ok": False, "error": "No image received."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Unsupported file type. Please upload PNG/JPG/JPEG/WEBP/BMP.",
                }
            ),
            400,
        )

    unique_name, saved_path = save_upload(file)
    image_url = url_for("static", filename=f"uploads/{unique_name}")

    try:
        cap = generate_caption(saved_path)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Caption generation failed: {e}"}), 500

    return jsonify({"ok": True, "caption": cap, "image_url": image_url})


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug, threaded=True)

