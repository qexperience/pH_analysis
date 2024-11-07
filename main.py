from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "ph_detection_model.h5"
model = load_model(MODEL_PATH)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match the model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Render the camera-based UI

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        # Open the uploaded image
        image = Image.open(io.BytesIO(file.read()))
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Make prediction
        predicted_ph = model.predict(processed_image)[0][0]
        return jsonify({"predicted_pH": round(float(predicted_ph), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
