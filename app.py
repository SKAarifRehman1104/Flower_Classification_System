from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "/Users/manaspurohit/Documents/Coding/Machine Learning /Oxford_Flowers102_model_MobileNet.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load the flower descriptions from the CSV file
CSV_PATH = "/Users/manaspurohit/Documents/Coding/Machine Learning /Flowers.csv"
flower_data = pd.read_csv(CSV_PATH)

# Create a dictionary of flower names and descriptions
# description_dict = dict(zip(flower_data['Cat_Name'], flower_data['Location']))

# Create a dictionary of flower names and their attributes
description_dict = {
    row['Cat_Name']: {
        "Location": row.get("Location", "Unknown"),
        "Scientific Name": row.get("Scientific Name", "Unknown"),
        "Family": row.get("Family", "Unknown"),
        "Genus": row.get("Genus", "Unknown"),
        "Species": row.get("Species", "Unknown"),
        "Bloom Time": row.get("Bloom Time", "Unknown")
    }
    for _, row in flower_data.iterrows()
}

# Define the flower class labels (Replace with your actual class names)
CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", 
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood", 
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", 
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", 
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", 
    "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster", 
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", 
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", 
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", 
    "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", 
    "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone", 
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus", 
    "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", 
    "thorn apple", "morning glory", "passion flower", "lotus lotus", "toad lily", "anthurium", 
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", 
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", 
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", 
    "trumpet creeper", "blackberry lily"
]

# Preprocess the input image
def preprocess_image(image_path, img_height=224, img_width=224):
    image = Image.open(image_path)
    image = image.resize((img_height, img_width))  # Resize to target dimensions
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Route for the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!"
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected!"
        
        if file:
            # Save the uploaded file
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Preprocess and predict
            processed_image = preprocess_image(file_path)
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction) * 100

            # Get the description for the predicted class
            flower_description = description_dict.get(predicted_class, "Description not available.")

            return render_template(
                "index.html",
                uploaded_image=file.filename,
                predicted_class=predicted_class,
                confidence=confidence,
                flower_description=flower_description
            )
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)