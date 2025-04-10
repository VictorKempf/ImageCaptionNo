from flask import Flask, request, render_template
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
from googletrans import Translator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the image captioning model, processor, and tokenizer
model_id = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
processor = ViTImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Choose device: CUDA if available, if not, CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    caption_text = None
    error = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No image file provided"
        else:
            file = request.files['image']
            if file.filename == '':
                error = "No selected file"
            else:
                # Save the uploaded image temporarily
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                try:
                    image = Image.open(file_path)
                except Exception as e:
                    error = "Invalid image file"
                else:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    # Preprocess the image
                    inputs = processor(images=image, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(device)

                    # Generate caption using beam search
                    output_ids = model.generate(pixel_values, max_length=32, num_beams=4)
                    caption_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    translator = Translator()
                    caption_text = translator.translate(caption_text, dest="no").text
                
                # Remove the temporarily saved image
                os.remove(file_path)

    return render_template('index.html', caption=caption_text, error=error)

if __name__ == '__main__':
    # Set debug=False in production
    app.run(debug=True)
