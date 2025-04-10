
# ImageCaptionNo

A simple Flask web application that uses a pre-trained image captioning model to generate captions for uploaded images and translates them to Norwegian.

## Installation

1. **Install the required dependencies:**

   ```
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```
   python app.py
   ```

3. **Open your web browser and go to:**

   ```
   http://localhost:5000
   ```

## Directory Structure

```
ImageCaptionNo/
├── templates/
│   └── index.html
├── uploads/
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Usage

- Upload an image on the web page.
- The application will process the image, generate an English caption, translate it to Norwegian, and display the caption.
