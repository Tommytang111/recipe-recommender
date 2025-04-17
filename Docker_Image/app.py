#Libraries
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import io
from typing import List
import base64
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI(title="Ingredient Recognition API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (replace with specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#load YOLO model and dataset
#Also available on https://hub.ultralytics.com/models/ABUNZMg7ykq7mQz6Xt0L
model = YOLO("yolo_fruits_and_vegetables_v3.pt") 

# Custom Class to Food map
cls_names = {0: 'Almond', 1: 'Apple', 2: 'Apricot', 3: 'Artichoke', 4: 'Asparagus', 5: 'Avocado', 6: 'Banana', 
             7: 'Tofu', 8: 'Bell pepper', 9: 'Blackberry', 10: 'Blueberry', 11: 'Broccoli', 
             12: 'Brussels sprouts', 13: 'Cantaloupe', 14: 'Carrot', 15: 'Cauliflower', 
             16: 'Cayenne', 17: 'Celery', 18: 'Cherry', 19: 'Chickpea', 
             20: 'Chili', 21: 'Clementine', 22: 'Coconut', 23: 'Corn', 
             24: 'Cucumber', 25: 'Date', 26: 'Eggplant', 27: 'Fig', 
             28: 'Garlic', 29: 'Ginger', 30: 'Strawberry', 31: 'Gourd', 
             32: 'Grape', 33: 'Green bean', 34: 'Green onion', 35: 'Tomato', 
             36: 'Kiwi fruit', 37: 'Lemon', 38: 'Lettuce', 39: 'Lime', 
             40: 'Mandarin orange', 41: 'Melon', 42: 'Mushroom', 43: 'Onion', 
             44: 'Orange', 45: 'Papaya', 46: 'Pea', 47: 'Peach', 
             48: 'Pear', 49: 'Persimmon', 50: 'Pickle', 51: 'Pineapple', 
             52: 'Potato', 53: 'Prune', 54: 'Pumpkin', 55: 'Radish', 
             56: 'Raspberry', 57: 'Strawberry', 58: 'Sweet potato', 59: 'Tomato', 
             60: 'Turnip', 61: 'Watermelon', 62: 'Zucchini'}

# Add a home route for basic API information
@app.get("/")
async def home():
    """
    Home page with API information
    """
    return {
        "api_name": "Ingredient Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This information",
            "/upload": "Upload an image to detect ingredients",
            "/docs": "API documentation"
        },
        "model": "YOLO Fruits and Vegetables v3",
        "detectable_items": len(cls_names)
    }

# Endpoint to handle file uploads
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for processing
    """
    # Read the uploaded file
    contents = await file.read()
    
    # Convert to numpy array for OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file. Please upload a valid image."}
        )
    
    # Process with the predict function
    image_with_boxes, unique_foods = predict_objects(img)
    
    # Convert the processed image to base64 string
    _, encoded_img = cv2.imencode('.png', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    encoded_img_str = base64.b64encode(encoded_img).decode('utf-8')
    
    # Return the results
    return {
        "processed_image": encoded_img_str,
        "detected_foods": list(unique_foods)
    }
    
@app.get("/upload", response_class=HTMLResponse)
async def upload_form():
    """
    Return HTML form for uploading images
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ingredient Recognition</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #4a4a4a;
                text-align: center;
            }
            .upload-container {
                border: 2px dashed #ccc;
                padding: 20px;
                text-align: center;
                margin: 20px 0;
                border-radius: 5px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            #result-container {
                margin-top: 20px;
                display: none;
            }
            #result-image {
                max-width: 100%;
                margin-bottom: 10px;
            }
            #detected-foods {
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Ingredient Recognition</h1>
        <div class="upload-container">
            <h2>Upload an image to detect ingredients</h2>
            <form id="upload-form" enctype="multipart/form-data">
                <input type="file" id="file-input" name="file" accept="image/*">
                <br>
                <button type="submit">Detect Ingredients</button>
            </form>
        </div>
        
        <div id="result-container">
            <h2>Results</h2>
            <img id="result-image" src="" alt="Processed image">
            <h3>Detected Ingredients:</h3>
            <div id="detected-foods"></div>
        </div>

        <script>
            document.getElementById('upload-form').addEventListener('submit', async function(event) {
                event.preventDefault();
                
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error('Error uploading image');
                    }
                    
                    const data = await response.json();
                    
                    // Show results
                    document.getElementById('result-container').style.display = 'block';
                    document.getElementById('result-image').src = 'data:image/png;base64,' + data.processed_image;
                    
                    const detectedFoodsContainer = document.getElementById('detected-foods');
                    detectedFoodsContainer.innerHTML = '';
                    
                    if (data.detected_foods.length > 0) {
                        const ul = document.createElement('ul');
                        data.detected_foods.forEach(food => {
                            const li = document.createElement('li');
                            li.textContent = food;
                            ul.appendChild(li);
                        });
                        detectedFoodsContainer.appendChild(ul);
                    } else {
                        detectedFoodsContainer.textContent = 'No ingredients detected';
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error processing image: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    '''

#@app.get("/predict")
def predict_objects(image):
    """Run prediction on an image and return labelled image with unique labels"""
    # Check if image is a path or already a numpy array
    if isinstance(image, str):
        # Load and convert image to RGB
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray) and image.ndim == 3:
        # If BGR, convert to RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    result = model(image)
    predictions = result[0]
    
    # Create a copy of the image to avoid modifying the original
    image_with_boxes = image.copy()
    
    for box, cls, conf in zip(predictions.boxes.xyxy, predictions.boxes.cls, predictions.boxes.conf):
        x1, y1, x2, y2 = box
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        # Draw rectangle
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # Add text label
        label = f"{cls_names[cls]} Conf:{conf:.2f}" #Converts class from number to actual name of the class
        cv2.putText(image_with_boxes, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print(f"Found {len(predictions.boxes)} detections in image")
    
    return image_with_boxes, set(cls_names[int(i)] for i in predictions.boxes.cls)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)