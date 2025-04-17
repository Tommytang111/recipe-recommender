#Libraries
import cv2
from ultralytics import YOLO
from fastapi import FastAPI

#load YOLO model and dataset
model = YOLO("Models/yolo_fruits_and_vegetables_v3.pt") 

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

#Functions
@app.get("/predict")
def predict_objects(image):
    """Run prediction on an image and return labelled image with unique labels"""
    #Load and convert image to RGB
    image = cv2.imread(str(image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Run inference
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
    
    print(f"Found {len(predictions)} detections in sample image")
    
    return image_with_boxes, set(cls_names[int(i)] for i in predictions.boxes.cls)

if __name__ == "__main__":
    image = "some_user_input"
    print(f"Unique foods: {predict_objects(image)[1]}")