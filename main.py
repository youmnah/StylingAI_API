from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from utils.color import process_image_and_detect_colors
from tensorflow.keras.models import model_from_json # type: ignore
from utils.config import pattern_labels, type_label_to_index_group, bottom_list, top_list, foot_list, sub_list, types_to_index_group
from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch # type: ignore
import tensorflow as tf
from roboflow import Roboflow
import tempfile
from utils.item_type import fine_tuning
import onnxruntime as ort
import cv2

app = FastAPI()

## Models
# Type Model
processor = ViTImageProcessor.from_pretrained("model/model_type")        
type_model = AutoModelForImageClassification.from_pretrained("model/model_type") 
# Pattern Model
with open('model/model_pattern/model_pattern.json', 'r') as json_file:
    model_json = json_file.read()
pattern_model = model_from_json(model_json)
pattern_model.load_weights('model/model_pattern/model_pattern.h5') 
# Bottom Model 
bottom_model = tf.keras.models.load_model('model/model_bottom')
# Top Model 
top_model = tf.keras.models.load_model('model/model_top')
# Shoes Model 
shoes_model = tf.keras.models.load_model('model/model_shoes')
# Sub Model 
sub_model = tf.keras.models.load_model('model/model_sub')
# Type 2 Model
rf = Roboflow(api_key="OwCGkBZECheBsv9Ydrcs")
project = rf.workspace().project("clothing-detection-p8vmn")
rf_type_model = project.version(6).model

# Load the ONNX model
onnx_model_path = "model/model_rf/best.onnx"  # Ensure the model is in the correct path
session = ort.InferenceSession(onnx_model_path)

def apply_nms(boxes, confidences, threshold=0.4):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=threshold)
    return indices

@app.post("/predict-item/")
async def predictItem(file: UploadFile = File(...)):
    try:
        season = ''
        style = ''                 
        sub_type = ''
        rf_cat = ''
        rf_type = ''

        # Read the uploaded file
        content = await file.read() 
        # Preprocess the input for Pattern
        pattern_processed_image = image.load_img(BytesIO(content), target_size=(64, 64))               
        pattern_processed_image = image.img_to_array(pattern_processed_image)
        pattern_processed_image = np.expand_dims(pattern_processed_image, axis = 0)
        # Make prediction for Pattern          
        prediction_pattern = pattern_model.predict(pattern_processed_image)
        pattern_id = int(np.argmax(prediction_pattern[0]))
        pattern_label = pattern_labels[np.argmax(prediction_pattern[0])]

        # Preprocess the input for color
        processed_image = Image.open(BytesIO(content)).convert("RGB")
        processed_image = processed_image.resize((299, 299)) 
        # Color
        color_name, rgb_detected = process_image_and_detect_colors(processed_image)
                
        # Make prediction for type 
        input_processed_image = Image.open(BytesIO(content)).convert("RGB")
        input_processed_image = input_processed_image.resize((299, 299))                   
        inputs = processor(images=input_processed_image, return_tensors="pt")       
        with torch.no_grad():
            outputs = type_model(**inputs)         
        predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()              
        if hasattr(type_model.config, "id2label"):
            type_label = type_model.config.id2label[predicted_class_idx]

        type = type_label_to_index_group(type_label)
        type_Id = type[0]                    
        
        # Double check 
        # rf
        processed_image = Image.open(BytesIO(content))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            processed_image.save(temp_file, format='JPEG')  # Save the image as JPEG
            temp_file_path = temp_file.name 
        result = rf_type_model.predict(temp_file_path, confidence=40, overlap=30).json()
        if len(result["predictions"]) > 0:
            labels = [item["class"] for item in result["predictions"]]            
            rf_type = labels[0]
            rf_cat = types_to_index_group("rf", rf_type)[1]

        """
        # Run inference
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)        
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0  # Normalize pixel values
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)  # Shape: (1,3,640,640)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})   

        boxes = outputs[0]  # Shape: (N, 4)
        confidences = outputs[1]  # Shape: (N,)

        # Example of filtering based on confidence and applying NMS
        filtered_boxes = []
        filtered_confidences = []

        for i in range(len(confidences)):
            if confidences[i] > 0.4:  # Example confidence threshold
                filtered_boxes.append(boxes[i])
                filtered_confidences.append(confidences[i])

        for i in indices.flatten():
            box = filtered_boxes[i]
            print(f"Box: {box}, Confidence: {filtered_confidences[i]}")

        # Apply NMS to the filtered boxes
        indices = apply_nms(filtered_boxes, filtered_confidences)         
        predicted_classes = np.argmax(outputs[2], axis=1)        
        
        confidence_scores = outputs[1].flatten()
        class_ids = outputs[5].flatten()
        confidence_threshold = 0.5
        confidence_scores = np.max(outputs[1], axis=1).flatten()

        max_confidence_index = np.argmax(confidence_scores)

        # Get the highest confidence score and corresponding class ID
        highest_confidence = confidence_scores[max_confidence_index]
        highest_class_id = class_ids[max_confidence_index]

        # Print the highest confidence prediction
        print(f"Highest Confidence Prediction:")
        print(f"Confidence: {highest_confidence}, Class ID: {highest_class_id}")       
        """

        # Sub Model      
        sub_processed_image = image.load_img(BytesIO(content), target_size=(80,60,3))               
        sub_processed_image = image.img_to_array(sub_processed_image)
        sub_processed_image = np.expand_dims(sub_processed_image, axis = 0)    
        prediction_sub = sub_model.predict(sub_processed_image)        
        sub_predict = sub_list[np.argmax(prediction_sub)]

        # Season and Style Predictions      
        cat = type[1]

        if sub_type == "Other":
            type_Id = 0
            type_label = ''
            pattern_id = 0
            pattern_label = 0
        else:            
            if cat == "Accessories":                
                cat = sub_predict if rf_cat is None else rf_cat

            if cat == "Top":            
                prediction_top = top_model.predict(sub_processed_image)
                season = top_list[3][np.argmax(prediction_top[3][0])]
                style = top_list[4][np.argmax(prediction_top[4][0])]                
                sub_type = top_list[0][np.argmax(prediction_top[0][0])] 
            elif cat == "Bottom":                
                prediction_bottom = bottom_model.predict(sub_processed_image)
                season = bottom_list[3][np.argmax(prediction_bottom[3][0])]
                style = bottom_list[4][np.argmax(prediction_bottom[4][0])]                
                sub_type = bottom_list[0][np.argmax(prediction_bottom[0][0])] 
            elif cat == "Shoes":                
                prediction_shoes = shoes_model.predict(sub_processed_image)
                season = foot_list[3][np.argmax(prediction_shoes[3][0])]                
                style = foot_list[4][np.argmax(prediction_shoes[4][0])]
                sub_type = foot_list[0][np.argmax(prediction_shoes[0][0])] 

        # Final value
        final = fine_tuning(type[1], type_Id, type_label, rf_cat, rf_type, sub_type, sub_predict, season, style)       
        cat = final[0]
        type_Id = final[1]        
        type_label = final[2]
        season = final[3]
        style = final[4]
        
        return  {
                    "category": cat.lower(),
                    "type_Id": type_Id,
                    "type": type_label.lower(),
                    "pattern_Id": pattern_id,
                    "pattern": pattern_label,
                    "color": color_name, 
                    "color_rgb": rgb_detected.tolist(),
                    "season": season.lower(),
                    "style": style.lower()
                }    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        