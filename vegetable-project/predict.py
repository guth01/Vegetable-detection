#!/usr/bin/env python3
import sys
import json
import os
from ultralytics import YOLO

def predict_vegetable(image_path):
    try:
        # Load your trained YOLO model
        model = YOLO('best (5).pt')
        
        # Run prediction on the uploaded image
        results = model.predict(source=image_path, save=False, verbose=False)
        
        # Process results
        predictions = []
        
        if results and len(results) > 0:
            result = results[0]  # Get first result
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Get detection boxes, classes, and confidences
                boxes = result.boxes
                
                if len(boxes) > 0:
                    # Get class names (you may need to adjust this based on your model)
                    class_names = result.names if hasattr(result, 'names') else {}
                    
                    # Process each detection
                    for i in range(len(boxes.cls)):
                        class_id = int(boxes.cls[i].item())
                        confidence = float(boxes.conf[i].item())
                        
                        # Get class name
                        vegetable_name = class_names.get(class_id, f'Class_{class_id}')
                        
                        predictions.append({
                            'vegetable': vegetable_name,
                            'confidence': confidence
                        })
                    
                    # Sort by confidence (highest first)
                    predictions.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Return top 3 predictions
                    predictions = predictions[:3]
        
        # If no detections found, return a default response
        if not predictions:
            predictions = [{
                'vegetable': 'Unknown',
                'confidence': 0.0
            }]
        
        return predictions
        
    except Exception as e:
        # Return error information
        return [{
            'vegetable': 'Error',
            'confidence': 0.0,
            'error': str(e)
        }]

def main():
    if len(sys.argv) != 2:
        print(json.dumps([{'vegetable': 'Error', 'confidence': 0.0, 'error': 'No image path provided'}]))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(json.dumps([{'vegetable': 'Error', 'confidence': 0.0, 'error': 'Image file not found'}]))
        sys.exit(1)
    
    # Run prediction
    predictions = predict_vegetable(image_path)
    
    # Output results as JSON
    print(json.dumps(predictions))

if __name__ == "__main__":
    main()