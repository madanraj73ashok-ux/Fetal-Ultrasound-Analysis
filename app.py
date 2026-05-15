import gradio as gr
from ultralytics import YOLO
import cv2
import os

# Adjust path if model is in a subfolder
model_path = "best.pt"
if not os.path.exists(model_path):
    # This checks the subfolder if the root check fails
    model_path = "Fetal-Ultrasound-Analysis-ad1c3cfa329daf49eaff1f1c12b625e995067d78/best.pt"

model = YOLO(model_path)

def predict(image):
    results = model(image)
    res_plotted = results[0].plot()
    return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="numpy"), 
    outputs=gr.Image(type="numpy"),
    title="FetalNet AI: Ultrasound Analysis"
)

# Crucial fix for Hugging Face Spaces
demo.launch(server_name="0.0.0.0", server_port=7860)
