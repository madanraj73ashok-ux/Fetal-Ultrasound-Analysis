import gradio as gr
from ultralytics import YOLO
import cv2
import os

# Model path logic
model_path = "best.pt"
if not os.path.exists(model_path):
    model_path = "Fetal-Ultrasound-Analysis-ad1c3cfa329daf49eaff1f1c12b625e995067d78/best.pt"

model = YOLO(model_path)

def predict(image):
    results = model(image)
    res_plotted = results[0].plot()
    return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

# CUSTOM CSS FOR THE "DASHBOARD" LOOK
custom_css = """
body {background-color: #0b1a19;}
.gradio-container {background-color: #0b1a19 !important; color: white !important;}
#header {text-align: center; margin-bottom: 20px;}
.pill {background-color: #102a28 !important; border: 1px solid #1f4d4a !important;}
button.primary {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
}
.feedback {display: none !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# ?? FetalNet AI: Medical Analysis Dashboard")
        gr.Markdown("### Real-time Fetal Ultrasound Anatomical Plane Detection")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### ?? Upload Ultrasound Scan")
            input_img = gr.Image(type="numpy", label=None)
            btn = gr.Button("START ANALYSIS", variant="primary")
            
        with gr.Column():
            gr.Markdown("#### ??? Analysis Results")
            output_img = gr.Image(type="numpy", label=None)
    
    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("?? **Model:** YOLOv8-Medical | ?? **Status:** System Ready")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
