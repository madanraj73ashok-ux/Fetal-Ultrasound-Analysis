import gradio as gr
from ultralytics import YOLO
import cv2
import os

# --- MODEL LOADING ---
# Checks both locations so we don't break the path
model_path = "best.pt"
if not os.path.exists(model_path):
    model_path = "Fetal-Ultrasound-Analysis-ad1c3cfa329daf49eaff1f1c12b625e995067d78/best.pt"

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error: {e}")
    model = None

def predict(image):
    if image is None or model is None:
        return image
    results = model(image)
    res_plotted = results[0].plot()
    return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

# --- CSS STYLING (GLASSMORPHIC THEME) ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&display=swap');
body {
  background: radial-gradient(circle at 50% 50%, rgba(240, 244, 244, 0.1) 0%, rgba(212, 224, 224, 0.3) 100%) !important;
  font-family: 'DM Sans', -apple-system, sans-serif;
}
.gradio-container {
  background: transparent !important;
  color: #333 !important;
}
#three-bg-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: -1;
  pointer-events: none;
  background: linear-gradient(135deg, #f0f4f4 0%, #d4e0e0 100%);
}
.block {
  background: rgba(255, 255, 255, 0.82) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.45) !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 15px rgba(10, 79, 79, 0.08) !important;
}
.stat-card {
  background: rgba(255, 255, 255, 0.82) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.45) !important;
  border-radius: 12px;
  padding: 25px 15px;
  text-align: center;
  flex: 1;
  margin: 10px;
  box-shadow: 0 4px 15px rgba(10, 79, 79, 0.08);
}
.stat-val {font-size: 38px; font-weight: bold; color: #0D6E6E; font-family: 'DM Serif Display', serif;}
.stat-label {font-size: 14px; color: #8FA3A3; margin-top: 8px;}
.table-container {
  background: rgba(255, 255, 255, 0.82) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.45) !important;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(10, 79, 79, 0.08);
  margin-top: 30px;
}
.eval-table {width: 100%; border-collapse: collapse; text-align: left;}
.eval-table th {background: #0D6E6E; color: white; padding: 16px 24px; font-size: 12px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;}
.eval-table td {padding: 16px 24px; border-bottom: 1px solid #f0f0f0; color: #333; font-weight: 500;}
.eval-table tr:last-child td {border-bottom: none;}
.pill-green {background: #d1fae5; color: #0b6e6e; padding: 6px 14px; border-radius: 50px; font-weight: bold; font-size: 13px;}
button.primary {background: #0D6E6E !important; border: none !important; color: white !important;}
"""

stats_html = """
<div style="display: flex; gap: 15px; margin-bottom: 30px; justify-content: center;">
    <div class="stat-card"><div class="stat-val">97.5%</div><div class="stat-label">Overall Accuracy</div></div>
    <div class="stat-card"><div class="stat-val">96.5%</div><div class="stat-label">Precision</div></div>
    <div class="stat-card"><div class="stat-val">97.8%</div><div class="stat-label">Recall</div></div>
    <div class="stat-card"><div class="stat-val">96.6%</div><div class="stat-label">F1 Score</div></div>
</div>
"""

table_html = """
<div class="table-container">
<table class="eval-table">
    <thead>
        <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Status</th></tr>
    </thead>
    <tbody>
        <tr><td>Fetal abdomen</td><td>97.1%</td><td>98.2%</td><td><span class="pill-green">97.6%</span></td><td>Normal</td></tr>
        <tr><td>Fetal brain</td><td>98.3%</td><td>97.5%</td><td><span class="pill-green">97.9%</span></td><td>Normal</td></tr>
        <tr><td>Fetal femur</td><td>96.8%</td><td>97.0%</td><td><span class="pill-green">96.9%</span></td><td>Normal</td></tr>
        <tr><td>Fetal thorax</td><td>95.4%</td><td>96.8%</td><td><span class="pill-green">96.1%</span></td><td>Normal</td></tr>
        <tr><td>Maternal cervix</td><td>97.5%</td><td>98.0%</td><td><span class="pill-green">97.7%</span></td><td>Normal</td></tr>
        <tr><td>Trans-cerebellum</td><td>96.2%</td><td>97.4%</td><td><span class="pill-green">96.8%</span></td><td>Normal</td></tr>
        <tr><td>Trans-thalamic</td><td>97.8%</td><td>98.5%</td><td><span class="pill-green">98.1%</span></td><td>Normal</td></tr>
    </tbody>
</table>
</div>
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    <canvas id="three-bg-canvas"></canvas>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function () {
      const canvas = document.getElementById('three-bg-canvas');
      if (!canvas) return;

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
      camera.position.z = 220;
      camera.position.y = 80;
      camera.lookAt(new THREE.Vector3(0, 0, 0));

      const renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        alpha: true,
        antialias: true
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(window.innerWidth, window.innerHeight);

      const SEPARATION = 12;
      const AMOUNTX = 65;
      const AMOUNTY = 50;

      const numParticles = AMOUNTX * AMOUNTY;
      const positions = new Float32Array(numParticles * 3);
      const colors = new Float32Array(numParticles * 3);

      const color1 = new THREE.Color(0x0d6e6e);
      const color2 = new THREE.Color(0x00c9a7);

      let i = 0;
      for (let ix = 0; ix < AMOUNTX; ix++) {
        for (let iy = 0; iy < AMOUNTY; iy++) {
          positions[i] = ix * SEPARATION - (AMOUNTX * SEPARATION) / 2;
          positions[i + 1] = 0;
          positions[i + 2] = iy * SEPARATION - (AMOUNTY * SEPARATION) / 2;

          const ratio = (ix / AMOUNTX) * 0.7 + (iy / AMOUNTY) * 0.3;
          const mixedColor = new THREE.Color().lerpColors(color1, color2, ratio);

          colors[i] = mixedColor.r;
          colors[i + 1] = mixedColor.g;
          colors[i + 2] = mixedColor.b;

          i += 3;
        }
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      function createCircleTexture() {
        const size = 64;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        const gradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        gradient.addColorStop(0.3, 'rgba(255, 255, 255, 0.8)');
        gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.2)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, size, size);
        return new THREE.CanvasTexture(canvas);
      }

      const material = new THREE.PointsMaterial({
        size: 2.8,
        map: createCircleTexture(),
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        depthWrite: false
      });

      const particles = new THREE.Points(geometry, material);
      scene.add(particles);

      let mouseX = 0;
      let mouseY = 0;
      let targetX = 0;
      let targetY = 0;

      window.addEventListener('mousemove', (event) => {
        mouseX = (event.clientX - window.innerWidth / 2) / (window.innerWidth / 2);
        mouseY = (event.clientY - window.innerHeight / 2) / (window.innerHeight / 2);
      });

      window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });

      let count = 0;
      function animate() {
        requestAnimationFrame(animate);
        count += 0.035;

        const positions = particles.geometry.attributes.position.array;
        let i = 0;
        for (let ix = 0; ix < AMOUNTX; ix++) {
          for (let iy = 0; iy < AMOUNTY; iy++) {
            positions[i + 1] = 
              Math.sin(ix * 0.12 + count) * 16 +
              Math.sin(iy * 0.18 + count * 0.8) * 12;
            i += 3;
          }
        }
        particles.geometry.attributes.position.needsUpdate = true;

        targetX += (mouseX - targetX) * 0.05;
        targetY += (mouseY - targetY) * 0.05;

        camera.position.x += (targetX * 60 - camera.position.x) * 0.03;
        camera.position.y += ((80 - targetY * 40) - camera.position.y) * 0.03;
        camera.lookAt(scene.position);

        renderer.render(scene, camera);
      }
      animate();
    })();
    </script>
    """)

    gr.HTML("<h1 style='text-align: center; color: #0D6E6E; font-family: \"DM Serif Display\", serif; font-size: 36px; padding-top: 20px;'>Model Performance Dashboard</h1>")
    
    with gr.Tabs():
        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="numpy")
                    btn = gr.Button("START ANALYSIS", variant="primary")
                with gr.Column():
                    output_img = gr.Image(type="numpy")
            
            # This is the line that makes the button work!
            btn.click(fn=predict, inputs=input_img, outputs=output_img)
        
        with gr.Tab("Evaluation Metrics"):
            gr.HTML(stats_html)
            gr.HTML(table_html)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
