// Three.js Background Animation for Fetal Ultrasound Dashboard
(function () {
  const canvas = document.getElementById('three-bg-canvas');
  if (!canvas) return;

  // Scene configuration
  const scene = new THREE.Scene();

  // Camera configuration
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
  camera.position.z = 220;
  camera.position.y = 80;
  camera.lookAt(new THREE.Vector3(0, 0, 0));

  // Renderer configuration
  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    alpha: true,
    antialias: true
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  // Particles / Wave parameters
  const SEPARATION = 12;
  const AMOUNTX = 65;
  const AMOUNTY = 50;

  const numParticles = AMOUNTX * AMOUNTY;
  const positions = new Float32Array(numParticles * 3);
  const colors = new Float32Array(numParticles * 3);

  // Colors: primary (#0D6E6E -> RGB 13, 110, 110) and accent (#00C9A7 -> RGB 0, 201, 167)
  const color1 = new THREE.Color(0x0d6e6e);
  const color2 = new THREE.Color(0x00c9a7);

  let i = 0;
  for (let ix = 0; ix < AMOUNTX; ix++) {
    for (let iy = 0; iy < AMOUNTY; iy++) {
      // Map grid index to positions
      positions[i] = ix * SEPARATION - (AMOUNTX * SEPARATION) / 2; // x
      positions[i + 1] = 0;                                         // y
      positions[i + 2] = iy * SEPARATION - (AMOUNTY * SEPARATION) / 2; // z

      // Blend colors based on location
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

  // Particle Material
  // Using a custom canvas-generated texture or standard round particles
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

  // Mouse Interaction variables
  let mouseX = 0;
  let mouseY = 0;
  let targetX = 0;
  let targetY = 0;

  window.addEventListener('mousemove', (event) => {
    // Normalize coordinates -0.5 to 0.5
    mouseX = (event.clientX - window.innerWidth / 2) / (window.innerWidth / 2);
    mouseY = (event.clientY - window.innerHeight / 2) / (window.innerHeight / 2);
  });

  // Resize listener
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // Animation variables
  let count = 0;

  function animate() {
    requestAnimationFrame(animate);

    count += 0.035;

    // Fluid organic movement of the particle plane
    const positions = particles.geometry.attributes.position.array;
    let i = 0;
    for (let ix = 0; ix < AMOUNTX; ix++) {
      for (let iy = 0; iy < AMOUNTY; iy++) {
        // Double sine wave equations to simulate sonar wave front
        positions[i + 1] = 
          Math.sin(ix * 0.12 + count) * 16 +
          Math.sin(iy * 0.18 + count * 0.8) * 12;
        i += 3;
      }
    }
    particles.geometry.attributes.position.needsUpdate = true;

    // Smooth mouse damping
    targetX += (mouseX - targetX) * 0.05;
    targetY += (mouseY - targetY) * 0.05;

    // Subtle camera tilt following mouse
    camera.position.x += (targetX * 60 - camera.position.x) * 0.03;
    // Keep camera angle tilted downwards
    camera.position.y += ((80 - targetY * 40) - camera.position.y) * 0.03;
    camera.lookAt(scene.position);

    renderer.render(scene, camera);
  }

  // Start animation loop
  animate();
})();
