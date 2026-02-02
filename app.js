// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const uploadedImage = document.getElementById('uploadedImage');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const imageUpload = document.getElementById('imageUpload');
const loadingEl = document.getElementById('loading');
const totalCountEl = document.getElementById('totalCount');
const fpsValueEl = document.getElementById('fpsValue');
const objectCountsEl = document.getElementById('objectCounts');
const detectionLogEl = document.getElementById('detectionLog');
const confidenceThreshold = document.getElementById('confidenceThreshold');
const confidenceValue = document.getElementById('confidenceValue');
const maxDetections = document.getElementById('maxDetections');
const maxDetectionsValue = document.getElementById('maxDetectionsValue');
const modelSelect = document.getElementById('modelSelect');
const modelHint = document.getElementById('modelHint');
const modelValueEl = document.getElementById('modelValue');
const sceneDescriptionEl = document.getElementById('sceneDescription');

// State
let currentModel = null;
let currentModelType = 'coco-lite';
let isDetecting = false;
let animationId = null;
let stream = null;
let lastTime = 0;
let frameCount = 0;
let fps = 0;
let inputMode = 'camera'; // 'camera' or 'image'
let isModelLoading = false;

// Model configurations
const modelConfigs = {
    'coco-lite': {
        name: 'COCO Lite',
        hint: '80 object classes, optimized for speed',
        type: 'object'
    },
    'coco-full': {
        name: 'COCO Full',
        hint: '80 object classes, higher accuracy',
        type: 'object'
    },
    'movenet': {
        name: 'MoveNet',
        hint: '17 body keypoints for pose estimation',
        type: 'pose'
    },
    'handpose': {
        name: 'HandPose',
        hint: '21 hand landmarks per hand detected',
        type: 'hand'
    }
};

// Colors for different object classes
const classColors = {
    person: '#ef4444',
    car: '#3b82f6',
    truck: '#8b5cf6',
    bus: '#f59e0b',
    motorcycle: '#10b981',
    bicycle: '#ec4899',
    dog: '#f97316',
    cat: '#14b8a6',
    bird: '#6366f1',
    horse: '#a855f7',
    default: '#00d9ff'
};

// Pose keypoint connections for drawing skeleton
const POSE_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4], // Head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
    [5, 11], [6, 12], [11, 12], // Torso
    [11, 13], [13, 15], [12, 14], [14, 16] // Legs
];

// Hand landmark connections
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17] // Palm
];

// Update loading message
function updateLoadingMessage(message) {
    const loadingText = loadingEl.querySelector('p');
    if (loadingText) {
        loadingText.textContent = message;
    }
}

// Load model based on type
async function loadModel(modelType) {
    if (isModelLoading) return;
    isModelLoading = true;
    
    loadingEl.classList.remove('hidden');
    updateLoadingMessage(`Initializing TensorFlow.js...`);
    
    try {
        // Ensure TensorFlow.js backend is ready
        await tf.ready();
        console.log('TensorFlow.js backend:', tf.getBackend());
        
        updateLoadingMessage(`Loading ${modelConfigs[modelType].name}...`);
        addLogEntry(`Loading ${modelConfigs[modelType].name}...`);
        
        switch (modelType) {
            case 'coco-lite':
                currentModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
                break;
            case 'coco-full':
                currentModel = await cocoSsd.load({ base: 'mobilenet_v2' });
                break;
            case 'movenet':
                currentModel = await poseDetection.createDetector(
                    poseDetection.SupportedModels.MoveNet,
                    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
                );
                break;
            case 'handpose':
                currentModel = await handPoseDetection.createDetector(
                    handPoseDetection.SupportedModels.MediaPipeHands,
                    { runtime: 'tfjs', maxHands: 2 }
                );
                break;
        }
        
        currentModelType = modelType;
        modelValueEl.textContent = modelConfigs[modelType].name;
        modelHint.textContent = modelConfigs[modelType].hint;
        
        console.log(`${modelConfigs[modelType].name} loaded successfully!`);
        addLogEntry(`${modelConfigs[modelType].name} loaded - Ready!`);
        
        loadingEl.classList.add('hidden');
        startBtn.disabled = false;
        isModelLoading = false;
        
        return true;
    } catch (error) {
        console.error('Error loading model:', error);
        addLogEntry(`Error loading model: ${error.message}`);
        loadingEl.innerHTML = `
            <p style="color: #ef4444;">Error loading model. Please try another.</p>
            <p style="color: #94a3b8; font-size: 0.875rem;">${error.message}</p>
        `;
        isModelLoading = false;
        return false;
    }
}

// Initialize the application
async function init() {
    await loadModel('coco-lite');
}

// Start camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        
        video.srcObject = stream;
        video.style.display = 'block';
        uploadedImage.style.display = 'none';
        inputMode = 'camera';
        
        imageUpload.value = '';
        canvas.classList.add('mirrored');
        
        await video.play();
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isDetecting = true;
        
        addLogEntry('Camera started');
        detectFrame();
    } catch (error) {
        console.error('Error accessing camera:', error);
        addLogEntry('Error: Could not access camera');
        alert('Could not access camera. Please ensure camera permissions are granted.');
    }
}

// Stop detection
function stopDetection() {
    isDetecting = false;
    
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    totalCountEl.textContent = '0';
    fpsValueEl.textContent = '0';
    objectCountsEl.innerHTML = '<p class="placeholder">Start detection to see counts...</p>';
    
    addLogEntry('Detection stopped');
}

// Handle image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = async (e) => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = 'block';
        video.style.display = 'none';
        inputMode = 'image';
        
        canvas.classList.remove('mirrored');
        
        uploadedImage.onload = async () => {
            canvas.width = uploadedImage.naturalWidth;
            canvas.height = uploadedImage.naturalHeight;
            
            addLogEntry('Image uploaded: ' + file.name);
            await detectImage();
        };
    };
    reader.readAsDataURL(file);
    
    imageUpload.value = '';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    isDetecting = false;
}

// Run detection based on current model type
async function runDetection(source, width, height) {
    const threshold = parseFloat(confidenceThreshold.value);
    const maxDet = parseInt(maxDetections.value);
    const modelType = modelConfigs[currentModelType].type;
    
    try {
        let results = [];
        
        if (modelType === 'object') {
            const predictions = await currentModel.detect(source, maxDet);
            results = predictions.filter(p => p.score >= threshold);
            drawObjectDetections(results, width, height);
            updateObjectCounts(results);
        } else if (modelType === 'pose') {
            const poses = await currentModel.estimatePoses(source);
            results = poses;
            drawPoseDetections(poses, width, height);
            updatePoseCounts(poses);
        } else if (modelType === 'hand') {
            const hands = await currentModel.estimateHands(source);
            results = hands;
            drawHandDetections(hands, width, height);
            updateHandCounts(hands);
        }
        
        return results;
    } catch (error) {
        console.error('Detection error:', error);
        return [];
    }
}

// Detect objects in uploaded image
async function detectImage() {
    if (!currentModel) return;
    
    try {
        const results = await runDetection(uploadedImage, uploadedImage.naturalWidth, uploadedImage.naturalHeight);
        addLogEntry(`Detected ${Array.isArray(results) ? results.length : 0} items in image`);
    } catch (error) {
        console.error('Detection error:', error);
        addLogEntry('Error detecting objects');
    }
}

// Detection loop for video
async function detectFrame() {
    if (!isDetecting || !currentModel) return;
    
    try {
        await runDetection(video, video.videoWidth, video.videoHeight);
        updateFPS();
    } catch (error) {
        console.error('Detection error:', error);
    }
    
    animationId = requestAnimationFrame(detectFrame);
}

// Draw object detections (COCO-SSD)
function drawObjectDetections(predictions, sourceWidth, sourceHeight) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scaleX = canvas.width / sourceWidth;
    const scaleY = canvas.height / sourceHeight;
    
    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        const color = classColors[prediction.class] || classColors.default;
        const confidence = Math.round(prediction.score * 100);
        const label = `${prediction.class} ${confidence}%`;
        
        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw label background
        ctx.font = 'bold 14px Segoe UI, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const textHeight = 20;
        const padding = 6;
        
        ctx.fillStyle = color;
        ctx.fillRect(scaledX - 1, scaledY - textHeight - padding, textWidth + padding * 2, textHeight + padding);
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, scaledX + padding - 1, scaledY - padding);
        
        // Draw corner accents
        const cornerLength = 15;
        ctx.lineWidth = 4;
        
        ctx.beginPath();
        ctx.moveTo(scaledX, scaledY + cornerLength);
        ctx.lineTo(scaledX, scaledY);
        ctx.lineTo(scaledX + cornerLength, scaledY);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(scaledX + scaledWidth - cornerLength, scaledY);
        ctx.lineTo(scaledX + scaledWidth, scaledY);
        ctx.lineTo(scaledX + scaledWidth, scaledY + cornerLength);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(scaledX, scaledY + scaledHeight - cornerLength);
        ctx.lineTo(scaledX, scaledY + scaledHeight);
        ctx.lineTo(scaledX + cornerLength, scaledY + scaledHeight);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(scaledX + scaledWidth - cornerLength, scaledY + scaledHeight);
        ctx.lineTo(scaledX + scaledWidth, scaledY + scaledHeight);
        ctx.lineTo(scaledX + scaledWidth, scaledY + scaledHeight - cornerLength);
        ctx.stroke();
    });
}

// Draw pose detections (MoveNet)
function drawPoseDetections(poses, sourceWidth, sourceHeight) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scaleX = canvas.width / sourceWidth;
    const scaleY = canvas.height / sourceHeight;
    const threshold = parseFloat(confidenceThreshold.value);
    
    poses.forEach((pose, poseIndex) => {
        const keypoints = pose.keypoints;
        const color = poseIndex === 0 ? '#00d9ff' : '#7c3aed';
        
        // Draw skeleton connections
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        
        POSE_CONNECTIONS.forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];
            
            if (kp1.score > threshold && kp2.score > threshold) {
                ctx.beginPath();
                ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY);
                ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY);
                ctx.stroke();
            }
        });
        
        // Draw keypoints
        keypoints.forEach((kp, index) => {
            if (kp.score > threshold) {
                const x = kp.x * scaleX;
                const y = kp.y * scaleY;
                
                // Outer circle
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
                
                // Inner circle
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fillStyle = '#ffffff';
                ctx.fill();
            }
        });
        
        // Draw label
        if (keypoints[0] && keypoints[0].score > threshold) {
            const noseX = keypoints[0].x * scaleX;
            const noseY = keypoints[0].y * scaleY;
            const label = `Person ${poseIndex + 1}`;
            
            ctx.font = 'bold 14px Segoe UI, sans-serif';
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = color;
            ctx.fillRect(noseX - textWidth/2 - 6, noseY - 50, textWidth + 12, 24);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, noseX - textWidth/2, noseY - 32);
        }
    });
}

// Draw hand detections (HandPose)
function drawHandDetections(hands, sourceWidth, sourceHeight) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scaleX = canvas.width / sourceWidth;
    const scaleY = canvas.height / sourceHeight;
    
    const handColors = ['#10b981', '#f59e0b'];
    
    hands.forEach((hand, handIndex) => {
        const keypoints = hand.keypoints;
        const color = handColors[handIndex % 2];
        
        // Draw connections
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        
        HAND_CONNECTIONS.forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];
            
            ctx.beginPath();
            ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY);
            ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY);
            ctx.stroke();
        });
        
        // Draw keypoints
        keypoints.forEach((kp, index) => {
            const x = kp.x * scaleX;
            const y = kp.y * scaleY;
            
            // Fingertips are larger
            const isFingertip = [4, 8, 12, 16, 20].includes(index);
            const radius = isFingertip ? 8 : 5;
            
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fillStyle = isFingertip ? '#ffffff' : color;
            ctx.fill();
            
            if (isFingertip) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        });
        
        // Draw label
        const wrist = keypoints[0];
        const label = `${hand.handedness} Hand`;
        
        ctx.font = 'bold 14px Segoe UI, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const labelX = wrist.x * scaleX;
        const labelY = wrist.y * scaleY + 30;
        
        ctx.fillStyle = color;
        ctx.fillRect(labelX - textWidth/2 - 6, labelY - 16, textWidth + 12, 24);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, labelX - textWidth/2, labelY + 2);
    });
}

// Generate natural language description for objects
function generateObjectDescription(predictions) {
    if (predictions.length === 0) {
        return '<p class="placeholder">No objects detected in the scene.</p>';
    }
    
    const counts = {};
    predictions.forEach(p => {
        counts[p.class] = (counts[p.class] || 0) + 1;
    });
    
    const items = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const totalObjects = predictions.length;
    
    // Build description parts
    const parts = items.map(([name, count]) => {
        const plural = count > 1 ? getPluralForm(name) : name;
        return `<span class="highlight">${count} ${plural}</span>`;
    });
    
    let description = '';
    
    if (parts.length === 1) {
        description = `The scene contains ${parts[0]}.`;
    } else if (parts.length === 2) {
        description = `The scene contains ${parts[0]} and ${parts[1]}.`;
    } else {
        const lastPart = parts.pop();
        description = `The scene contains ${parts.join(', ')}, and ${lastPart}.`;
    }
    
    // Add context based on what's detected
    const hasPersons = counts['person'] > 0;
    const hasVehicles = ['car', 'truck', 'bus', 'motorcycle', 'bicycle'].some(v => counts[v] > 0);
    const hasAnimals = ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep'].some(a => counts[a] > 0);
    
    if (hasPersons && hasVehicles) {
        description += ' This appears to be a street or traffic scene.';
    } else if (hasPersons && hasAnimals) {
        description += ' People are seen with animals.';
    } else if (hasPersons && counts['person'] > 3) {
        description += ' This looks like a crowded area.';
    }
    
    return `<p class="description-text">${description}</p>`;
}

// Get plural form of object names
function getPluralForm(word) {
    const irregulars = {
        'person': 'people',
        'mouse': 'mice',
        'knife': 'knives'
    };
    
    if (irregulars[word]) return irregulars[word];
    if (word.endsWith('s') || word.endsWith('x') || word.endsWith('ch') || word.endsWith('sh')) {
        return word + 'es';
    }
    if (word.endsWith('y') && !['a','e','i','o','u'].includes(word[word.length-2])) {
        return word.slice(0, -1) + 'ies';
    }
    return word + 's';
}

// Generate description for pose detection
function generatePoseDescription(poses) {
    if (poses.length === 0) {
        return '<p class="placeholder">No people detected in the scene.</p>';
    }
    
    const threshold = parseFloat(confidenceThreshold.value);
    const count = poses.length;
    const personWord = count === 1 ? 'person' : 'people';
    
    let description = `<span class="highlight">${count} ${personWord}</span> detected. `;
    
    poses.forEach((pose, i) => {
        const keypoints = pose.keypoints;
        const visibleCount = keypoints.filter(kp => kp.score > threshold).length;
        
        // Analyze pose
        const leftWrist = keypoints[9];
        const rightWrist = keypoints[10];
        const nose = keypoints[0];
        
        const handsRaised = (leftWrist?.score > threshold && leftWrist.y < nose?.y) ||
                          (rightWrist?.score > threshold && rightWrist.y < nose?.y);
        
        if (count === 1) {
            if (handsRaised) {
                description += 'The person has their hand raised.';
            } else if (visibleCount > 14) {
                description += 'Full body is visible.';
            } else if (visibleCount > 8) {
                description += 'Upper body is visible.';
            } else {
                description += 'Partial pose detected.';
            }
        }
    });
    
    if (count > 1) {
        description += 'Multiple body poses are being tracked.';
    }
    
    return `<p class="description-text">${description}</p>`;
}

// Generate description for hand detection
function generateHandDescription(hands) {
    if (hands.length === 0) {
        return '<p class="placeholder">No hands detected in the scene.</p>';
    }
    
    const count = hands.length;
    
    if (count === 1) {
        const hand = hands[0];
        return `<p class="description-text"><span class="highlight">1 ${hand.handedness.toLowerCase()} hand</span> detected with all 21 landmarks tracked.</p>`;
    } else if (count === 2) {
        const types = hands.map(h => h.handedness.toLowerCase()).join(' and ');
        return `<p class="description-text"><span class="highlight">2 hands</span> detected (${types}). Both hands are being tracked with full landmark coverage.</p>`;
    } else {
        return `<p class="description-text"><span class="highlight">${count} hands</span> detected and tracked.</p>`;
    }
}

// Update counts for object detection
function updateObjectCounts(predictions) {
    const counts = {};
    
    predictions.forEach(p => {
        counts[p.class] = (counts[p.class] || 0) + 1;
    });
    
    const total = predictions.length;
    totalCountEl.textContent = total;
    
    // Update description
    sceneDescriptionEl.innerHTML = generateObjectDescription(predictions);
    
    if (total === 0) {
        objectCountsEl.innerHTML = '<p class="placeholder">No objects detected</p>';
        return;
    }
    
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    
    objectCountsEl.innerHTML = sorted.map(([className, count]) => {
        const color = classColors[className] || classColors.default;
        return `
            <div class="count-item" style="border-left-color: ${color}">
                <span class="label">${className}</span>
                <span class="count" style="background: ${color}">${count}</span>
            </div>
        `;
    }).join('');
}

// Update counts for pose detection
function updatePoseCounts(poses) {
    const total = poses.length;
    totalCountEl.textContent = total;
    
    // Update description
    sceneDescriptionEl.innerHTML = generatePoseDescription(poses);
    
    if (total === 0) {
        objectCountsEl.innerHTML = '<p class="placeholder">No poses detected</p>';
        return;
    }
    
    const threshold = parseFloat(confidenceThreshold.value);
    
    objectCountsEl.innerHTML = poses.map((pose, index) => {
        const visibleKeypoints = pose.keypoints.filter(kp => kp.score > threshold).length;
        const color = index === 0 ? '#00d9ff' : '#7c3aed';
        return `
            <div class="count-item" style="border-left-color: ${color}">
                <span class="label">Person ${index + 1}</span>
                <span class="count" style="background: ${color}">${visibleKeypoints}/17 pts</span>
            </div>
        `;
    }).join('');
}

// Update counts for hand detection
function updateHandCounts(hands) {
    const total = hands.length;
    totalCountEl.textContent = total;
    
    // Update description
    sceneDescriptionEl.innerHTML = generateHandDescription(hands);
    
    if (total === 0) {
        objectCountsEl.innerHTML = '<p class="placeholder">No hands detected</p>';
        return;
    }
    
    const handColors = ['#10b981', '#f59e0b'];
    
    objectCountsEl.innerHTML = hands.map((hand, index) => {
        const color = handColors[index % 2];
        return `
            <div class="count-item" style="border-left-color: ${color}">
                <span class="label">${hand.handedness} Hand</span>
                <span class="count" style="background: ${color}">21 pts</span>
            </div>
        `;
    }).join('');
}

// Update FPS counter
function updateFPS() {
    frameCount++;
    const currentTime = performance.now();
    
    if (currentTime - lastTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastTime = currentTime;
        fpsValueEl.textContent = fps;
    }
}

// Add entry to detection log
function addLogEntry(message) {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-item';
    entry.innerHTML = `<span class="time">[${time}]</span>${message}`;
    
    const placeholder = detectionLogEl.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    detectionLogEl.insertBefore(entry, detectionLogEl.firstChild);
    
    while (detectionLogEl.children.length > 50) {
        detectionLogEl.removeChild(detectionLogEl.lastChild);
    }
}

// Event Listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopDetection);
imageUpload.addEventListener('change', handleImageUpload);

modelSelect.addEventListener('change', async (e) => {
    const newModel = e.target.value;
    if (newModel === currentModelType) return;
    
    // Stop current detection if running
    if (isDetecting) {
        stopDetection();
    }
    
    // Clear canvas and counts
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    totalCountEl.textContent = '0';
    objectCountsEl.innerHTML = '<p class="placeholder">Loading new model...</p>';
    
    // Load new model
    await loadModel(newModel);
    
    // Re-detect if image is loaded
    if (inputMode === 'image' && uploadedImage.src) {
        detectImage();
    }
});

confidenceThreshold.addEventListener('input', (e) => {
    const value = Math.round(e.target.value * 100);
    confidenceValue.textContent = `${value}%`;
    
    if (inputMode === 'image' && uploadedImage.src) {
        detectImage();
    }
});

maxDetections.addEventListener('input', (e) => {
    maxDetectionsValue.textContent = e.target.value;
    
    if (inputMode === 'image' && uploadedImage.src) {
        detectImage();
    }
});

window.addEventListener('resize', () => {
    if (inputMode === 'camera' && video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    } else if (inputMode === 'image' && uploadedImage.naturalWidth) {
        canvas.width = uploadedImage.naturalWidth;
        canvas.height = uploadedImage.naturalHeight;
    }
});

// Initialize on page load
init();
