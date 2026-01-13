# Mask Generation API Guide for HTML/JavaScript

This guide explains how to use the `/api/generate_mask` endpoint in a web application.

## API Endpoint

**POST** `/api/generate_mask`

## Workflow Overview

1. Upload a video and get a `video_id`
2. Display the video frame where user wants to segment
3. Collect user clicks (positive/negative points)
4. Send points to the API
5. Receive and display the mask

---

## Step-by-Step Implementation

### 1. Upload Video

```javascript
async function uploadVideo(videoFile) {
    const formData = new FormData();
    formData.append('file', videoFile);

    const response = await fetch('http://localhost:8000/api/upload_video', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    return data.video_id; // Save this for later use
}
```

### 2. Display Video Frame and Collect Points

```html
<!DOCTYPE html>
<html>
<head>
    <title>SAM2 Mask Generation</title>
    <style>
        #canvas-container {
            position: relative;
            display: inline-block;
        }
        #video-canvas {
            border: 2px solid #333;
            cursor: crosshair;
        }
        .point-marker {
            position: absolute;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 2px solid white;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
        .positive-point {
            background-color: green;
        }
        .negative-point {
            background-color: red;
        }
        #mask-canvas {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <h1>SAM2 Mask Generation</h1>

    <div>
        <input type="file" id="video-input" accept="video/*">
        <button onclick="uploadVideoFile()">Upload Video</button>
    </div>

    <div>
        <label>
            <input type="radio" name="point-type" value="1" checked>
            Positive Point (Green)
        </label>
        <label>
            <input type="radio" name="point-type" value="0">
            Negative Point (Red)
        </label>
        <button onclick="clearPoints()">Clear Points</button>
        <button onclick="generateMask()">Generate Mask</button>
    </div>

    <div id="canvas-container">
        <canvas id="video-canvas"></canvas>
        <canvas id="mask-canvas"></canvas>
    </div>

    <script src="mask-generation.js"></script>
</body>
</html>
```

### 3. JavaScript Implementation

```javascript
// mask-generation.js

let videoId = null;
let videoElement = null;
let points = []; // Store {x, y, label} in relative coordinates (0-1)
let canvasWidth, canvasHeight;

// Upload video
async function uploadVideoFile() {
    const fileInput = document.getElementById('video-input');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a video file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/api/upload_video', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        videoId = data.video_id;

        console.log('Video uploaded:', videoId);
        alert('Video uploaded successfully!');

        // Load video for display
        loadVideo(file);
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload video');
    }
}

// Load and display video
function loadVideo(file) {
    const canvas = document.getElementById('video-canvas');
    const ctx = canvas.getContext('2d');

    videoElement = document.createElement('video');
    videoElement.src = URL.createObjectURL(file);

    videoElement.addEventListener('loadeddata', () => {
        // Set canvas size to match video
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        canvasWidth = canvas.width;
        canvasHeight = canvas.height;

        // Also set mask canvas size
        const maskCanvas = document.getElementById('mask-canvas');
        maskCanvas.width = canvasWidth;
        maskCanvas.height = canvasHeight;

        // Draw first frame
        videoElement.currentTime = 0;
    });

    videoElement.addEventListener('seeked', () => {
        ctx.drawImage(videoElement, 0, 0, canvasWidth, canvasHeight);
    });

    // Add click handler for points
    canvas.addEventListener('click', handleCanvasClick);
}

// Handle canvas clicks to add points
function handleCanvasClick(event) {
    if (!videoId) {
        alert('Please upload a video first');
        return;
    }

    const canvas = document.getElementById('video-canvas');
    const rect = canvas.getBoundingClientRect();

    // Get click position relative to canvas
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert to relative coordinates (0-1)
    const relativeX = x / canvasWidth;
    const relativeY = y / canvasHeight;

    // Get selected point type (positive or negative)
    const pointType = document.querySelector('input[name="point-type"]:checked').value;
    const label = parseInt(pointType);

    // Add point to array
    points.push({
        x: relativeX,
        y: relativeY,
        label: label,
        displayX: x,  // Store for visual display
        displayY: y
    });

    // Draw point marker
    drawPointMarker(x, y, label);

    console.log('Point added:', { x: relativeX, y: relativeY, label });
}

// Draw visual marker for point
function drawPointMarker(x, y, label) {
    const container = document.getElementById('canvas-container');
    const marker = document.createElement('div');
    marker.className = `point-marker ${label === 1 ? 'positive-point' : 'negative-point'}`;
    marker.style.left = x + 'px';
    marker.style.top = y + 'px';
    container.appendChild(marker);
}

// Clear all points
function clearPoints() {
    points = [];

    // Remove visual markers
    const markers = document.querySelectorAll('.point-marker');
    markers.forEach(marker => marker.remove());

    // Clear mask
    const maskCanvas = document.getElementById('mask-canvas');
    const ctx = maskCanvas.getContext('2d');
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

    console.log('Points cleared');
}

// Generate mask from points
async function generateMask() {
    if (!videoId) {
        alert('Please upload a video first');
        return;
    }

    if (points.length === 0) {
        alert('Please add at least one point');
        return;
    }

    // Get current video time
    const videoTime = videoElement ? videoElement.currentTime : 0;

    // Prepare request
    const requestData = {
        video_id: videoId,
        video_time: videoTime,
        points: points.map(p => ({
            x: p.x,
            y: p.y,
            label: p.label
        }))
    };

    console.log('Generating mask with:', requestData);

    try {
        const response = await fetch('http://localhost:8000/api/generate_mask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Mask generated:', data);

        // Display the mask
        displayMask(data.mask);

    } catch (error) {
        console.error('Mask generation error:', error);
        alert('Failed to generate mask: ' + error.message);
    }
}

// Display mask on canvas
function displayMask(maskBase64) {
    const maskCanvas = document.getElementById('mask-canvas');
    const ctx = maskCanvas.getContext('2d');

    // Create image from base64
    const img = new Image();
    img.onload = () => {
        // Clear previous mask
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

        // Draw mask with color overlay
        ctx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);

        // Apply color tint (optional)
        ctx.globalCompositeOperation = 'source-in';
        ctx.fillStyle = 'rgba(0, 255, 0, 0.5)'; // Green tint
        ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        ctx.globalCompositeOperation = 'source-over';

        console.log('Mask displayed');
    };

    img.src = 'data:image/png;base64,' + maskBase64;
}
```

---

## API Request Format

```javascript
{
    "video_id": "abc123...",
    "video_time": 1.5,  // Time in seconds
    "points": [
        {
            "x": 0.5,    // Relative X (0-1)
            "y": 0.5,    // Relative Y (0-1)
            "label": 1   // 1 = positive, 0 = negative
        },
        // ... more points
    ]
}
```

## API Response Format

```javascript
{
    "status": "success",
    "mask": "iVBORw0KGgoAAAANS...",  // Base64 PNG image
    "video_id": "abc123...",
    "frame_time": 1.5
}
```

---

## Important Notes

### Coordinate System
- **Use relative coordinates (0-1)** for all points
- `x: 0.0` is left edge, `x: 1.0` is right edge
- `y: 0.0` is top edge, `y: 1.0` is bottom edge
- This makes the API resolution-independent

### Point Labels
- `label: 1` = **Positive point** (click inside the object you want to segment)
- `label: 0` = **Negative point** (click outside to exclude unwanted areas)

### CORS Configuration
The server has CORS enabled for all origins. If you need to restrict this:
```python
# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specific origin
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Example Use Cases

### Use Case 1: Simple Click-to-Segment
User clicks once on an object → Generate mask

```javascript
// Just one positive point
const requestData = {
    video_id: videoId,
    video_time: 0,
    points: [
        { x: 0.5, y: 0.5, label: 1 }
    ]
};
```

### Use Case 2: Refine Segmentation
User clicks positive points on object + negative points on background

```javascript
const requestData = {
    video_id: videoId,
    video_time: 0,
    points: [
        { x: 0.5, y: 0.5, label: 1 },   // Center of object
        { x: 0.48, y: 0.52, label: 1 }, // Another part
        { x: 0.2, y: 0.2, label: 0 },   // Background
        { x: 0.8, y: 0.8, label: 0 }    // More background
    ]
};
```

### Use Case 3: Multiple Frames
Generate masks for different frames in the video

```javascript
for (let time = 0; time < videoDuration; time += 1) {
    const mask = await generateMaskAtTime(videoId, time, points);
    // Process mask
}
```

---

## Debugging Tips

1. **Check console logs**: All functions log useful debugging info
2. **Verify video upload**: Make sure `videoId` is set before generating masks
3. **Check coordinates**: Points should be between 0 and 1
4. **Inspect response**: Use browser dev tools to see the actual API response
5. **Test with server**: Make sure the FastAPI server is running on `http://localhost:8000`

---

## Complete Minimal Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Quick SAM2 Test</title>
</head>
<body>
    <input type="file" id="video" accept="video/*">
    <button onclick="test()">Upload & Test</button>
    <canvas id="canvas" width="640" height="480"></canvas>

    <script>
        let videoId;

        async function test() {
            // 1. Upload video
            const file = document.getElementById('video').files[0];
            const formData = new FormData();
            formData.append('file', file);

            const uploadRes = await fetch('http://localhost:8000/api/upload_video', {
                method: 'POST',
                body: formData
            });
            videoId = (await uploadRes.json()).video_id;
            console.log('Video ID:', videoId);

            // 2. Generate mask with center point
            const maskRes = await fetch('http://localhost:8000/api/generate_mask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    video_id: videoId,
                    video_time: 0,
                    points: [{x: 0.5, y: 0.5, label: 1}]
                })
            });

            const data = await maskRes.json();
            console.log('Mask received');

            // 3. Display mask
            const img = new Image();
            img.onload = () => {
                const ctx = document.getElementById('canvas').getContext('2d');
                ctx.drawImage(img, 0, 0);
            };
            img.src = 'data:image/png;base64,' + data.mask;
        }
    </script>
</body>
</html>
```

This minimal example uploads a video, generates a mask at the center point, and displays it.
