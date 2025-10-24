const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'vegetable-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (extname && mimetype) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'));
    }
  }
});

const { spawn } = require('child_process');

// Detect the correct Python command
let PYTHON_CMD = null;

function detectPythonCommand() {
  const commands = ['python', 'python3', 'py'];
  
  for (const cmd of commands) {
    try {
      const result = require('child_process').spawnSync(cmd, ['--version'], { encoding: 'utf8' });
      if (result.status === 0) {
        console.log(`Detected Python command: ${cmd} (${result.stdout.trim()})`);
        return cmd;
      }
    } catch (err) {
      // Command not found, continue
    }
  }
  
  console.warn('Warning: Could not detect Python command, defaulting to "python"');
  return 'python';
}

// YOLO prediction function
async function predictVegetable(imagePath) {
  return new Promise((resolve, reject) => {
    console.log('Running YOLO prediction on:', imagePath);
    
    // Use detected Python command
    if (!PYTHON_CMD) {
      PYTHON_CMD = detectPythonCommand();
    }
    
    // Run Python script with the image path
    const pythonProcess = spawn(PYTHON_CMD, ['predict.py', imagePath], {
      env: { 
        ...process.env,
        OMP_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
        KMP_DUPLICATE_LIB_OK: 'TRUE'
      }
    });
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python script error:', stderr);
        reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        return;
      }
      
      try {
        // Parse the JSON output from Python script
        const result = JSON.parse(stdout.trim());
        console.log('YOLO prediction result:', result);
        resolve(result);
      } catch (error) {
        console.error('Failed to parse Python output:', stdout);
        reject(new Error('Failed to parse prediction results'));
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      reject(new Error('Failed to run prediction script'));
    });
  });
}

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    console.log('Processing image:', req.file.filename);
    
    // Make prediction using YOLO
    const predictions = await predictVegetable(req.file.path);
    
    // Don't delete the file immediately - let Python script handle it
    // The Python script might need time to process
    setTimeout(() => {
      fs.unlink(req.file.path, (err) => {
        if (err) console.error('Error deleting file:', err);
      });
    }, 5000); // Delete after 5 seconds
    
    res.json({
      success: true,
      predictions: predictions,
      filename: req.file.filename
    });
    
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Failed to process image',
      details: error.message 
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large. Maximum size is 5MB.' });
    }
  }
  res.status(500).json({ error: error.message });
});

// Preload model on server startup
function warmupModel() {
  console.log('Warming up YOLO model...');
  const { spawn } = require('child_process');
  
  // Try different Python commands for cross-platform compatibility
  const pythonCommands = ['python', 'python3', 'py'];
  let success = false;
  
  for (const pythonCmd of pythonCommands) {
    try {
      const warmupProcess = spawn(pythonCmd, ['-c', 'from ultralytics import YOLO; model = YOLO("best (5).pt"); print("Model loaded")'], {
        cwd: __dirname,
        env: { 
          ...process.env,
          OMP_NUM_THREADS: '1',
          MKL_NUM_THREADS: '1',
          KMP_DUPLICATE_LIB_OK: 'TRUE'
        }
      });
      
      warmupProcess.stdout.on('data', (data) => {
        console.log(`Model warmup (${pythonCmd}):`, data.toString().trim());
      });
      
      warmupProcess.stderr.on('data', (data) => {
        console.error(`Warmup stderr (${pythonCmd}):`, data.toString().trim());
      });
      
      warmupProcess.on('error', (err) => {
        if (!success) {
          console.log(`${pythonCmd} not found, trying next...`);
        }
      });
      
      warmupProcess.on('close', (code) => {
        if (code === 0) {
          console.log(`✓ Model preloaded and ready with ${pythonCmd}!`);
          success = true;
        } else if (code !== 9009 && code !== null) {
          console.log(`Model warmup with ${pythonCmd} completed with code:`, code);
        }
      });
      
      // Break after first successful spawn attempt
      break;
    } catch (err) {
      console.log(`Failed to spawn ${pythonCmd}, trying next...`);
    }
  }
}

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('Upload your vegetable images to get predictions!');
  
  // Warmup model after server starts
  setTimeout(warmupModel, 1000);
});
