const express = require('express');
const path = require('path');

const app = express();
const port = process.env.PORT || 3001;

// Serve static files from the "src" directory
app.use(express.static(path.join(__dirname, 'src')));

// Serve the "pretrained" directory for pre-trained models
app.use("/pretrained", express.static(path.join(__dirname, 'pretrained')));

// Serve the "models" directory for custom models
app.use("/models", express.static(path.join(__dirname, 'models')));

// Serve the "asr" directory (stage-2 whisper WASM verifier artifacts)
app.use("/asr", express.static(path.join(__dirname, 'asr')));

// Serve the index.html file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
    console.log(`Development server running at http://localhost:${port}`);
});
