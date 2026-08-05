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

// Serve the speaker-verification WASM runtime (single-threaded slim build).
// Kept separate from sv-wasm/ (the frozen feasibility proof served on :3010).
app.use("/sv-runtime", express.static(path.join(__dirname, 'sv-runtime')));

// Serve the index.html file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
    console.log(`Development server running at http://localhost:${port}`);
});
