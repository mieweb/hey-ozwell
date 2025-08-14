import { AudioProcessor } from './AudioProcessor.js';

/**
 * Main wake-word listener class
 */
export class WakeListener extends EventTarget {
  constructor(options = {}) {
    super();
    
    this.threshold = options.threshold || 0.7;
    this.bufferSize = options.bufferSize || 4096;
    this.sampleRate = options.sampleRate || 16000;
    this.windowDuration = options.windowDuration || 2.0; // seconds
    
    this.audioContext = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.isListening = false;
    
    this.models = {};
    this.onWakeCallback = null;
    
    // Audio processing
    this.audioProcessor = new AudioProcessor({
      sampleRate: this.sampleRate,
      frameSize: this.bufferSize
    });
    
    // Sliding window for audio analysis
    this.windowSize = this.windowDuration * this.sampleRate;
    this.audioWindow = new Float32Array(this.windowSize);
    this.windowIndex = 0;
    
    // Detection state
    this.lastDetectionTime = 0;
    this.detectionCooldown = 1000; // ms between detections
    
    console.log('WakeListener initialized');
  }
  
  /**
   * Start wake-word detection
   * @param {MediaStream} stream - Audio stream from getUserMedia
   * @param {Object} config - Configuration object
   * @param {Object} config.models - Loaded ONNX models
   * @param {Function} config.onWake - Callback for wake detection
   */
  async start(stream, config = {}) {
    if (this.isListening) {
      console.warn('Already listening');
      return;
    }
    
    this.models = config.models || {};
    this.onWakeCallback = config.onWake;
    
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate
      });
      
      this.sourceNode = this.audioContext.createMediaStreamSource(stream);
      
      // Use AudioWorklet if available, fallback to ScriptProcessor
      if (this.audioContext.audioWorklet) {
        await this.setupAudioWorklet();
      } else {
        this.setupScriptProcessor();
      }
      
      this.sourceNode.connect(this.processorNode);
      // Note: Don't connect to destination to avoid feedback
      
      this.isListening = true;
      
      console.log('Wake-word detection started');
      this.dispatchEvent(new CustomEvent('started'));
      
    } catch (error) {
      console.error('Error starting wake listener:', error);
      throw error;
    }
  }
  
  /**
   * Stop wake-word detection
   */
  stop() {
    if (!this.isListening) {
      return;
    }
    
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    this.isListening = false;
    
    console.log('Wake-word detection stopped');
    this.dispatchEvent(new CustomEvent('stopped'));
  }
  
  /**
   * Set up AudioWorklet processor
   */
  async setupAudioWorklet() {
    await this.audioContext.audioWorklet.addModule(this.createWorkletCode());
    this.processorNode = new AudioWorkletNode(this.audioContext, 'wake-listener-processor', {
      processorOptions: {
        bufferSize: this.bufferSize
      }
    });
    
    this.processorNode.port.onmessage = (event) => {
      const { audioData } = event.data;
      this.processAudioData(audioData);
    };
  }
  
  /**
   * Set up ScriptProcessor (fallback)
   */
  setupScriptProcessor() {
    this.processorNode = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
    this.processorNode.onaudioprocess = (event) => {
      const inputBuffer = event.inputBuffer.getChannelData(0);
      this.processAudioData(inputBuffer);
    };
  }
  
  /**
   * Process incoming audio data
   * @param {Float32Array} audioData - Audio samples
   */
  processAudioData(audioData) {
    // Add to sliding window
    this.addToWindow(audioData);
    
    // Check if we have enough data for inference
    if (this.windowIndex >= this.windowSize) {
      this.runInference();
      
      // Shift window by half to create overlap
      const shiftSize = this.windowSize / 2;
      this.audioWindow.copyWithin(0, shiftSize);
      this.windowIndex = this.windowSize - shiftSize;
    }
  }
  
  /**
   * Add audio data to sliding window
   * @param {Float32Array} audioData - Audio samples
   */
  addToWindow(audioData) {
    for (let i = 0; i < audioData.length && this.windowIndex < this.windowSize; i++) {
      this.audioWindow[this.windowIndex++] = audioData[i];
    }
  }
  
  /**
   * Run wake-word inference on current window
   */
  async runInference() {
    try {
      // Skip if in cooldown period
      const now = Date.now();
      if (now - this.lastDetectionTime < this.detectionCooldown) {
        return;
      }
      
      // Normalize and process audio
      const normalizedAudio = this.audioProcessor.normalizeAudio(this.audioWindow);
      
      // For now, we'll use a simple energy-based detection as placeholder
      // In production, this would run the ONNX models
      const detection = await this.detectWakeWord(normalizedAudio);
      
      if (detection.detected && detection.confidence > this.threshold) {
        this.lastDetectionTime = now;
        
        console.log(`Wake word detected: ${detection.label} (confidence: ${detection.confidence})`);
        
        this.dispatchEvent(new CustomEvent('wake', {
          detail: {
            label: detection.label,
            confidence: detection.confidence,
            timestamp: now
          }
        }));
        
        if (this.onWakeCallback) {
          this.onWakeCallback(detection.label, detection.confidence);
        }
      }
      
    } catch (error) {
      console.error('Error during inference:', error);
    }
  }
  
  /**
   * Detect wake words in audio (placeholder implementation)
   * @param {Float32Array} audioData - Processed audio data
   * @returns {Promise<Object>} Detection result
   */
  async detectWakeWord(audioData) {
    // This is a placeholder implementation
    // In production, this would:
    // 1. Convert audio to mel-spectrogram
    // 2. Run ONNX inference for each model
    // 3. Return the highest confidence detection above threshold
    
    // Simple energy-based detection for demonstration
    let energy = 0;
    for (let i = 0; i < audioData.length; i++) {
      energy += audioData[i] * audioData[i];
    }
    energy = Math.sqrt(energy / audioData.length);
    
    // Simulate different wake phrases based on energy patterns
    const phrases = ['hey-ozwell', 'im-done', 'go-ozwell', 'ozwell-go'];
    const randomPhrase = phrases[Math.floor(Math.random() * phrases.length)];
    
    // Simulate detection based on energy threshold
    if (energy > 0.01) { // Arbitrary threshold for demo
      return {
        detected: true,
        label: randomPhrase,
        confidence: Math.min(energy * 10, 1.0) // Scale energy to confidence
      };
    }
    
    return {
      detected: false,
      label: null,
      confidence: 0
    };
  }
  
  /**
   * Update detection threshold
   * @param {number} threshold - New threshold (0.0-1.0)
   */
  setThreshold(threshold) {
    this.threshold = Math.max(0, Math.min(1, threshold));
    console.log(`Detection threshold updated to ${this.threshold}`);
  }
  
  /**
   * Get current listening state
   * @returns {boolean} True if listening
   */
  getIsListening() {
    return this.isListening;
  }
  
  /**
   * Create AudioWorklet processor code
   * @returns {string} Blob URL for worklet code
   */
  createWorkletCode() {
    const workletCode = `
      class WakeListenerProcessor extends AudioWorkletProcessor {
        constructor(options) {
          super();
          this.bufferSize = options.processorOptions?.bufferSize || 4096;
        }
        
        process(inputs, outputs, parameters) {
          const input = inputs[0];
          
          if (input && input[0] && input[0].length > 0) {
            // Send audio data to main thread
            this.port.postMessage({
              audioData: input[0].slice() // Copy array
            });
          }
          
          return true; // Keep processor alive
        }
      }
      
      registerProcessor('wake-listener-processor', WakeListenerProcessor);
    `;
    
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
  }
}