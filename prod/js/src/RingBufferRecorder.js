/**
 * Ring buffer recorder for maintaining pre-roll audio
 */
export class RingBufferRecorder {
  constructor(options = {}) {
    this.bufferDuration = options.bufferDuration || 30; // seconds
    this.sampleRate = options.sampleRate || 16000;
    this.bufferSize = this.bufferDuration * this.sampleRate;
    
    // Circular buffer for audio samples
    this.buffer = new Float32Array(this.bufferSize);
    this.writeIndex = 0;
    this.isFull = false;
    
    this.audioContext = null;
    this.sourceNode = null;
    this.processorNode = null;
    
    if (options.stream) {
      this.start(options.stream);
    }
  }
  
  /**
   * Start recording audio stream into ring buffer
   * @param {MediaStream} stream - Audio stream from getUserMedia
   */
  async start(stream) {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate
      });
      
      this.sourceNode = this.audioContext.createMediaStreamSource(stream);
      
      // Create AudioWorklet processor for efficient audio processing
      if (this.audioContext.audioWorklet) {
        await this.audioContext.audioWorklet.addModule(this.createWorkletCode());
        this.processorNode = new AudioWorkletNode(this.audioContext, 'ring-buffer-processor');
        
        // Handle audio data from worklet
        this.processorNode.port.onmessage = (event) => {
          const { audioData } = event.data;
          this.writeToBuffer(audioData);
        };
      } else {
        // Fallback to ScriptProcessorNode (deprecated but more compatible)
        this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
        this.processorNode.onaudioprocess = (event) => {
          const inputBuffer = event.inputBuffer.getChannelData(0);
          this.writeToBuffer(inputBuffer);
        };
      }
      
      this.sourceNode.connect(this.processorNode);
      this.processorNode.connect(this.audioContext.destination);
      
      console.log('Ring buffer recorder started');
    } catch (error) {
      console.error('Error starting ring buffer recorder:', error);
      throw error;
    }
  }
  
  /**
   * Stop recording
   */
  stop() {
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
    
    console.log('Ring buffer recorder stopped');
  }
  
  /**
   * Write audio data to circular buffer
   * @param {Float32Array} audioData - Audio samples
   */
  writeToBuffer(audioData) {
    for (let i = 0; i < audioData.length; i++) {
      this.buffer[this.writeIndex] = audioData[i];
      this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
      
      if (this.writeIndex === 0) {
        this.isFull = true;
      }
    }
  }
  
  /**
   * Get audio buffer content
   * @param {number} duration - Duration in seconds to retrieve (default: all)
   * @returns {Float32Array} Audio samples
   */
  getBuffer(duration) {
    const requestedSamples = duration ? Math.min(duration * this.sampleRate, this.bufferSize) : this.bufferSize;
    const actualSamples = this.isFull ? this.bufferSize : this.writeIndex;
    const samplesToReturn = Math.min(requestedSamples, actualSamples);
    
    const result = new Float32Array(samplesToReturn);
    
    if (this.isFull) {
      // Buffer is full, need to handle wrap-around
      const startIndex = (this.writeIndex - samplesToReturn + this.bufferSize) % this.bufferSize;
      
      if (startIndex + samplesToReturn <= this.bufferSize) {
        // No wrap-around needed
        result.set(this.buffer.subarray(startIndex, startIndex + samplesToReturn));
      } else {
        // Handle wrap-around
        const firstPart = this.bufferSize - startIndex;
        const secondPart = samplesToReturn - firstPart;
        
        result.set(this.buffer.subarray(startIndex), 0);
        result.set(this.buffer.subarray(0, secondPart), firstPart);
      }
    } else {
      // Buffer not full yet, simple case
      const startIndex = Math.max(0, this.writeIndex - samplesToReturn);
      result.set(this.buffer.subarray(startIndex, this.writeIndex));
    }
    
    return result;
  }
  
  /**
   * Clear the ring buffer
   */
  clear() {
    this.buffer.fill(0);
    this.writeIndex = 0;
    this.isFull = false;
  }
  
  /**
   * Get the current buffer size in samples
   * @returns {number} Number of audio samples currently in buffer
   */
  getCurrentSize() {
    return this.isFull ? this.bufferSize : this.writeIndex;
  }
  
  /**
   * Get the current buffer duration in seconds
   * @returns {number} Duration in seconds
   */
  getCurrentDuration() {
    return this.getCurrentSize() / this.sampleRate;
  }
  
  /**
   * Create AudioWorklet processor code (as a string)
   * @returns {string} Worklet processor code
   */
  createWorkletCode() {
    const workletCode = `
      class RingBufferProcessor extends AudioWorkletProcessor {
        process(inputs, outputs, parameters) {
          const input = inputs[0];
          if (input && input[0]) {
            // Send audio data to main thread
            this.port.postMessage({
              audioData: input[0]
            });
          }
          return true; // Keep processor alive
        }
      }
      
      registerProcessor('ring-buffer-processor', RingBufferProcessor);
    `;
    
    // Create blob URL for the worklet
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
  }
}