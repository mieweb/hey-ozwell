/**
 * Audio processing utilities for wake-word detection
 */
export class AudioProcessor {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.frameSize = options.frameSize || 4096;
    this.hopLength = options.hopLength || 2048;
    this.nMels = options.nMels || 80;
    this.nFFT = options.nFFT || 2048;
  }
  
  /**
   * Convert audio samples to mel-spectrogram features
   * @param {Float32Array} audioData - Audio samples
   * @returns {Float32Array} Mel-spectrogram features
   */
  audioToMelSpectrogram(audioData) {
    // This is a simplified implementation
    // In production, you'd want a more sophisticated mel-spectrogram calculation
    // For now, we'll create a placeholder that returns the expected shape
    
    const numFrames = Math.floor((audioData.length - this.nFFT) / this.hopLength) + 1;
    const features = new Float32Array(this.nMels * numFrames);
    
    // Compute STFT and convert to mel scale
    for (let frame = 0; frame < numFrames; frame++) {
      const startSample = frame * this.hopLength;
      const frameData = audioData.slice(startSample, startSample + this.nFFT);
      
      // Apply window function (simplified Hann window)
      const windowedFrame = this.applyHannWindow(frameData);
      
      // Compute FFT (simplified - in production use a proper FFT library)
      const spectrum = this.computeSpectrum(windowedFrame);
      
      // Convert to mel scale
      const melFrame = this.spectrumToMel(spectrum);
      
      // Copy to output
      features.set(melFrame, frame * this.nMels);
    }
    
    return features;
  }
  
  /**
   * Apply Hann window to audio frame
   * @param {Float32Array} frame - Audio frame
   * @returns {Float32Array} Windowed frame
   */
  applyHannWindow(frame) {
    const windowed = new Float32Array(frame.length);
    
    for (let i = 0; i < frame.length; i++) {
      const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (frame.length - 1)));
      windowed[i] = frame[i] * window;
    }
    
    return windowed;
  }
  
  /**
   * Compute power spectrum (simplified)
   * @param {Float32Array} frame - Windowed audio frame
   * @returns {Float32Array} Power spectrum
   */
  computeSpectrum(frame) {
    // This is a very simplified spectrum calculation
    // In production, use a proper FFT implementation like fft.js
    const spectrum = new Float32Array(this.nFFT / 2 + 1);
    
    for (let i = 0; i < spectrum.length; i++) {
      // Simplified: just take magnitude of frame data
      let sum = 0;
      for (let j = 0; j < frame.length; j++) {
        const phase = 2 * Math.PI * i * j / frame.length;
        sum += frame[j] * Math.cos(phase);
      }
      spectrum[i] = Math.abs(sum);
    }
    
    return spectrum;
  }
  
  /**
   * Convert linear spectrum to mel scale
   * @param {Float32Array} spectrum - Linear spectrum
   * @returns {Float32Array} Mel-scale features
   */
  spectrumToMel(spectrum) {
    const melFeatures = new Float32Array(this.nMels);
    
    // Simplified mel conversion - map linear bins to mel bins
    const melBins = this.createMelFilterBank();
    
    for (let mel = 0; mel < this.nMels; mel++) {
      let sum = 0;
      const startBin = Math.floor(mel * spectrum.length / this.nMels);
      const endBin = Math.floor((mel + 1) * spectrum.length / this.nMels);
      
      for (let bin = startBin; bin < endBin; bin++) {
        sum += spectrum[bin];
      }
      
      melFeatures[mel] = Math.log(Math.max(sum, 1e-10)); // Log scale with epsilon
    }
    
    return melFeatures;
  }
  
  /**
   * Create mel filter bank (simplified)
   * @returns {Array} Mel filter bank
   */
  createMelFilterBank() {
    // Simplified implementation - in production use proper mel filter calculation
    const filterBank = [];
    
    for (let i = 0; i < this.nMels; i++) {
      filterBank.push(1.0); // Placeholder
    }
    
    return filterBank;
  }
  
  /**
   * Normalize audio data
   * @param {Float32Array} audioData - Raw audio samples
   * @returns {Float32Array} Normalized audio
   */
  normalizeAudio(audioData) {
    const normalized = new Float32Array(audioData.length);
    
    // Find max amplitude
    let maxAmp = 0;
    for (let i = 0; i < audioData.length; i++) {
      maxAmp = Math.max(maxAmp, Math.abs(audioData[i]));
    }
    
    // Normalize to prevent clipping
    const scale = maxAmp > 0 ? 0.95 / maxAmp : 1.0;
    
    for (let i = 0; i < audioData.length; i++) {
      normalized[i] = audioData[i] * scale;
    }
    
    return normalized;
  }
  
  /**
   * Resample audio to target sample rate
   * @param {Float32Array} audioData - Input audio
   * @param {number} sourceSampleRate - Source sample rate
   * @param {number} targetSampleRate - Target sample rate
   * @returns {Float32Array} Resampled audio
   */
  resampleAudio(audioData, sourceSampleRate, targetSampleRate) {
    if (sourceSampleRate === targetSampleRate) {
      return audioData;
    }
    
    const ratio = sourceSampleRate / targetSampleRate;
    const outputLength = Math.floor(audioData.length / ratio);
    const resampled = new Float32Array(outputLength);
    
    // Simple linear interpolation resampling
    for (let i = 0; i < outputLength; i++) {
      const sourceIndex = i * ratio;
      const index = Math.floor(sourceIndex);
      const fraction = sourceIndex - index;
      
      if (index + 1 < audioData.length) {
        resampled[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
      } else {
        resampled[i] = audioData[index];
      }
    }
    
    return resampled;
  }
  
  /**
   * Apply pre-emphasis filter
   * @param {Float32Array} audioData - Input audio
   * @param {number} alpha - Pre-emphasis coefficient (default: 0.97)
   * @returns {Float32Array} Filtered audio
   */
  preEmphasis(audioData, alpha = 0.97) {
    const filtered = new Float32Array(audioData.length);
    filtered[0] = audioData[0];
    
    for (let i = 1; i < audioData.length; i++) {
      filtered[i] = audioData[i] - alpha * audioData[i - 1];
    }
    
    return filtered;
  }
}