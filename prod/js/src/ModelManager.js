import { get, set, del, clear } from 'idb-keyval';

/**
 * Manages ONNX model loading and IndexedDB caching
 */
export class ModelManager {
  static CACHE_PREFIX = 'hey-ozwell-model-';
  
  /**
   * Fetch and cache ONNX models
   * @param {string} baseUrl - Base URL for model files
   * @param {Object} config - Map of label to filename
   * @returns {Promise<Object>} Map of label to loaded ONNX models
   */
  static async fetchAndCache(baseUrl, config) {
    const models = {};
    
    for (const [label, filename] of Object.entries(config)) {
      const cacheKey = this.CACHE_PREFIX + label;
      
      try {
        // Try to load from cache first
        let modelData = await get(cacheKey);
        
        if (!modelData) {
          console.log(`Fetching model ${label} from ${baseUrl}${filename}`);
          const response = await fetch(`${baseUrl}${filename}`);
          
          if (!response.ok) {
            throw new Error(`Failed to fetch ${filename}: ${response.statusText}`);
          }
          
          modelData = await response.arrayBuffer();
          
          // Cache the model data
          await set(cacheKey, modelData);
          console.log(`Cached model ${label} in IndexedDB`);
        } else {
          console.log(`Loaded model ${label} from cache`);
        }
        
        // Create ONNX session (this will be implemented when we have ONNX models)
        models[label] = {
          data: modelData,
          label,
          filename
        };
        
      } catch (error) {
        console.error(`Error loading model ${label}:`, error);
        throw error;
      }
    }
    
    return models;
  }
  
  /**
   * Get a cached model by label
   * @param {string} label - Model label
   * @returns {Promise<ArrayBuffer|null>} Cached model data
   */
  static async getCachedModel(label) {
    const cacheKey = this.CACHE_PREFIX + label;
    return await get(cacheKey);
  }
  
  /**
   * Clear all cached models
   * @returns {Promise<void>}
   */
  static async clearCache() {
    try {
      // Get all keys and filter for our models
      const keys = await this.getAllCacheKeys();
      const deletePromises = keys.map(key => del(key));
      await Promise.all(deletePromises);
      console.log('Cleared all cached models');
    } catch (error) {
      console.error('Error clearing cache:', error);
      throw error;
    }
  }
  
  /**
   * Get all cache keys for our models
   * @returns {Promise<string[]>} Array of cache keys
   */
  static async getAllCacheKeys() {
    // Note: idb-keyval doesn't provide a way to list keys
    // In a real implementation, we'd maintain a separate index
    // For now, we'll return the known model labels
    const knownLabels = ['hey-ozwell', 'im-done', 'go-ozwell', 'ozwell-go'];
    return knownLabels.map(label => this.CACHE_PREFIX + label);
  }
  
  /**
   * Check if a model is cached
   * @param {string} label - Model label
   * @returns {Promise<boolean>} True if cached
   */
  static async isCached(label) {
    const cacheKey = this.CACHE_PREFIX + label;
    const data = await get(cacheKey);
    return data !== undefined;
  }
}