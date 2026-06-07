using UnityEngine;
using Unity.InferenceEngine;
using System;

namespace CPritch.DepthForge.Editor.Inference
{
    public static class ImageProcessor
    {
        /// <summary>
        /// Converts a Texture2D to a Tensor for Sentis inference.
        /// Depth Anything models expect input normalized using ImageNet stats.
        /// Resolution is typical 518x518.
        /// </summary>
        public static Tensor<float> PreprocessTexture(Texture inputTex, int targetWidth = 518, int targetHeight = 518, int inputRank = 4)
        {
            // Convert Texture to Tensor
            // Notice: TextureConverter handles resizing.
            Tensor<float> tensor = new Tensor<float>(new TensorShape(1, 3, targetHeight, targetWidth));
            TextureConverter.ToTensor(inputTex, tensor, default);
            
            // Download data to CPU to perform ImageNet normalization
            float[] rawInput = tensor.DownloadToArray();
            
            int channelSize = targetHeight * targetWidth;
            for (int c = 0; c < 3; c++)
            {
                float meanVal = c == 0 ? 0.485f : (c == 1 ? 0.456f : 0.406f);
                float stdVal = c == 0 ? 0.229f : (c == 1 ? 0.224f : 0.225f);
                int offset = c * channelSize;
                
                for (int i = 0; i < channelSize; i++)
                {
                    rawInput[offset + i] = (rawInput[offset + i] - meanVal) / stdVal;
                }
            }
            
            // Upload normalized values back to the tensor
            tensor.Upload(rawInput);
            
            // Reshape the 4D tensor to 5D [1, 1, 3, targetHeight, targetWidth] only if expected by the model (e.g. video models)
            if (inputRank == 5)
            {
                tensor.Reshape(new TensorShape(1, 1, 3, targetHeight, targetWidth));
            }
            
            return tensor;
        }

        /// <summary>
        /// Converts the output Tensor (1 channel depth map) back to a Unity Texture2D.
        /// </summary>
        public static Texture2D PostprocessTensor(Tensor<float> outputTensor, int originalWidth, int originalHeight)
        {
            int rank = outputTensor.shape.rank;
            if (rank < 2)
            {
                Debug.LogError($"Output tensor has invalid rank: {rank}");
                return null;
            }

            int height = outputTensor.shape[rank - 2];
            int width = outputTensor.shape[rank - 1];

            // Download the output tensor data to CPU
            float[] rawData = outputTensor.DownloadToArray();

            // Find min/max for normalization
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;
            for (int i = 0; i < rawData.Length; i++)
            {
                float v = rawData[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }

            float range = maxVal - minVal;
            if (range < 1e-5f) range = 1e-5f;

            // Create a grayscale texture (RGB24) to display correctly in the UI and be fully compatible with PNG encoding
            Texture2D outTex = new Texture2D(width, height, TextureFormat.RGB24, false, true);
            Color32[] colors = new Color32[width * height];

            for (int y = 0; y < height; y++)
            {
                // Flip the Y-axis because Unity's texture origin is bottom-left,
                // whereas neural network output coordinates start from top-left.
                int tensorY = height - 1 - y;
                
                for (int x = 0; x < width; x++)
                {
                    int tensorIdx = tensorY * width + x;
                    float rawVal = rawData[tensorIdx];
                    float normVal = (rawVal - minVal) / range;
                    
                    byte grayValue = (byte)Mathf.Clamp(normVal * 255f, 0f, 255f);
                    
                    int texIdx = y * width + x;
                    colors[texIdx] = new Color32(grayValue, grayValue, grayValue, 255);
                }
            }

            outTex.SetPixels32(colors);
            outTex.Apply();

            return outTex;
        }

        /// <summary>
        /// Applies Contrast, Midpoint, and Invert adjustments to a heightmap texture in-memory.
        /// </summary>
        public static Texture2D ApplyAdjustments(Texture2D source, float contrast, float midpoint, bool invert)
        {
            if (source == null) return null;

            int w = source.width;
            int h = source.height;

            Texture2D adjusted = new Texture2D(w, h, TextureFormat.RGB24, false, true);
            Color32[] pixels = source.GetPixels32();

            for (int i = 0; i < pixels.Length; i++)
            {
                float val = pixels[i].r / 255f;

                if (invert)
                {
                    val = 1f - val;
                }

                // Apply contrast around midpoint
                val = Mathf.Clamp01((val - midpoint) * contrast + midpoint);

                byte grayValue = (byte)Mathf.Clamp(val * 255f, 0f, 255f);
                pixels[i] = new Color32(grayValue, grayValue, grayValue, 255);
            }

            // Fix border artifacts: copy adjacent pixels to the outer 2-pixel border
            // This removes neural network padding anomalies without losing details or adding sloped pill shapes
            int borderSize = 2;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int clampedX = Mathf.Clamp(x, borderSize, w - 1 - borderSize);
                    int clampedY = Mathf.Clamp(y, borderSize, h - 1 - borderSize);
                    
                    if (clampedX != x || clampedY != y)
                    {
                        pixels[y * w + x] = pixels[clampedY * w + clampedX];
                    }
                }
            }

            adjusted.SetPixels32(pixels);
            adjusted.Apply();

            return adjusted;
        }
    }
}
