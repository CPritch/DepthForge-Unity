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
        /// Applies Contrast, Midpoint, Invert, and High-Pass Shape Flattening adjustments to a heightmap texture.
        /// </summary>
        public static Texture2D ApplyAdjustments(Texture2D source, float contrast, float midpoint, bool invert, float flattenAmount = 0f)
        {
            if (source == null) return null;

            int w = source.width;
            int h = source.height;

            Texture2D adjusted = new Texture2D(w, h, TextureFormat.RGB24, false, true);
            Color32[] pixels = source.GetPixels32();
            float[] heightValues = new float[pixels.Length];

            // 1. Apply baseline adjustments (invert, contrast, midpoint)
            for (int i = 0; i < pixels.Length; i++)
            {
                float val = pixels[i].r / 255f;

                if (invert)
                {
                    val = 1f - val;
                }

                // Apply contrast around midpoint
                val = Mathf.Clamp01((val - midpoint) * contrast + midpoint);
                heightValues[i] = val;
            }

            // 2. Apply High-Pass detrending if flattenAmount > 0
            if (flattenAmount > 0f)
            {
                int radius = Mathf.Max(2, w / 16);
                float[] blurred = BoxBlur(heightValues, w, h, radius);

                for (int i = 0; i < heightValues.Length; i++)
                {
                    float detail = heightValues[i] - blurred[i];
                    // Center the details around neutral gray 0.5
                    float flatVal = Mathf.Clamp01(detail + 0.5f);
                    heightValues[i] = Mathf.Lerp(heightValues[i], flatVal, flattenAmount);
                }
            }

            // 3. Write back to color array
            for (int i = 0; i < pixels.Length; i++)
            {
                byte grayValue = (byte)Mathf.Clamp(heightValues[i] * 255f, 0f, 255f);
                pixels[i] = new Color32(grayValue, grayValue, grayValue, 255);
            }

            // 4. Fix border padding artifacts (2-pixel border clamp)
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

        /// <summary>
        /// Generates a tangent-space normal map from a (grayscale) heightmap using a Sobel gradient.
        /// Output is a standard Unity normal map (Y+ / green-up), linear, mip-mapped.
        /// Feed this the adjusted heightmap so the normals match the exported height.
        /// </summary>
        /// <param name="height">Grayscale heightmap (height read from the red channel).</param>
        /// <param name="strength">Slope multiplier. Higher = more pronounced relief. ~1-20.</param>
        public static Texture2D GenerateNormalMap(Texture2D height, float strength)
        {
            if (height == null) return null;

            int w = height.width;
            int h = height.height;

            Color32[] src = height.GetPixels32();
            float[] hv = new float[w * h];
            for (int i = 0; i < src.Length; i++)
            {
                hv[i] = src[i].r / 255f;
            }

            // Linear, mip-mapped RGB so MicroSplat (or any importer set to NormalMap) reads it cleanly.
            Texture2D normal = new Texture2D(w, h, TextureFormat.RGB24, true, true);
            Color32[] outPixels = new Color32[w * h];

            for (int y = 0; y < h; y++)
            {
                int ym = Mathf.Max(y - 1, 0);
                int yp = Mathf.Min(y + 1, h - 1);

                for (int x = 0; x < w; x++)
                {
                    int xm = Mathf.Max(x - 1, 0);
                    int xp = Mathf.Min(x + 1, w - 1);

                    // 3x3 Sobel gradient for smoother normals than a simple central difference.
                    float tl = hv[yp * w + xm], t = hv[yp * w + x], tr = hv[yp * w + xp];
                    float l  = hv[y  * w + xm],                      r  = hv[y  * w + xp];
                    float bl = hv[ym * w + xm], b = hv[ym * w + x], br = hv[ym * w + xp];

                    float dx = (tr + 2f * r + br) - (tl + 2f * l + bl);
                    float dy = (tl + 2f * t + tr) - (bl + 2f * b + br);

                    // Y+ (green-up) tangent-space normal. z fixed; strength tilts x/y.
                    Vector3 n = new Vector3(-dx * strength, -dy * strength, 1f).normalized;

                    outPixels[y * w + x] = new Color32(
                        (byte)Mathf.Clamp((n.x * 0.5f + 0.5f) * 255f, 0f, 255f),
                        (byte)Mathf.Clamp((n.y * 0.5f + 0.5f) * 255f, 0f, 255f),
                        (byte)Mathf.Clamp((n.z * 0.5f + 0.5f) * 255f, 0f, 255f),
                        255);
                }
            }

            normal.SetPixels32(outPixels);
            normal.Apply();
            return normal;
        }

        /// <summary>
        /// Generates a horizon-based ambient-occlusion map from a (grayscale) heightmap. For each
        /// pixel it marches a set of directions, tracks the steepest horizon, and darkens crevices
        /// accordingly. Output is linear, mip-mapped grayscale. Feed it the adjusted heightmap so the
        /// AO matches the exported height.
        /// </summary>
        /// <param name="height">Grayscale heightmap (height read from the red channel).</param>
        /// <param name="strength">Occlusion strength / relief scale. ~0-3.</param>
        /// <param name="radius">Sampling radius as a fraction of the largest image dimension. ~0.005-0.1.</param>
        public static Texture2D GenerateAO(Texture2D height, float strength, float radius)
        {
            if (height == null) return null;

            int w = height.width;
            int h = height.height;

            Color32[] src = height.GetPixels32();
            float[] hv = new float[w * h];
            for (int i = 0; i < src.Length; i++)
            {
                hv[i] = src[i].r / 255f;
            }

            Texture2D ao = new Texture2D(w, h, TextureFormat.RGB24, true, true); // linear, mip-mapped
            Color32[] outPixels = new Color32[w * h];

            const int dirs = 8;
            const int steps = 6;
            int maxDim = Mathf.Max(w, h);
            float maxDist = Mathf.Max(2f, radius * maxDim);
            // Scale height (0..1) into pixel units so slope is comparable to horizontal distance.
            float heightScale = strength * maxDist;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float hP = hv[y * w + x];
                    float occ = 0f;

                    for (int d = 0; d < dirs; d++)
                    {
                        float ang = (Mathf.PI * 2f) * d / dirs;
                        float dx = Mathf.Cos(ang);
                        float dy = Mathf.Sin(ang);

                        float maxSlope = 0f;
                        for (int s = 1; s <= steps; s++)
                        {
                            float dist = maxDist * s / steps;
                            float hs = SampleBilinearClamped(hv, w, h, x + dx * dist, y + dy * dist);
                            float slope = (hs - hP) * heightScale / dist;
                            if (slope > maxSlope) maxSlope = slope;
                        }

                        // Steepest slope -> sine of the horizon angle = directional occlusion.
                        occ += maxSlope / Mathf.Sqrt(maxSlope * maxSlope + 1f);
                    }

                    occ /= dirs;
                    float aoVal = Mathf.Clamp01(1f - occ);
                    byte g = (byte)Mathf.Clamp(aoVal * 255f, 0f, 255f);
                    outPixels[y * w + x] = new Color32(g, g, g, 255);
                }
            }

            ao.SetPixels32(outPixels);
            ao.Apply();
            return ao;
        }

        private static float SampleBilinearClamped(float[] hv, int w, int h, float x, float y)
        {
            x = Mathf.Clamp(x, 0f, w - 1f);
            y = Mathf.Clamp(y, 0f, h - 1f);
            int x0 = Mathf.FloorToInt(x);
            int y0 = Mathf.FloorToInt(y);
            int x1 = Mathf.Min(x0 + 1, w - 1);
            int y1 = Mathf.Min(y0 + 1, h - 1);
            float fx = x - x0;
            float fy = y - y0;

            float v00 = hv[y0 * w + x0];
            float v01 = hv[y0 * w + x1];
            float v10 = hv[y1 * w + x0];
            float v11 = hv[y1 * w + x1];

            return (1f - fy) * ((1f - fx) * v00 + fx * v01) + fy * ((1f - fx) * v10 + fx * v11);
        }

        /// <summary>
        /// Highly performant O(W * H) sliding-window box blur on a 1D float array.
        /// </summary>
        private static float[] BoxBlur(float[] src, int w, int h, int radius)
        {
            float[] dest = new float[w * h];
            float[] temp = new float[w * h];

            // Horizontal Pass
            for (int y = 0; y < h; y++)
            {
                float sum = 0f;
                int denominator = 0;

                for (int x = -radius; x <= radius; x++)
                {
                    int clampedX = Mathf.Clamp(x, 0, w - 1);
                    sum += src[y * w + clampedX];
                    denominator++;
                }
                temp[y * w + 0] = sum / denominator;

                for (int x = 1; x < w; x++)
                {
                    int leftX = Mathf.Clamp(x - radius - 1, 0, w - 1);
                    int rightX = Mathf.Clamp(x + radius, 0, w - 1);
                    sum = sum - src[y * w + leftX] + src[y * w + rightX];
                    temp[y * w + x] = sum / denominator;
                }
            }

            // Vertical Pass
            for (int x = 0; x < w; x++)
            {
                float sum = 0f;
                int denominator = 0;

                for (int y = -radius; y <= radius; y++)
                {
                    int clampedY = Mathf.Clamp(y, 0, h - 1);
                    sum += temp[clampedY * w + x];
                    denominator++;
                }
                dest[0 * w + x] = sum / denominator;

                for (int y = 1; y < h; y++)
                {
                    int topY = Mathf.Clamp(y - radius - 1, 0, h - 1);
                    int bottomY = Mathf.Clamp(y + radius, 0, h - 1);
                    sum = sum - temp[topY * w + x] + temp[bottomY * w + x];
                    dest[y * w + x] = sum / denominator;
                }
            }

            return dest;
        }
    }
}
