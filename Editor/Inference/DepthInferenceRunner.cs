using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using Unity.InferenceEngine;

namespace CPritch.DepthForge.Editor.Inference
{
    public class DepthInferenceRunner : IDisposable
    {
        private Model _runtimeModel;
        private Worker _worker;
        private int _inputRank = 4;
        private BackendType _backend = BackendType.GPUCompute;
        
        public bool IsInitialized => _worker != null;

        public void Initialize(ModelAsset modelAsset, BackendType backend = BackendType.GPUCompute)
        {
            if (modelAsset == null)
            {
                Debug.LogError("ModelAsset is null.");
                return;
            }

            Dispose(); // clean up if already initialized
            
            _runtimeModel = ModelLoader.Load(modelAsset);
            _backend = backend;
            _worker = new Worker(_runtimeModel, _backend);

            if (_runtimeModel.inputs != null && _runtimeModel.inputs.Count > 0)
            {
                _inputRank = _runtimeModel.inputs[0].shape.rank;
                Debug.Log($"DepthInferenceRunner loaded model with input rank: {_inputRank}");
            }
            
            Debug.Log($"DepthInferenceRunner initialized with backend {_backend}");
        }

        private void FallbackToCPU()
        {
            _worker?.Dispose();
            _backend = BackendType.CPU;
            _worker = new Worker(_runtimeModel, _backend);
            Debug.LogWarning("DepthInferenceRunner: GPUCompute failed, successfully fell back to CPU backend.");
        }

        public enum TilingAlignment
        {
            OffsetOnly,
            LinearRegressionFull,
            LinearRegressionDownscaled
        }

        public Texture2D Execute(Texture2D inputTexture, int targetSize = 518, bool useTiling = true, TilingAlignment alignmentStrategy = TilingAlignment.LinearRegressionDownscaled)
        {
            if (!IsInitialized)
            {
                Debug.LogError("DepthInferenceRunner is not initialized. Call Initialize first.");
                return null;
            }

            int W = inputTexture.width;
            int H = inputTexture.height;
            int T = 518; // Standard tile size for Depth Anything V3 ONNX

            // If the texture fits within a single tile, or tiling is disabled, run single-pass inference
            if ((W <= T && H <= T) || !useTiling)
            {
                try
                {
                    using Tensor<float> inputTensor = ImageProcessor.PreprocessTexture(inputTexture, T, T, _inputRank);
                    _worker.Schedule(inputTensor);
                    Tensor<float> outputTensor = _worker.PeekOutput() as Tensor<float>;
                    return ImageProcessor.PostprocessTensor(outputTensor, W, H);
                }
                catch (Exception) when (_backend == BackendType.GPUCompute)
                {
                    FallbackToCPU();
                    return Execute(inputTexture, targetSize, useTiling, alignmentStrategy);
                }
            }

            // Otherwise, run tiled inference to preserve high-resolution crack and ridge details
            int stride = 388; // 25% overlap (518 * 0.25 = 130, 518 - 130 = 388)
            
            // Run global low-resolution inference first to act as a guide for tiling alignment
            float[] globalDepthOriginal = null;
            int drawW = T, drawH = T, drawX = 0, drawY = 0;
            int maxDim = Mathf.Max(W, H);

            Texture2D globalTex = new Texture2D(T, T, TextureFormat.RGB24, false, true);
            Color[] globalPixels = new Color[T * T];
            for (int i = 0; i < T * T; i++) globalPixels[i] = Color.black;

            float scaleWH = (float)T / maxDim;
            drawW = Mathf.RoundToInt(W * scaleWH);
            drawH = Mathf.RoundToInt(H * scaleWH);
            drawX = (T - drawW) / 2;
            drawY = (T - drawH) / 2; // bottom-up Y

            Texture2D inputTex2D = inputTexture as Texture2D;
            if (inputTex2D != null)
            {
                for (int y = 0; y < drawH; y++)
                {
                    float v = (float)y / (drawH > 1 ? drawH - 1 : 1f);
                    for (int x = 0; x < drawW; x++)
                    {
                        float u = (float)x / (drawW > 1 ? drawW - 1 : 1f);
                        globalPixels[(drawY + y) * T + (drawX + x)] = inputTex2D.GetPixelBilinear(u, v);
                    }
                }
            }
            globalTex.SetPixels(globalPixels);
            globalTex.Apply();

            try
            {
                using Tensor<float> globalInput = ImageProcessor.PreprocessTexture(globalTex, T, T, _inputRank);
                _worker.Schedule(globalInput);
                Tensor<float> globalOutput = _worker.PeekOutput() as Tensor<float>;
                float[] paddedGlobalDepth = globalOutput.DownloadToArray();

                globalDepthOriginal = new float[drawW * drawH];
                int tensorTopY = T - drawY - drawH;

                for (int y = 0; y < drawH; y++)
                {
                    for (int x = 0; x < drawW; x++)
                    {
                        int tensorY = tensorTopY + y;
                        int tensorX = drawX + x;
                        globalDepthOriginal[y * drawW + x] = paddedGlobalDepth[tensorY * T + tensorX];
                    }
                }
            }
            catch (Exception) when (_backend == BackendType.GPUCompute)
            {
                UnityEngine.Object.DestroyImmediate(globalTex);
                FallbackToCPU();
                return Execute(inputTexture, targetSize, useTiling, alignmentStrategy);
            }

            UnityEngine.Object.DestroyImmediate(globalTex);

            // Bilinearly upscale the global low-resolution guide to original resolution (W x H)
            // and flip its Y axis to match texture coordinate space (bottom-up)
            float[] globalDepth = new float[W * H];
            float divH = H > 1 ? (float)(H - 1) : 1f;
            float divW = W > 1 ? (float)(W - 1) : 1f;
            for (int y = 0; y < H; y++)
            {
                float gy = (float)y / divH * (drawH - 1);
                int y0 = Mathf.FloorToInt(gy);
                int y1 = Mathf.Min(y0 + 1, drawH - 1);
                float dy = gy - y0;

                int ty0 = drawH - 1 - y0;
                int ty1 = drawH - 1 - y1;

                for (int x = 0; x < W; x++)
                {
                    float gx = (float)x / divW * (drawW - 1);
                    int x0 = Mathf.FloorToInt(gx);
                    int x1 = Mathf.Min(x0 + 1, drawW - 1);
                    float dx = gx - x0;

                    float v00 = globalDepthOriginal[ty0 * drawW + x0];
                    float v01 = globalDepthOriginal[ty0 * drawW + x1];
                    float v10 = globalDepthOriginal[ty1 * drawW + x0];
                    float v11 = globalDepthOriginal[ty1 * drawW + x1];

                    float val = (1f - dy) * ((1f - dx) * v00 + dx * v01) + dy * ((1f - dx) * v10 + dx * v11);
                    globalDepth[y * W + x] = val;
                }
            }

            List<int> xCoords = new List<int>();
            if (W <= T)
            {
                xCoords.Add(0);
            }
            else
            {
                int cx = 0;
                while (cx + T < W)
                {
                    xCoords.Add(cx);
                    cx += stride;
                }
                if (xCoords[xCoords.Count - 1] != W - T)
                {
                    xCoords.Add(W - T);
                }
            }

            List<int> yCoords = new List<int>();
            if (H <= T)
            {
                yCoords.Add(0);
            }
            else
            {
                int cy = 0;
                while (cy + T < H)
                {
                    yCoords.Add(cy);
                    cy += stride;
                }
                if (yCoords[yCoords.Count - 1] != H - T)
                {
                    yCoords.Add(H - T);
                }
            }

            float[] accumulatedDepth = new float[W * H];
            float[] accumulatedWeight = new float[W * H];

            // Precompute 1D smoothstep weight masks focusing fade only on the overlapping regions
            float[] tileWeightMaskX = new float[T];
            float[] tileWeightMaskY = new float[T];
            int overlapX = T - stride;
            int overlapY = T - stride;

            for (int i = 0; i < T; i++)
            {
                float nx = 1.0f;
                if (i < overlapX) nx = (float)i / overlapX;
                else if (i >= T - overlapX) nx = (float)(T - 1 - i) / overlapX;
                tileWeightMaskX[i] = Mathf.SmoothStep(0f, 1f, nx);

                float ny = 1.0f;
                if (i < overlapY) ny = (float)i / overlapY;
                else if (i >= T - overlapY) ny = (float)(T - 1 - i) / overlapY;
                tileWeightMaskY[i] = Mathf.SmoothStep(0f, 1f, ny);
            }

            Texture2D tileTex = new Texture2D(T, T, TextureFormat.RGB24, false, true);

            try
            {
                foreach (int ty in yCoords)
                {
                    foreach (int tx in xCoords)
                    {
                        // Crop the tile on the CPU (immune to platform-specific UV Y-flips in Graphics API)
                        CropTextureCPU(inputTexture, tileTex, tx, ty, T, T);

                        // Preprocess tile
                        using Tensor<float> inputTensor = ImageProcessor.PreprocessTexture(tileTex, T, T, _inputRank);

                        // Execute model on tile
                        _worker.Schedule(inputTensor);

                        // Get output
                        Tensor<float> outputTensor = _worker.PeekOutput() as Tensor<float>;

                        // Download tile data to CPU
                        float[] tileData = outputTensor.DownloadToArray();

                        // Extract guide crop from upscaled global reference, matching the region bottom-up
                        float[] guideCrop = new float[T * T];
                        for (int y = 0; y < T; y++)
                        {
                            int globalY = H - T - ty + y;
                            for (int x = 0; x < T; x++)
                            {
                                int fullX = tx + x;
                                guideCrop[y * T + x] = globalDepth[globalY * W + fullX];
                            }
                        }

                        // Align tile to global guide using selected strategy
                        float scale = 1.0f;
                        float offset = 0.0f;

                        if (alignmentStrategy == TilingAlignment.OffsetOnly)
                        {
                            double sumX = 0;
                            double sumY = 0;
                            int pixelCount = T * T;

                            for (int i = 0; i < pixelCount; i++)
                            {
                                int tileIdx = (T - 1 - i / T) * T + (i % T);
                                sumX += tileData[tileIdx];
                                sumY += guideCrop[i];
                            }
                            offset = (float)((sumY - sumX) / pixelCount);
                        }
                        else if (alignmentStrategy == TilingAlignment.LinearRegressionFull)
                        {
                            float[] alignedTile = GetAlignedTileArray(tileData, T);
                            ComputeLinearRegression(alignedTile, guideCrop, out scale, out offset);
                        }
                        else if (alignmentStrategy == TilingAlignment.LinearRegressionDownscaled)
                        {
                            float[] alignedTile = GetAlignedTileArray(tileData, T);
                            int dsSize = 64;
                            float[] dsTile = Downscale(alignedTile, T, T, dsSize, dsSize);
                            float[] dsGuide = Downscale(guideCrop, T, T, dsSize, dsSize);
                            ComputeLinearRegression(dsTile, dsGuide, out scale, out offset);
                        }

                        // Accumulate depth and weight
                        for (int y = 0; y < T; y++)
                        {
                            int globalY = H - T - ty + y;
                            if (globalY >= H) continue;

                            float wy = tileWeightMaskY[y];
                            if (ty == 0 && y >= T - overlapY) wy = 1.0f;
                            if (ty + T >= H && y < overlapY) wy = 1.0f;

                            for (int x = 0; x < T; x++)
                            {
                                int fullX = tx + x;
                                if (fullX >= W) continue;

                                float wx = tileWeightMaskX[x];
                                if (tx == 0 && x < overlapX) wx = 1.0f;
                                if (tx + T >= W && x >= T - overlapX) wx = 1.0f;

                                int tileIdx = (T - 1 - y) * T + x; // Flip vertically to match bottom-up
                                float w = wx * wy;

                                // Prevent dividing by zero or getting NaN weight
                                if (w < 1e-5f) w = 1e-5f;

                                // Apply scale + offset alignment to tile pixel
                                float alignedVal = tileData[tileIdx] * scale + offset;

                                int fullIdx = globalY * W + fullX;
                                accumulatedDepth[fullIdx] += alignedVal * w;
                                accumulatedWeight[fullIdx] += w;
                            }
                        }
                    }
                }
            }
            catch (Exception) when (_backend == BackendType.GPUCompute)
            {
                UnityEngine.Object.DestroyImmediate(tileTex);
                FallbackToCPU();
                return Execute(inputTexture, targetSize, useTiling, alignmentStrategy);
            }

            UnityEngine.Object.DestroyImmediate(tileTex);

            // Reconstruct the final depth texture
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;
            
            float[] finalData = new float[W * H];
            for (int i = 0; i < W * H; i++)
            {
                float w = accumulatedWeight[i];
                if (w < 1e-5f) w = 1e-5f;
                
                float v = accumulatedDepth[i] / w;
                finalData[i] = v;
                
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }

            float range = maxVal - minVal;
            if (range < 1e-5f) range = 1e-5f;

            Texture2D outTex = new Texture2D(W, H, TextureFormat.RGB24, false, true);
            Color32[] colors = new Color32[W * H];

            for (int y = 0; y < H; y++)
            {
                int tensorIdx = y * W;
                int texIdx = y * W;
                
                for (int x = 0; x < W; x++)
                {
                    float rawVal = finalData[tensorIdx + x];
                    float normVal = (rawVal - minVal) / range;
                    
                    byte grayValue = (byte)Mathf.Clamp(normVal * 255f, 0f, 255f);
                    
                    colors[texIdx + x] = new Color32(grayValue, grayValue, grayValue, 255);
                }
            }

            outTex.SetPixels32(colors);
            outTex.Apply();

            return outTex;
        }

        private static float[] GetAlignedTileArray(float[] tileData, int T)
        {
            float[] aligned = new float[T * T];
            for (int y = 0; y < T; y++)
            {
                int tileIdx = (T - 1 - y) * T;
                int dstIdx = y * T;
                Array.Copy(tileData, tileIdx, aligned, dstIdx, T);
            }
            return aligned;
        }

        private static float[] Downscale(float[] src, int srcW, int srcH, int dstW, int dstH)
        {
            float[] dst = new float[dstW * dstH];
            float scaleX = (float)srcW / dstW;
            float scaleY = (float)srcH / dstH;
            for (int dy = 0; dy < dstH; dy++)
            {
                int sy0 = Mathf.FloorToInt(dy * scaleY);
                int sy1 = Mathf.Min(Mathf.FloorToInt((dy + 1) * scaleY), srcH);
                for (int dx = 0; dx < dstW; dx++)
                {
                    int sx0 = Mathf.FloorToInt(dx * scaleX);
                    int sx1 = Mathf.Min(Mathf.FloorToInt((dx + 1) * scaleX), srcW);
                    double sum = 0;
                    int count = 0;
                    for (int y = sy0; y < sy1; y++)
                    {
                        for (int x = sx0; x < sx1; x++)
                        {
                            sum += src[y * srcW + x];
                            count++;
                        }
                    }
                    dst[dy * dstW + dx] = count > 0 ? (float)(sum / count) : 0f;
                }
            }
            return dst;
        }

        private static void ComputeLinearRegression(float[] xVals, float[] yVals, out float scale, out float offset)
        {
            int N = xVals.Length;
            double sumX = 0;
            double sumY = 0;

            for (int i = 0; i < N; i++)
            {
                sumX += xVals[i];
                sumY += yVals[i];
            }

            double meanX = sumX / N;
            double meanY = sumY / N;

            double varX = 0;
            double varY = 0;
            double covXY = 0;

            for (int i = 0; i < N; i++)
            {
                double dx = xVals[i] - meanX;
                double dy = yVals[i] - meanY;
                varX += dx * dx;
                varY += dy * dy;
                covXY += dx * dy;
            }

            varX /= N;
            varY /= N;
            covXY /= N;

            if (varX < 1e-6)
            {
                scale = 1.0f;
            }
            else
            {
                // Standard Deviation Matching is significantly more stable for tiled depth alignment
                // than OLS slope, which can collapse to 0 or explode if noise decorrelates the signals.
                scale = (float)Math.Sqrt(varY / varX);
                
                // If they are negatively correlated, the depth map is inverted (should never happen),
                // but just in case, fall back to scale 1.0.
                if (covXY < 0) scale = 1.0f;
                
                // Tighter clamp to prevent extreme contrast changes in tiles
                scale = Mathf.Clamp(scale, 0.5f, 2.0f);
            }
            
            offset = (float)(meanY - scale * meanX);
        }

        private void CropTextureCPU(Texture source, Texture2D dest, int tx, int ty, int width, int height)
        {
            Texture2D srcTex = source as Texture2D;
            if (srcTex == null) return;
            
            // Unity Texture2D origin (0,0) is at the bottom-left.
            // ty is the offset from the TOP of the image (0 = top tile).
            // We need to read from the bottom-left of the tile region.
            int bottomY = source.height - ty - height;
            
            Color[] pixels = srcTex.GetPixels(tx, bottomY, width, height);
            dest.SetPixels(pixels);
            dest.Apply();
        }

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
        }
    }
}
