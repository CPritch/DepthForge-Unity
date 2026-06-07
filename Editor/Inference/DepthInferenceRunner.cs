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

        public Texture2D Execute(Texture2D inputTexture, int targetSize = 518)
        {
            if (!IsInitialized)
            {
                Debug.LogError("DepthInferenceRunner is not initialized. Call Initialize first.");
                return null;
            }

            int W = inputTexture.width;
            int H = inputTexture.height;
            int T = 518; // Standard tile size for Depth Anything V3 ONNX

            // If the texture fits within a single tile, run single-pass inference
            if (W <= T && H <= T)
            {
                try
                {
                    using Tensor<float> inputTensor = ImageProcessor.PreprocessTexture(inputTexture, T, T, _inputRank);
                    _worker.Schedule(inputTensor);
                    Tensor<float> outputTensor = _worker.PeekOutput() as Tensor<float>;
                    return ImageProcessor.PostprocessTensor(outputTensor, W, H);
                }
                catch (Exception ex) when (_backend == BackendType.GPUCompute)
                {
                    FallbackToCPU();
                    return Execute(inputTexture, targetSize);
                }
            }

            // Otherwise, run tiled inference to preserve high-resolution crack and ridge details
            int stride = 388; // 25% overlap (518 * 0.25 = 130, 518 - 130 = 388)
            
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

            // Precompute 2D smoothstep weight mask
            float[] tileWeightMask = new float[T * T];
            for (int y = 0; y < T; y++)
            {
                float ny = Mathf.Min(y, T - 1 - y) / (T * 0.5f);
                ny = Mathf.Clamp01(ny);
                float wy = Mathf.SmoothStep(0f, 1f, ny);
                
                for (int x = 0; x < T; x++)
                {
                    float nx = Mathf.Min(x, T - 1 - x) / (T * 0.5f);
                    nx = Mathf.Clamp01(nx);
                    float wx = Mathf.SmoothStep(0f, 1f, nx);
                    
                    tileWeightMask[y * T + x] = wx * wy;
                }
            }

            RenderTexture tileRT = RenderTexture.GetTemporary(T, T, 0, RenderTextureFormat.ARGB32);

            try
            {
                foreach (int ty in yCoords)
                {
                    foreach (int tx in xCoords)
                    {
                        // Crop the tile on the GPU (immune to platform-specific UV Y-flips)
                        CropTextureGPU(inputTexture, tileRT, tx, ty, T, T);

                        // Preprocess tile
                        using Tensor<float> inputTensor = ImageProcessor.PreprocessTexture(tileRT, T, T, _inputRank);

                        // Execute model on tile
                        _worker.Schedule(inputTensor);

                        // Get output
                        Tensor<float> outputTensor = _worker.PeekOutput() as Tensor<float>;

                        // Download tile data to CPU
                        float[] tileData = outputTensor.DownloadToArray();

                        // Accumulate depth and weight
                        for (int y = 0; y < T; y++)
                        {
                            int fullY = ty + y;
                            if (fullY >= H) continue;

                            for (int x = 0; x < T; x++)
                            {
                                int fullX = tx + x;
                                if (fullX >= W) continue;

                                // Flip the Y axis of tileData because RenderTexture input to ToTensor
                                // is vertically flipped on GPU graphics APIs (Vulkan, Metal, D3D).
                                int tileIdx = (T - 1 - y) * T + x;
                                float w = tileWeightMask[y * T + x];

                                int fullIdx = fullY * W + fullX;
                                accumulatedDepth[fullIdx] += tileData[tileIdx] * w;
                                accumulatedWeight[fullIdx] += w;
                            }
                        }
                    }
                }
            }
            catch (Exception ex) when (_backend == BackendType.GPUCompute)
            {
                RenderTexture.ReleaseTemporary(tileRT);
                FallbackToCPU();
                return Execute(inputTexture, targetSize);
            }

            RenderTexture.ReleaseTemporary(tileRT);

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
                int tensorY = H - 1 - y;
                
                for (int x = 0; x < W; x++)
                {
                    int tensorIdx = tensorY * W + x;
                    float rawVal = finalData[tensorIdx];
                    float normVal = (rawVal - minVal) / range;
                    
                    byte grayValue = (byte)Mathf.Clamp(normVal * 255f, 0f, 255f);
                    
                    int texIdx = y * W + x;
                    colors[texIdx] = new Color32(grayValue, grayValue, grayValue, 255);
                }
            }

            outTex.SetPixels32(colors);
            outTex.Apply();

            return outTex;
        }

        private void CropTextureGPU(Texture source, RenderTexture dest, int x, int y, int width, int height)
        {
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = dest;
            
            GL.PushMatrix();
            GL.LoadPixelMatrix(0, dest.width, 0, dest.height);
            
            float srcX = (float)x / source.width;
            float srcY = (float)(source.height - y - height) / source.height;
            float srcW = (float)width / source.width;
            float srcH = (float)height / source.height;
            
            Rect sourceRect = new Rect(srcX, srcY, srcW, srcH);
            Graphics.DrawTexture(new Rect(0, 0, dest.width, dest.height), source, sourceRect, 0, 0, 0, 0);
            
            GL.PopMatrix();
            RenderTexture.active = previousActive;
        }

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
        }
    }
}
