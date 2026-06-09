using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Collections;
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

        // --- Async (non-blocking) single-pass state, pumped off EditorApplication.update ---
        private enum AsyncPhase { Idle, Scheduling, Readback }
        private AsyncPhase _asyncPhase = AsyncPhase.Idle;
        private bool _asyncRunning;
        private IEnumerator _asyncSchedule;
        private Tensor<float> _asyncInput;
        private Tensor<float> _asyncOutput;
        private Func<Tensor<float>, Texture2D> _asyncPostprocess;
        private Action<float> _asyncProgress;
        private Action<Texture2D> _asyncComplete;
        private Action<string> _asyncError;
        private int _asyncStepCount;
        private const double AsyncBudgetMs = 6.0;

        public bool IsInitialized => _worker != null;
        public bool IsBusy => _asyncRunning;

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

        public Texture2D Execute(Texture2D inputTexture, int targetSize = 518, bool useTiling = true, TilingAlignment alignmentStrategy = TilingAlignment.LinearRegressionDownscaled, bool letterbox = true)
        {
            if (!IsInitialized)
            {
                Debug.LogError("DepthInferenceRunner is not initialized. Call Initialize first.");
                return null;
            }

            int W = inputTexture.width;
            int H = inputTexture.height;
            int T = 518; // Standard tile size for Depth Anything V3 ONNX

            // Tiling crops fixed T×T tiles, so it needs BOTH dimensions to be at least one full tile.
            // When either side is smaller than a tile (e.g. a 1024x512 texture), a T×T crop runs off
            // the image edge — CropTextureCPU's bottomY goes negative and the guide indexing goes
            // out of bounds — which produced the "breaks on some aspect ratios" failures. Fall back
            // to single-pass inference in that case (and when tiling is disabled / both sides fit one tile).
            if (!useTiling || W < T || H < T)
            {
                try
                {
                    // Letterbox: pad the (non-square) image to a square tile preserving aspect, so
                    // the model sees even scaling instead of a squashed image. Square images need no
                    // padding and use the direct resize path.
                    if (letterbox && W != H)
                    {
                        return ExecuteLetterboxed(inputTexture, W, H, T);
                    }

                    using Tensor<float> inputTensor = ImageProcessor.PreprocessTexture(inputTexture, T, T, _inputRank);
                    _worker.Schedule(inputTensor);
                    Tensor<float> outputTensor = _worker.PeekOutput() as Tensor<float>;
                    return ImageProcessor.PostprocessTensor(outputTensor, W, H);
                }
                catch (Exception) when (_backend == BackendType.GPUCompute)
                {
                    FallbackToCPU();
                    return Execute(inputTexture, targetSize, useTiling, alignmentStrategy, letterbox);
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
                return Execute(inputTexture, targetSize, useTiling, alignmentStrategy, letterbox);
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
                return Execute(inputTexture, targetSize, useTiling, alignmentStrategy, letterbox);
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

        /// <summary>
        /// Single-pass inference that preserves aspect ratio: the image is scaled to fit a T×T tile
        /// (aspect-correct), centered, with its edges replicated into the padding (so the model does
        /// not see a hard black border), run once, then the valid region is cropped back out and
        /// resized to the original W×H.
        /// </summary>
        private Texture2D ExecuteLetterboxed(Texture2D inputTexture, int W, int H, int T)
        {
            int maxDim = Mathf.Max(W, H);
            float scale = (float)T / maxDim;
            int drawW = Mathf.Clamp(Mathf.RoundToInt(W * scale), 1, T);
            int drawH = Mathf.Clamp(Mathf.RoundToInt(H * scale), 1, T);
            int drawX = (T - drawW) / 2;
            int drawY = (T - drawH) / 2; // bottom-up

            // Build the letterboxed T×T input (edge-replicated padding).
            Texture2D padded = new Texture2D(T, T, TextureFormat.RGB24, false, true);
            Color[] pix = new Color[T * T];
            for (int y = 0; y < T; y++)
            {
                int yy = Mathf.Clamp(y - drawY, 0, drawH - 1);
                float v = drawH > 1 ? (float)yy / (drawH - 1) : 0f;
                for (int x = 0; x < T; x++)
                {
                    int xx = Mathf.Clamp(x - drawX, 0, drawW - 1);
                    float u = drawW > 1 ? (float)xx / (drawW - 1) : 0f;
                    pix[y * T + x] = inputTexture.GetPixelBilinear(u, v);
                }
            }
            padded.SetPixels(pix);
            padded.Apply();

            float[] outData;
            try
            {
                using Tensor<float> input = ImageProcessor.PreprocessTexture(padded, T, T, _inputRank);
                _worker.Schedule(input);
                Tensor<float> output = _worker.PeekOutput() as Tensor<float>;
                outData = output.DownloadToArray(); // T*T, top-down (NN origin)
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(padded);
            }

            // Extract the valid (aspect-correct) region from the padded output. Tensor is top-down,
            // drawY is a bottom-up offset, so the region's top row in tensor space is T - drawY - drawH.
            int tensorTopY = T - drawY - drawH;
            float[] region = new float[drawW * drawH];
            float minV = float.MaxValue, maxV = float.MinValue;
            for (int y = 0; y < drawH; y++)
            {
                for (int x = 0; x < drawW; x++)
                {
                    float val = outData[(tensorTopY + y) * T + (drawX + x)];
                    region[y * drawW + x] = val;
                    if (float.IsNaN(val) || float.IsInfinity(val)) continue;
                    if (val < minV) minV = val;
                    if (val > maxV) maxV = val;
                }
            }
            float range = maxV - minV;
            if (range < 1e-5f) range = 1e-5f;

            // Resize the region (drawW×drawH, top-down) to W×H, flipping Y to texture bottom-up, and normalize.
            Texture2D outTex = new Texture2D(W, H, TextureFormat.RGB24, false, true);
            Color32[] colors = new Color32[W * H];
            float divW = W > 1 ? W - 1 : 1f;
            float divH = H > 1 ? H - 1 : 1f;
            for (int y = 0; y < H; y++)
            {
                float gy = (1f - y / divH) * (drawH - 1); // y=0 (bottom) -> region bottom row
                int y0 = Mathf.FloorToInt(gy);
                int y1 = Mathf.Min(y0 + 1, drawH - 1);
                float fy = gy - y0;

                for (int x = 0; x < W; x++)
                {
                    float gx = x / divW * (drawW - 1);
                    int x0 = Mathf.FloorToInt(gx);
                    int x1 = Mathf.Min(x0 + 1, drawW - 1);
                    float fx = gx - x0;

                    float v00 = region[y0 * drawW + x0];
                    float v01 = region[y0 * drawW + x1];
                    float v10 = region[y1 * drawW + x0];
                    float v11 = region[y1 * drawW + x1];
                    float val = (1f - fy) * ((1f - fx) * v00 + fx * v01) + fy * ((1f - fx) * v10 + fx * v11);

                    byte g = (byte)Mathf.Clamp((val - minV) / range * 255f, 0f, 255f);
                    colors[y * W + x] = new Color32(g, g, g, 255);
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

        /// <summary>
        /// Non-blocking inference. The default single-pass (square / letterbox-off) path runs
        /// cooperatively off EditorApplication.update so the editor stays responsive; results arrive
        /// via <paramref name="onComplete"/>. The tiled path and letterboxed non-square path are not
        /// cooperative yet, so they run synchronously here and report completion immediately.
        /// </summary>
        public void ExecuteAsync(Texture2D inputTexture, bool useTiling, TilingAlignment alignment, bool letterbox,
                                 Action<float> onProgress, Action<Texture2D> onComplete, Action<string> onError)
        {
            if (!IsInitialized) { onError?.Invoke("DepthInferenceRunner is not initialized."); return; }
            if (_asyncRunning) { onError?.Invoke("An inference run is already in progress."); return; }

            int W = inputTexture.width;
            int H = inputTexture.height;
            int T = 518;

            // Paths that aren't cooperative yet run synchronously and report completion:
            //  - tiled (experimental, off by default)
            //  - letterboxed non-square (CPU-bound pad + resize)
            bool tiled = useTiling && W >= T && H >= T;
            bool letterboxed = letterbox && W != H;
            if (tiled || letterboxed)
            {
                try
                {
                    onProgress?.Invoke(0.5f);
                    onComplete?.Invoke(Execute(inputTexture, T, useTiling, alignment, letterbox));
                }
                catch (Exception ex)
                {
                    onError?.Invoke($"Inference failed: {ex.Message}");
                }
                return;
            }

            // Async single-pass (square, or letterbox off) — the common large-texture case.
            try
            {
                _asyncInput = ImageProcessor.PreprocessTexture(inputTexture, T, T, _inputRank);
                _asyncPostprocess = t => ImageProcessor.PostprocessTensor(t, W, H);
                _asyncProgress = onProgress;
                _asyncComplete = onComplete;
                _asyncError = onError;
                StartAsyncSchedule();
            }
            catch (Exception ex)
            {
                CleanupAsync();
                onError?.Invoke($"Failed to start inference: {ex.Message}");
            }
        }

        private void StartAsyncSchedule()
        {
            _asyncRunning = true;
            _asyncStepCount = 0;
            _asyncSchedule = _worker.ScheduleIterable(_asyncInput);
            _asyncPhase = AsyncPhase.Scheduling;
            EditorApplication.update -= AsyncTick; // avoid double-subscribe
            EditorApplication.update += AsyncTick;
        }

        private void AsyncTick()
        {
            if (_asyncPhase == AsyncPhase.Scheduling)
            {
                var budget = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    while (budget.Elapsed.TotalMilliseconds < AsyncBudgetMs)
                    {
                        if (!_asyncSchedule.MoveNext())
                        {
                            // Compute scheduled — request a non-blocking GPU->CPU readback.
                            _asyncOutput = _worker.PeekOutput() as Tensor<float>;
                            _asyncOutput.ReadbackRequest();
                            _asyncPhase = AsyncPhase.Readback;
                            _asyncProgress?.Invoke(0.95f);
                            return;
                        }
                        _asyncStepCount++;
                    }
                    // ScheduleIterable doesn't expose a layer total, so approach 0.9 asymptotically.
                    _asyncProgress?.Invoke(Mathf.Min(0.9f, _asyncStepCount / 300f));
                }
                catch (Exception ex)
                {
                    FinishAsyncError($"Inference failed: {ex.Message}");
                }
            }
            else if (_asyncPhase == AsyncPhase.Readback)
            {
                try
                {
                    if (!_asyncOutput.IsReadbackRequestDone()) return; // still copying; editor stays free
                    Texture2D result = _asyncPostprocess(_asyncOutput);
                    FinishAsyncSuccess(result);
                }
                catch (Exception ex)
                {
                    FinishAsyncError($"Readback failed: {ex.Message}");
                }
            }
        }

        private void FinishAsyncSuccess(Texture2D result)
        {
            var progress = _asyncProgress;
            var complete = _asyncComplete;
            CleanupAsync();
            progress?.Invoke(1f);
            complete?.Invoke(result);
        }

        private void FinishAsyncError(string message)
        {
            var error = _asyncError;
            CleanupAsync();
            error?.Invoke(message);
        }

        private void CleanupAsync()
        {
            EditorApplication.update -= AsyncTick;
            _asyncSchedule = null;
            _asyncOutput = null; // owned by the worker
            _asyncInput?.Dispose();
            _asyncInput = null;
            _asyncPostprocess = null;
            _asyncProgress = null;
            _asyncComplete = null;
            _asyncError = null;
            _asyncPhase = AsyncPhase.Idle;
            _asyncRunning = false;
        }

        public void Dispose()
        {
            CleanupAsync();
            _worker?.Dispose();
            _worker = null;
        }
    }
}
