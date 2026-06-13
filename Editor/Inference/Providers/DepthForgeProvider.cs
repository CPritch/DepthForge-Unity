using System;
using System.IO;
using UnityEngine;
using UnityEditor;
using Unity.InferenceEngine;
using CPritch.DepthForge.Editor.Data;

namespace CPritch.DepthForge.Editor.Inference.Providers
{
    /// <summary>
    /// The shippable, commercial-clean material model (DINOv2 Apache-2.0 backbone + DPT decoder,
    /// trained on CC0 MatSynth). It emits <b>normal + roughness natively</b> and a <b>height map</b>
    /// integrated from the normal (Frankot-Chellappa, baked into the ONNX export — so the host reads
    /// height directly and only derives AO).
    ///
    /// STATUS: the weights are still in training (see ~/Dev/DepthForge-Models), so this provider is
    /// <see cref="IsAvailable"/> = false until a model is dropped at <see cref="ModelPath"/>. The
    /// inference body is written against the locked I/O contract below so it becomes a drop-in the day
    /// weights land — but it is UNVERIFIED until the first real export confirms output tensor names.
    ///
    /// Locked ONNX contract (decided 2026-06-13):
    ///   input  : [1, 3, H, W], RGB in [0,1], H and W multiples of 14 (DINOv2 patch).
    ///            ImageNet normalisation is INSIDE the model — do NOT pre-normalise here.
    ///   outputs: normal    [1, 3, H, W]  OpenGL tangent space, [-1,1], L2-normalised
    ///            roughness [1, 1, H, W]  [0,1]
    ///            height    [1, 1, H, W]  [0,1] (integrated, tileable by construction)
    /// </summary>
    public class DepthForgeProvider : IMapProvider
    {
        // Vendored/downloaded model location. Empty today; populated when V1 weights ship.
        private const string ModelPath = "Assets/DepthForge/Models/DepthForge/model.onnx";

        private const int Patch = 14;
        private const int MaxSide = 1036; // 74 * 14 — keep editor inference tractable

        private Model _model;
        private Worker _worker;

        public ProviderInfo Info { get; } = new ProviderInfo
        {
            id = "depthforge-material",
            displayName = "DepthForge (Material)",
            commercialClean = true,
            referenceOnly = false,
            licenseNote = "DINOv2 (Apache-2.0) + CC0 MatSynth. Commercial-clean."
        };

        public MapKinds NativeMaps => MapKinds.Height | MapKinds.Normal | MapKinds.Roughness;
        public bool IsBusy { get; private set; }

        public bool IsAvailable(out string reason)
        {
            if (File.Exists(ModelPath)) { reason = null; return true; }
            reason = "The DepthForge material model is still in training and isn't bundled yet. " +
                     "Use the Depth Anything reference provider for now.";
            return false;
        }

        public void PrepareAsync(Recipe recipe, Action<float, string> onProgress, Action onReady, Action<string> onError)
        {
            // No download wiring yet (model not released). When it ships: vendor the small model in the
            // package and optionally pull a heavier variant from HuggingFace here (mirror DA's two-stage
            // download). Until then, available iff the file already exists locally.
            if (IsAvailable(out string reason)) onReady?.Invoke();
            else onError?.Invoke(reason);
        }

        public void Initialize(Recipe recipe)
        {
            if (!IsAvailable(out string reason)) throw new InvalidOperationException(reason);

            ModelAsset asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
            if (asset == null)
            {
                AssetDatabase.ImportAsset(ModelPath);
                asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
            }
            if (asset == null) throw new InvalidOperationException($"DepthForge model not found at {ModelPath}.");

            _worker?.Dispose();
            _model = ModelLoader.Load(asset);
            BackendType backend = recipe.backend == InferenceBackendChoice.CPUBurst ? BackendType.CPU : BackendType.GPUCompute;
            _worker = new Worker(_model, backend);
        }

        public void ExecuteAsync(Texture2D input, Recipe recipe,
            Action<float> onProgress, Action<MapSet> onDone, Action<string> onError)
        {
            if (_worker == null) { onError?.Invoke("DepthForge provider not initialized."); return; }
            if (IsBusy) { onError?.Invoke("An inference run is already in progress."); return; }

            // TODO(model-launch): make this cooperative (ScheduleIterable + async readback) like
            // DepthInferenceRunner once timings on the real model are known. Synchronous is fine while
            // this path is unreachable (IsAvailable == false).
            IsBusy = true;
            Tensor<float> inputTensor = null;
            try
            {
                onProgress?.Invoke(0.1f);

                int w = RoundToPatch(input.width);
                int h = RoundToPatch(input.height);
                inputTensor = new Tensor<float>(new TensorShape(1, 3, h, w));
                TextureConverter.ToTensor(input, inputTensor, default); // [0,1]; model normalises internally

                _worker.Schedule(inputTensor);
                onProgress?.Invoke(0.6f);

                // Output names are the contract's; PeekOutput by name keeps us order-independent.
                var normalT = _worker.PeekOutput("normal") as Tensor<float>;
                var roughT = _worker.PeekOutput("roughness") as Tensor<float>;
                var heightT = _worker.PeekOutput("height") as Tensor<float>;

                var maps = new MapSet
                {
                    height = heightT != null ? ImageProcessor.PostprocessTensor(heightT, w, h) : null,
                    normal = normalT != null ? NormalTensorToTexture(normalT, w, h) : null,
                    roughness = roughT != null ? ScalarTensorToTexture(roughT, w, h) : null,
                    native = MapKinds.Height | MapKinds.Normal | MapKinds.Roughness
                };

                onProgress?.Invoke(1f);
                onDone?.Invoke(maps);
            }
            catch (Exception ex)
            {
                onError?.Invoke($"DepthForge inference failed: {ex.Message}");
            }
            finally
            {
                inputTensor?.Dispose();
                IsBusy = false;
            }
        }

        private static int RoundToPatch(int v)
        {
            int r = Mathf.Max(Patch, Mathf.RoundToInt(v / (float)Patch) * Patch);
            return Mathf.Min(r, MaxSide);
        }

        /// <summary>[1,3,H,W] OpenGL normals in [-1,1] → RGB texture ((n+1)/2), Y-flipped to Unity origin.</summary>
        private static Texture2D NormalTensorToTexture(Tensor<float> t, int width, int height)
        {
            float[] d = t.DownloadToArray();
            int plane = width * height;
            var tex = new Texture2D(width, height, TextureFormat.RGB24, false, true);
            var cols = new Color32[plane];
            for (int y = 0; y < height; y++)
            {
                int srcY = height - 1 - y;
                for (int x = 0; x < width; x++)
                {
                    int s = srcY * width + x;
                    byte R = ToByte01((d[s] + 1f) * 0.5f);
                    byte G = ToByte01((d[plane + s] + 1f) * 0.5f);
                    byte B = ToByte01((d[2 * plane + s] + 1f) * 0.5f);
                    cols[y * width + x] = new Color32(R, G, B, 255);
                }
            }
            tex.SetPixels32(cols);
            tex.Apply();
            return tex;
        }

        /// <summary>[1,1,H,W] scalar in [0,1] → grayscale texture, Y-flipped. No min/max renorm (already [0,1]).</summary>
        private static Texture2D ScalarTensorToTexture(Tensor<float> t, int width, int height)
        {
            float[] d = t.DownloadToArray();
            var tex = new Texture2D(width, height, TextureFormat.RGB24, false, true);
            var cols = new Color32[width * height];
            for (int y = 0; y < height; y++)
            {
                int srcY = height - 1 - y;
                for (int x = 0; x < width; x++)
                {
                    byte v = ToByte01(d[srcY * width + x]);
                    cols[y * width + x] = new Color32(v, v, v, 255);
                }
            }
            tex.SetPixels32(cols);
            tex.Apply();
            return tex;
        }

        private static byte ToByte01(float v) => (byte)Mathf.Clamp(v * 255f, 0f, 255f);

        public void Dispose()
        {
            _worker?.Dispose();
            _worker = null;
            _model = null;
        }
    }
}
