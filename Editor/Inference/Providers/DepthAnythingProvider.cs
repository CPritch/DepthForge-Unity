#if !DEPTHFORGE_SHIPPING
using System;
using System.IO;
using UnityEngine;
using UnityEditor;
using Unity.InferenceEngine;
using CPritch.DepthForge.Editor.Data;
using CPritch.DepthForge.Editor.Utils;

namespace CPritch.DepthForge.Editor.Inference.Providers
{
    /// <summary>
    /// Depth Anything V3 as a <b>reference-only</b> provider. DA3 is CC-BY-NC (non-commercial — see
    /// the model repo's LICENSING.md), so it can never be the shipped default; it lives behind the
    /// <c>!DEPTHFORGE_SHIPPING</c> compile gate purely so authors can compare against it in-editor.
    ///
    /// It emits a single-channel <b>height</b> map only (<see cref="MapKinds.Height"/>); the host
    /// derives normal + AO from that. This class owns everything DA-specific — model URLs/paths per
    /// size, download, backend selection — wrapping the existing <see cref="DepthInferenceRunner"/>.
    /// </summary>
    public class DepthAnythingProvider : IMapProvider
    {
        private const string CacheDir = "Assets/DepthForge/Models";

        private struct ModelLocation
        {
            public string url, dataUrl, path, dataPath, dir;
        }

        private readonly DepthInferenceRunner _runner = new DepthInferenceRunner();
        private DownloadHandle _download;

        public ProviderInfo Info { get; } = new ProviderInfo
        {
            id = "depth-anything-v3",
            displayName = "Depth Anything V3 — Reference (non-commercial)",
            commercialClean = false,
            referenceOnly = true,
            licenseNote = "CC-BY-NC. In-editor reference comparison only; never shipped or used commercially."
        };

        public MapKinds NativeMaps => MapKinds.Height;
        public bool IsBusy => _runner.IsBusy;

        private static ModelLocation Locate(DepthModelSize size)
        {
            string sub = size.ToString(); // Small / Base / Large
            string slug = size == DepthModelSize.Small ? "small" : size == DepthModelSize.Base ? "base" : "large";
            string baseUrl = $"https://huggingface.co/onnx-community/depth-anything-v3-{slug}/resolve/main/onnx";
            string dir = $"{CacheDir}/{sub}";
            return new ModelLocation
            {
                url = $"{baseUrl}/model.onnx",
                dataUrl = $"{baseUrl}/model.onnx_data",
                path = $"{dir}/model.onnx",
                dataPath = $"{dir}/model.onnx_data",
                dir = dir
            };
        }

        public bool IsAvailable(out string reason)
        {
            var loc = Locate(DepthModelSize.Small); // availability is per-size; default-check Small
            // The window passes the real size into PrepareAsync; this cheap check just reports whether
            // *anything* is downloaded so batch can decide. Re-check the specific size in PrepareAsync.
            bool any = File.Exists(loc.path) || File.Exists(Locate(DepthModelSize.Base).path) || File.Exists(Locate(DepthModelSize.Large).path);
            reason = any ? null : "No Depth Anything model downloaded yet. Run Generate once to fetch it.";
            return any;
        }

        public void PrepareAsync(Recipe recipe, Action<float, string> onProgress, Action onReady, Action<string> onError)
        {
            var loc = Locate(recipe.modelSize);

            if (File.Exists(loc.path) && File.Exists(loc.dataPath))
            {
                onReady?.Invoke();
                return;
            }

            EnsureGitIgnore();
            try { if (!Directory.Exists(loc.dir)) Directory.CreateDirectory(loc.dir); }
            catch (Exception ex) { Debug.LogWarning($"[DepthForge] Failed to create model dir: {ex.Message}"); }

            // Two stages: graph (.onnx, ~5% of the bytes) then weights (.onnx_data, ~95%).
            onProgress?.Invoke(0f, "Downloading depth model (structure)...");
            _download = ModelDownloader.DownloadModel(loc.url, loc.path,
                p => onProgress?.Invoke(p * 0.05f, $"Downloading depth model (structure) ({(int)(p * 100)}%)..."),
                structErr =>
                {
                    // Match on the cancel sentinel rather than the _download flag: Dispose() nulls
                    // _download before this callback fires (e.g. window closed mid-download), so the
                    // flag check would mis-report a cancel as a failure and pop a dialog on a dead window.
                    if (structErr == "Download cancelled.") return;
                    if (!string.IsNullOrEmpty(structErr)) { onError?.Invoke($"Failed to download model structure: {structErr}"); return; }

                    _download = ModelDownloader.DownloadModel(loc.dataUrl, loc.dataPath,
                        p => onProgress?.Invoke(0.05f + p * 0.95f, $"Downloading depth model (weights) ({(int)(p * 100)}%)..."),
                        dataErr =>
                        {
                            _download = null;
                            if (!string.IsNullOrEmpty(dataErr))
                            {
                                if (dataErr != "Download cancelled.") onError?.Invoke($"Failed to download model weights: {dataErr}");
                                return;
                            }

                            AssetDatabase.ImportAsset(loc.path);
                            AssetDatabase.Refresh();
                            if (AssetDatabase.LoadAssetAtPath<ModelAsset>(loc.path) == null)
                            {
                                onError?.Invoke("Failed to import the downloaded depth model into Unity's Asset Database.");
                                return;
                            }
                            onReady?.Invoke();
                        });
                });
        }

        public void Initialize(Recipe recipe)
        {
            var loc = Locate(recipe.modelSize);
            ModelAsset asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(loc.path);
            if (asset == null)
            {
                AssetDatabase.ImportAsset(loc.path);
                asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(loc.path);
            }
            if (asset == null) throw new InvalidOperationException($"Depth Anything model not found at {loc.path}.");

            _runner.Initialize(asset, ResolveBackend(recipe));
        }

        private static BackendType ResolveBackend(Recipe recipe)
        {
            switch (recipe.backend)
            {
                case InferenceBackendChoice.CPUBurst: return BackendType.CPU;
                case InferenceBackendChoice.GPUCompute: return BackendType.GPUCompute;
                default: // Auto — the larger DA models OOM on GPUCompute in-editor, so prefer CPU.
                    return (recipe.modelSize == DepthModelSize.Base || recipe.modelSize == DepthModelSize.Large)
                        ? BackendType.CPU : BackendType.GPUCompute;
            }
        }

        public void ExecuteAsync(Texture2D input, Recipe recipe,
            Action<float> onProgress, Action<MapSet> onDone, Action<string> onError)
        {
            var alignment = (DepthInferenceRunner.TilingAlignment)(int)recipe.tilingAlignment;
            _runner.ExecuteAsync(input, recipe.tiledInference, alignment, recipe.letterbox,
                onProgress,
                height => onDone?.Invoke(new MapSet { height = height, native = MapKinds.Height }),
                onError);
        }

        private static void EnsureGitIgnore()
        {
            try
            {
                if (!Directory.Exists(CacheDir)) Directory.CreateDirectory(CacheDir);
                string gitignorePath = Path.Combine(CacheDir, ".gitignore");
                if (!File.Exists(gitignorePath)) File.WriteAllText(gitignorePath, "*\n!.gitignore\n");
            }
            catch (Exception ex) { Debug.LogWarning($"[DepthForge] Failed to write models .gitignore: {ex.Message}"); }
        }

        public void Dispose()
        {
            _download?.Cancel();
            _download = null;
            _runner?.Dispose();
        }
    }
}
#endif
