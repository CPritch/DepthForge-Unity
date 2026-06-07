using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEditor.UIElements;
using Unity.InferenceEngine;
using CPritch.DepthForge.Editor.Inference;
using CPritch.DepthForge.Editor.Utils;

namespace CPritch.DepthForge.Editor
{
    public class DepthForgeWindow : EditorWindow
    {
        [MenuItem("Tools/DepthForge/Heightmap Generator")]
        public static void ShowExample()
        {
            DepthForgeWindow wnd = GetWindow<DepthForgeWindow>();
            wnd.titleContent = new GUIContent("DepthForge");
            wnd.minSize = new Vector2(450, 500);
        }

        public void CreateGUI()
        {
            // Each editor window contains a root VisualElement object
            VisualElement root = rootVisualElement;

            // Load UXML
            // Note: Since this is a UPM package, the path assumes the package is installed
            // or the files are accessible via the AssetDatabase.
            // Using a GUID is safer, but for development, AssetDatabase.LoadAssetAtPath might be used.
            // We will load by finding the GUID or path in a real package. For now, we try to load from typical package path.
            var visualTree = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>("Packages/com.cpritch.depthforge/Editor/UI/DepthForgeLayout.uxml");
            
            // If we are developing locally outside of the Packages folder (e.g. directly in Assets), fallback
            if (visualTree == null)
            {
                // Fallback for direct project development
                var guids = AssetDatabase.FindAssets("DepthForgeLayout t:VisualTreeAsset");
                if (guids.Length > 0)
                {
                    string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                    visualTree = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(path);
                }
            }

            if (visualTree != null)
            {
                VisualElement labelFromUXML = visualTree.Instantiate();
                labelFromUXML.style.flexGrow = 1;
                root.Add(labelFromUXML);

                // Setup Stylesheet dynamically in case UXML reference fails
                var styleSheet = AssetDatabase.LoadAssetAtPath<StyleSheet>("Packages/com.cpritch.depthforge/Editor/UI/DepthForgeStyle.uss");
                if (styleSheet == null)
                {
                    var guids = AssetDatabase.FindAssets("DepthForgeStyle t:StyleSheet");
                    if (guids.Length > 0)
                    {
                        string path = AssetDatabase.GUIDToAssetPath(guids[0]);
                        styleSheet = AssetDatabase.LoadAssetAtPath<StyleSheet>(path);
                    }
                }
                
                if (styleSheet != null)
                {
                    root.styleSheets.Add(styleSheet);
                }

                BindUI(root);
            }
            else
            {
                root.Add(new Label("Failed to load DepthForgeLayout.uxml. Ensure it is placed correctly in the project/package."));
            }
        }
        public enum ModelSize
        {
            [InspectorName("Performance (Small - 105MB)")]
            Small,
            [InspectorName("Balanced (Base - 240MB)")]
            Base,
            [InspectorName("Production (Large - 670MB)")]
            Large
        }

        public enum InferenceBackend
        {
            [InspectorName("Auto-detect (GPU with CPU Fallback)")]
            Auto,
            [InspectorName("GPU Compute Shader (Fastest)")]
            GPUCompute,
            [InspectorName("CPU Burst (Most Compatible)")]
            CPUBurst
        }

        private const string MODEL_SMALL_URL = "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx";
        private const string MODEL_SMALL_DATA_URL = "https://huggingface.co/onnx-community/depth-anything-v3-small/resolve/main/onnx/model.onnx_data";
        
        private const string MODEL_BASE_URL = "https://huggingface.co/onnx-community/depth-anything-v3-base/resolve/main/onnx/model.onnx";
        private const string MODEL_BASE_DATA_URL = "https://huggingface.co/onnx-community/depth-anything-v3-base/resolve/main/onnx/model.onnx_data";

        private const string MODEL_LARGE_URL = "https://huggingface.co/onnx-community/depth-anything-v3-large/resolve/main/onnx/model.onnx";
        private const string MODEL_LARGE_DATA_URL = "https://huggingface.co/onnx-community/depth-anything-v3-large/resolve/main/onnx/model.onnx_data";

        private const string MODEL_CACHE_DIR = "Assets/DepthForge/Models";
        private const string MODEL_SMALL_DIR = "Assets/DepthForge/Models/Small";
        private const string MODEL_BASE_DIR = "Assets/DepthForge/Models/Base";
        private const string MODEL_LARGE_DIR = "Assets/DepthForge/Models/Large";

        private const string MODEL_SMALL_PATH = "Assets/DepthForge/Models/Small/model.onnx";
        private const string MODEL_SMALL_DATA_PATH = "Assets/DepthForge/Models/Small/model.onnx_data";

        private const string MODEL_BASE_PATH = "Assets/DepthForge/Models/Base/model.onnx";
        private const string MODEL_BASE_DATA_PATH = "Assets/DepthForge/Models/Base/model.onnx_data";

        private const string MODEL_LARGE_PATH = "Assets/DepthForge/Models/Large/model.onnx";
        private const string MODEL_LARGE_DATA_PATH = "Assets/DepthForge/Models/Large/model.onnx_data";

        private DepthInferenceRunner _runner;
        private ObjectField _inputTextureField;
        private EnumField _modelSizeField;
        private EnumField _backendField;
        private EnumField _outputFormatField;
        private VisualElement _outputPreview;
        
        private Button _generateButton;
        private ProgressBar _progressBar;
        private DownloadHandle _downloadHandle;

        private void OnEnable()
        {
            _runner = new DepthInferenceRunner();
        }

        private void OnDisable()
        {
            _runner?.Dispose();
            
            if (_downloadHandle != null)
            {
                _downloadHandle.Cancel();
                _downloadHandle = null;
            }
        }

        private void BindUI(VisualElement root)
        {
            _generateButton = root.Q<Button>("generateButton");
            _inputTextureField = root.Q<ObjectField>("inputTextureField");
            _modelSizeField = root.Q<EnumField>("modelSizeField");
            _backendField = root.Q<EnumField>("backendField");
            _outputFormatField = root.Q<EnumField>("outputFormatField");
            
            var inputPreview = root.Q<VisualElement>("inputPreview");
            _outputPreview = root.Q<VisualElement>("outputPreview");
            _progressBar = root.Q<ProgressBar>("progressBar");

            if (_progressBar != null)
            {
                _progressBar.lowValue = 0f;
                _progressBar.highValue = 1f;
            }

            if (_modelSizeField != null)
            {
                _modelSizeField.Init(ModelSize.Small);
            }

            if (_backendField != null)
            {
                _backendField.Init(InferenceBackend.Auto);
            }


            if (_outputFormatField != null)
            {
                _outputFormatField.Init(CPritch.DepthForge.Editor.Utils.TextureExporter.ExportFormat.PNG_8Bit);
            }

            _inputTextureField.RegisterValueChangedCallback(evt =>
            {
                Texture2D tex = evt.newValue as Texture2D;
                if (tex != null)
                {
                    inputPreview.style.backgroundImage = tex;
                }
                else
                {
                    inputPreview.style.backgroundImage = null;
                }
            });

            _generateButton.clicked += OnGenerateClicked;
        }

        private void OnGenerateClicked()
        {
            Texture2D inputTex = _inputTextureField.value as Texture2D;

            if (inputTex == null)
            {
                EditorUtility.DisplayDialog("Missing references", "Please assign a Base Map texture.", "OK");
                return;
            }

            ModelSize selectedSize = _modelSizeField != null ? (ModelSize)_modelSizeField.value : ModelSize.Small;
            string modelPath = MODEL_SMALL_PATH;
            string modelDataPath = MODEL_SMALL_DATA_PATH;

            if (selectedSize == ModelSize.Base)
            {
                modelPath = MODEL_BASE_PATH;
                modelDataPath = MODEL_BASE_DATA_PATH;
            }
            else if (selectedSize == ModelSize.Large)
            {
                modelPath = MODEL_LARGE_PATH;
                modelDataPath = MODEL_LARGE_DATA_PATH;
            }

            if (!File.Exists(modelPath) || !File.Exists(modelDataPath))
            {
                StartModelDownload(inputTex, selectedSize);
            }
            else
            {
                ModelAsset modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelPath);
                if (modelAsset == null)
                {
                    // Force refresh/import in case Unity hasn't registered the file yet
                    AssetDatabase.ImportAsset(modelPath);
                    AssetDatabase.Refresh();
                    modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelPath);
                }

                if (modelAsset == null)
                {
                    // Fall back to downloading if import is failing or file is corrupted
                    StartModelDownload(inputTex, selectedSize);
                }
                else
                {
                    RunInference(inputTex, modelAsset);
                }
            }
        }

        private void StartModelDownload(Texture2D inputTex, ModelSize size)
        {
            EnsureGitIgnore();

            string modelUrl = MODEL_SMALL_URL;
            string modelDataUrl = MODEL_SMALL_DATA_URL;
            string modelPath = MODEL_SMALL_PATH;
            string modelDataPath = MODEL_SMALL_DATA_PATH;
            string modelDir = MODEL_SMALL_DIR;

            if (size == ModelSize.Base)
            {
                modelUrl = MODEL_BASE_URL;
                modelDataUrl = MODEL_BASE_DATA_URL;
                modelPath = MODEL_BASE_PATH;
                modelDataPath = MODEL_BASE_DATA_PATH;
                modelDir = MODEL_BASE_DIR;
            }
            else if (size == ModelSize.Large)
            {
                modelUrl = MODEL_LARGE_URL;
                modelDataUrl = MODEL_LARGE_DATA_URL;
                modelPath = MODEL_LARGE_PATH;
                modelDataPath = MODEL_LARGE_DATA_PATH;
                modelDir = MODEL_LARGE_DIR;
            }

            try
            {
                if (!Directory.Exists(modelDir))
                {
                    Directory.CreateDirectory(modelDir);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Failed to create models subdirectory: {ex.Message}");
            }

            _generateButton.SetEnabled(false);
            _progressBar.RemoveFromClassList("hidden");
            _progressBar.value = 0f;
            _progressBar.title = "Downloading depth model (structure)...";

            // Step 1: Download model.onnx (very small)
            _downloadHandle = CPritch.DepthForge.Editor.Utils.ModelDownloader.DownloadModel(
                modelUrl,
                modelPath,
                progress =>
                {
                    float currentProgress = progress * 0.05f; // Count as first 5%
                    _progressBar.value = currentProgress;
                    _progressBar.title = $"Downloading depth model (structure) ({(int)(progress * 100)}%)...";
                    _generateButton.text = $"Downloading depth model ({(int)(currentProgress * 100)}%)...";
                },
                error =>
                {
                    if (_downloadHandle != null && _downloadHandle.IsCancelled) return;

                    if (!string.IsNullOrEmpty(error))
                    {
                        HandleDownloadFailure($"Failed to download model structure: {error}");
                        return;
                    }

                    // Step 2: Download model.onnx_data (weights)
                    _progressBar.title = "Downloading depth model (weights)...";
                    _downloadHandle = CPritch.DepthForge.Editor.Utils.ModelDownloader.DownloadModel(
                        modelDataUrl,
                        modelDataPath,
                        progress =>
                        {
                            float currentProgress = 0.05f + progress * 0.95f; // Count as remaining 95%
                            _progressBar.value = currentProgress;
                            _progressBar.title = $"Downloading depth model (weights) ({(int)(progress * 100)}%)...";
                            _generateButton.text = $"Downloading depth model ({(int)(currentProgress * 100)}%)...";
                        },
                        errorData =>
                        {
                            _generateButton.SetEnabled(true);
                            _generateButton.text = "Generate Heightmap";
                            _progressBar.AddToClassList("hidden");
                            _downloadHandle = null;

                            if (!string.IsNullOrEmpty(errorData))
                            {
                                if (errorData != "Download cancelled.")
                                {
                                    EditorUtility.DisplayDialog("Download Failed", $"Failed to download model weights:\n{errorData}", "OK");
                                }
                            }
                            else
                            {
                                // Success! Import asset immediately so Unity compiles it to a ModelAsset
                                AssetDatabase.ImportAsset(modelPath);
                                AssetDatabase.Refresh();

                                ModelAsset modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelPath);
                                if (modelAsset == null)
                                {
                                    EditorUtility.DisplayDialog("Import Failed", "Failed to load imported depth model into Unity's Asset Database.", "OK");
                                    return;
                                }

                                RunInference(inputTex, modelAsset);
                            }
                        }
                    );
                }
            );
        }

        private void EnsureGitIgnore()
        {
            try
            {
                if (!Directory.Exists(MODEL_CACHE_DIR))
                {
                    Directory.CreateDirectory(MODEL_CACHE_DIR);
                }
                string gitignorePath = Path.Combine(MODEL_CACHE_DIR, ".gitignore");
                if (!File.Exists(gitignorePath))
                {
                    File.WriteAllText(gitignorePath, "*\n!.gitignore\n");
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Failed to write .gitignore in models cache: {ex.Message}");
            }
        }

        private void HandleDownloadFailure(string errorMsg)
        {
            _generateButton.SetEnabled(true);
            _generateButton.text = "Generate Heightmap";
            _progressBar.AddToClassList("hidden");
            _downloadHandle = null;
            EditorUtility.DisplayDialog("Download Failed", errorMsg, "OK");
        }

        private void RunInference(Texture2D inputTex, ModelAsset modelAsset)
        {
            try
            {
                EditorUtility.DisplayProgressBar("DepthForge", "Initializing Inference Engine...", 0.2f);
                
                ModelSize selectedSize = _modelSizeField != null ? (ModelSize)_modelSizeField.value : ModelSize.Small;
                InferenceBackend selectedBackend = _backendField != null ? (InferenceBackend)_backendField.value : InferenceBackend.Auto;
                
                BackendType backend = BackendType.GPUCompute;
                if (selectedBackend == InferenceBackend.CPUBurst)
                {
                    backend = BackendType.CPU;
                }
                else if (selectedBackend == InferenceBackend.Auto)
                {
                    // Fall back to CPU by default for Base and Large due to known Sentis GPUCompute compiler bugs in this version (2.6.1)
                    if (selectedSize == ModelSize.Base || selectedSize == ModelSize.Large)
                    {
                        backend = BackendType.CPU;
                    }
                    else
                    {
                        backend = BackendType.GPUCompute;
                    }
                }

                // Initialize model from compiled ModelAsset
                _runner.Initialize(modelAsset, backend);

                EditorUtility.DisplayProgressBar("DepthForge", "Running Depth Inference...", 0.6f);
                
                int targetSize = 518;

                // Run inference
                Texture2D resultTex = _runner.Execute(inputTex, targetSize);

                if (resultTex != null)
                {
                    // Update preview
                    _outputPreview.style.backgroundImage = resultTex;

                    EditorUtility.DisplayProgressBar("DepthForge", "Exporting Texture...", 0.9f);
                    
                    // Export
                    CPritch.DepthForge.Editor.Utils.TextureExporter.ExportFormat format = 
                        (CPritch.DepthForge.Editor.Utils.TextureExporter.ExportFormat)_outputFormatField.value;
                        
                    string path = CPritch.DepthForge.Editor.Utils.TextureExporter.SaveHeightmap(resultTex, inputTex, format);
                    
                    if (path != null)
                    {
                        EditorUtility.DisplayDialog("Success", $"Heightmap generated and saved at:\n{path}", "Awesome");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Heightmap generation failed: {ex.Message}");
                EditorUtility.DisplayDialog("Generation Error", $"An error occurred during heightmap generation:\n{ex.Message}", "OK");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }
    }
}
