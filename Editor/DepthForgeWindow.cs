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
            wnd.minSize = new Vector2(450, 600);
        }

        public void CreateGUI()
        {
            VisualElement root = rootVisualElement;

            var visualTree = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>("Packages/com.cpritch.depthforge/Editor/UI/DepthForgeLayout.uxml");
            
            if (visualTree == null)
            {
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

        // Runner and UI Fields
        private DepthInferenceRunner _runner;
        private ObjectField _inputTextureField;
        private EnumField _modelSizeField;
        private EnumField _backendField;
        private EnumField _outputFormatField;
        
        // Sliders & Toggles
        private Slider _strengthSlider;
        private Slider _contrastSlider;
        private Slider _midpointSlider;
        private Toggle _invertToggle;
        private Toggle _autoAssignToggle;
        private Toggle _texturedPreviewToggle;
        private Label _microSplatStatusLabel;

        // Tabs & Viewports
        private Button _btnPreview2D;
        private Button _btnPreview3D;
        private VisualElement _previewControls3D;
        private VisualElement _outputPreview2D;
        private IMGUIContainer _outputPreview3D;

        // Action Buttons
        private Button _generateButton;
        private Button _saveButton;
        private Button _previewMeshButton;
        private ProgressBar _progressBar;

        // Downloader
        private DownloadHandle _downloadHandle;

        // 3D Preview State
        private Texture2D _rawHeightmap;
        private Texture2D _adjustedHeightmap;
        private bool _show3DPreview = false;
        private PreviewRenderUtility _previewUtility;
        private Mesh _previewMesh;
        private Material _previewMaterial;
        private Texture2D _checkerboardTex;
        private Vector2 _previewDrag = new Vector2(45f, 45f);

        // Scene preview state
        private GameObject _inScenePreviewObject;

        private void OnEnable()
        {
            _runner = new DepthInferenceRunner();
            Selection.selectionChanged += OnSelectionChanged;
        }

        private void OnDisable()
        {
            _runner?.Dispose();
            Selection.selectionChanged -= OnSelectionChanged;
            
            if (_downloadHandle != null)
            {
                _downloadHandle.Cancel();
                _downloadHandle = null;
            }

            CleanupPreviewResources();
        }

        private void CleanupPreviewResources()
        {
            if (_rawHeightmap != null)
            {
                DestroyImmediate(_rawHeightmap);
                _rawHeightmap = null;
            }

            if (_adjustedHeightmap != null)
            {
                DestroyImmediate(_adjustedHeightmap);
                _adjustedHeightmap = null;
            }

            if (_previewMesh != null)
            {
                DestroyImmediate(_previewMesh);
                _previewMesh = null;
            }

            if (_previewMaterial != null)
            {
                DestroyImmediate(_previewMaterial);
                _previewMaterial = null;
            }

            if (_checkerboardTex != null)
            {
                DestroyImmediate(_checkerboardTex);
                _checkerboardTex = null;
            }

            if (_previewUtility != null)
            {
                _previewUtility.Cleanup();
                _previewUtility = null;
            }

            CleanupScenePreview();
        }

        private void CleanupScenePreview()
        {
            if (_inScenePreviewObject != null)
            {
                DestroyImmediate(_inScenePreviewObject);
                _inScenePreviewObject = null;
            }
        }

        private void OnSelectionChanged()
        {
            // If the user selects something else, clean up the temporary preview mesh to keep the scene tidy
            if (_inScenePreviewObject != null && Selection.activeGameObject != _inScenePreviewObject)
            {
                CleanupScenePreview();
                if (_previewMeshButton != null)
                {
                    _previewMeshButton.text = "Spawn Preview Mesh";
                }
            }
        }

        private void BindUI(VisualElement root)
        {
            // Core controls
            _generateButton = root.Q<Button>("generateButton");
            _inputTextureField = root.Q<ObjectField>("inputTextureField");
            _modelSizeField = root.Q<EnumField>("modelSizeField");
            _backendField = root.Q<EnumField>("backendField");
            _outputFormatField = root.Q<EnumField>("outputFormatField");
            
            // Adjustments
            _strengthSlider = root.Q<Slider>("strengthSlider");
            _contrastSlider = root.Q<Slider>("contrastSlider");
            _midpointSlider = root.Q<Slider>("midpointSlider");
            _invertToggle = root.Q<Toggle>("invertToggle");
            
            // MicroSplat settings
            _autoAssignToggle = root.Q<Toggle>("autoAssignToggle");
            _microSplatStatusLabel = root.Q<Label>("microSplatStatusLabel");

            // Tabs & Preview
            _btnPreview2D = root.Q<Button>("btnPreview2D");
            _btnPreview3D = root.Q<Button>("btnPreview3D");
            _previewControls3D = root.Q<VisualElement>("previewControls3D");
            _outputPreview2D = root.Q<VisualElement>("outputPreview2D");
            _outputPreview3D = root.Q<IMGUIContainer>("outputPreview3D");
            _texturedPreviewToggle = root.Q<Toggle>("texturedPreviewToggle");

            var inputPreview = root.Q<VisualElement>("inputPreview");
            _progressBar = root.Q<ProgressBar>("progressBar");

            // Action buttons
            _saveButton = root.Q<Button>("saveButton");
            _previewMeshButton = root.Q<Button>("previewMeshButton");

            // Set up ranges and values
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

            // Status checks
            bool msPresent = IsMicroSplatPresent();
            if (_microSplatStatusLabel != null)
            {
                if (msPresent)
                {
                    _microSplatStatusLabel.text = "MicroSplat Core detected! Auto-assign will update your TextureArrayConfig.";
                    _microSplatStatusLabel.style.color = new StyleColor(new Color(0.35f, 0.75f, 0.35f));
                }
                else
                {
                    _microSplatStatusLabel.text = "MicroSplat not found in project. Auto-assignment disabled.";
                    _microSplatStatusLabel.style.color = new StyleColor(new Color(0.85f, 0.45f, 0.45f));
                    if (_autoAssignToggle != null)
                    {
                        _autoAssignToggle.value = false;
                        _autoAssignToggle.SetEnabled(false);
                    }
                }
            }

            // Initially disable action buttons
            SetActionButtonsEnabled(false);

            // Register input texture preview callback
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
                UpdatePreviewMaterial();
                Repaint();
            });

            // Sliders callback
            if (_strengthSlider != null)
            {
                _strengthSlider.RegisterValueChangedCallback(evt =>
                {
                    UpdatePreviewMesh();
                    Repaint();
                });
            }
            if (_contrastSlider != null)
            {
                _contrastSlider.RegisterValueChangedCallback(evt => UpdateAdjustedHeightmap());
            }
            if (_midpointSlider != null)
            {
                _midpointSlider.RegisterValueChangedCallback(evt => UpdateAdjustedHeightmap());
            }
            if (_invertToggle != null)
            {
                _invertToggle.RegisterValueChangedCallback(evt => UpdateAdjustedHeightmap());
            }
            if (_texturedPreviewToggle != null)
            {
                _texturedPreviewToggle.RegisterValueChangedCallback(evt =>
                {
                    UpdatePreviewMaterial();
                    Repaint();
                });
            }

            // Tab navigation callbacks
            if (_btnPreview2D != null) _btnPreview2D.clicked += () => SetPreviewMode(false);
            if (_btnPreview3D != null) _btnPreview3D.clicked += () => SetPreviewMode(true);

            // Main actions callbacks
            _generateButton.clicked += OnGenerateClicked;
            if (_saveButton != null) _saveButton.clicked += OnSaveClicked;
            if (_previewMeshButton != null) _previewMeshButton.clicked += OnPreviewMeshClicked;

            // Bind the IMGUI viewport rendering
            if (_outputPreview3D != null)
            {
                _outputPreview3D.onGUIHandler = () =>
                {
                    Rect localRect = _outputPreview3D.contentRect;
                    if (localRect.width <= 0 || localRect.height <= 0)
                    {
                        localRect = _outputPreview3D.layout;
                    }
                    Draw3DPreview(localRect);
                };
            }
        }

        private void SetActionButtonsEnabled(bool enabled)
        {
            if (_saveButton != null) _saveButton.SetEnabled(enabled);
            if (_previewMeshButton != null) _previewMeshButton.SetEnabled(enabled);
        }

        private void SetPreviewMode(bool show3D)
        {
            _show3DPreview = show3D;
            if (show3D)
            {
                _btnPreview2D?.RemoveFromClassList("active-tab");
                _btnPreview3D?.AddToClassList("active-tab");

                _outputPreview2D?.AddToClassList("hidden");
                _outputPreview3D?.RemoveFromClassList("hidden");
                _previewControls3D?.RemoveFromClassList("hidden");
            }
            else
            {
                _btnPreview2D?.AddToClassList("active-tab");
                _btnPreview3D?.RemoveFromClassList("active-tab");

                _outputPreview2D?.RemoveFromClassList("hidden");
                _outputPreview3D?.AddToClassList("hidden");
                _previewControls3D?.AddToClassList("hidden");
            }
            Repaint();
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
                    AssetDatabase.ImportAsset(modelPath);
                    AssetDatabase.Refresh();
                    modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelPath);
                }

                if (modelAsset == null)
                {
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

            _downloadHandle = CPritch.DepthForge.Editor.Utils.ModelDownloader.DownloadModel(
                modelUrl,
                modelPath,
                progress =>
                {
                    float currentProgress = progress * 0.05f;
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

                    _progressBar.title = "Downloading depth model (weights)...";
                    _downloadHandle = CPritch.DepthForge.Editor.Utils.ModelDownloader.DownloadModel(
                        modelDataUrl,
                        modelDataPath,
                        progress =>
                        {
                            float currentProgress = 0.05f + progress * 0.95f;
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
                    if (selectedSize == ModelSize.Base || selectedSize == ModelSize.Large)
                    {
                        backend = BackendType.CPU;
                    }
                    else
                    {
                        backend = BackendType.GPUCompute;
                    }
                }

                _runner.Initialize(modelAsset, backend);

                EditorUtility.DisplayProgressBar("DepthForge", "Running Depth Inference...", 0.6f);
                
                int targetSize = 518;

                Texture2D resultTex = _runner.Execute(inputTex, targetSize);

                if (resultTex != null)
                {
                    if (_rawHeightmap != null)
                    {
                        DestroyImmediate(_rawHeightmap);
                    }
                    _rawHeightmap = resultTex;

                    // Compute initial preview heightmap
                    UpdateAdjustedHeightmap();

                    // Enable editing actions
                    SetActionButtonsEnabled(true);
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

        private void UpdateAdjustedHeightmap()
        {
            if (_rawHeightmap == null) return;

            float contrast = _contrastSlider != null ? _contrastSlider.value : 1f;
            float midpoint = _midpointSlider != null ? _midpointSlider.value : 0.5f;
            bool invert = _invertToggle != null ? _invertToggle.value : true;

            Texture2D oldAdjusted = _adjustedHeightmap;
            _adjustedHeightmap = ImageProcessor.ApplyAdjustments(_rawHeightmap, contrast, midpoint, invert);

            if (oldAdjusted != null)
            {
                DestroyImmediate(oldAdjusted);
            }

            if (_outputPreview2D != null)
            {
                _outputPreview2D.style.backgroundImage = _adjustedHeightmap;
            }

            // Regenerate the 3D meshes/materials based on the new pixels
            UpdatePreviewMesh();
            UpdatePreviewMaterial();
            Repaint();
        }

        private void UpdatePreviewMesh()
        {
            if (_previewMesh == null)
            {
                _previewMesh = new Mesh();
                _previewMesh.name = "DepthForgePreviewMesh";
                _previewMesh.hideFlags = HideFlags.HideAndDontSave;
            }

            int resolution = 128;
            int vertexCount = resolution * resolution;
            Vector3[] vertices = new Vector3[vertexCount];
            Vector2[] uvs = new Vector2[vertexCount];
            int[] triangles = new int[(resolution - 1) * (resolution - 1) * 6];

            float strength = _strengthSlider != null ? _strengthSlider.value : 1f;

            for (int y = 0; y < resolution; y++)
            {
                float v = (float)y / (resolution - 1);
                for (int x = 0; x < resolution; x++)
                {
                    float u = (float)x / (resolution - 1);
                    int idx = y * resolution + x;

                    float h = 0f;
                    if (_adjustedHeightmap != null)
                    {
                        Color col = _adjustedHeightmap.GetPixelBilinear(u, v);
                        h = col.r;
                    }

                    float posX = u - 0.5f;
                    float posZ = v - 0.5f;

                    float posY = h * strength * 0.2f;

                    vertices[idx] = new Vector3(posX, posY, posZ);
                    uvs[idx] = new Vector2(u, v);
                }
            }

            int triIdx = 0;
            for (int y = 0; y < resolution - 1; y++)
            {
                for (int x = 0; x < resolution - 1; x++)
                {
                    int row1 = y * resolution;
                    int row2 = (y + 1) * resolution;

                    triangles[triIdx++] = row1 + x;
                    triangles[triIdx++] = row2 + x;
                    triangles[triIdx++] = row1 + x + 1;

                    triangles[triIdx++] = row1 + x + 1;
                    triangles[triIdx++] = row2 + x;
                    triangles[triIdx++] = row2 + x + 1;
                }
            }

            _previewMesh.Clear();
            _previewMesh.vertices = vertices;
            _previewMesh.uv = uvs;
            _previewMesh.triangles = triangles;
            _previewMesh.RecalculateNormals();
            _previewMesh.RecalculateBounds();
        }

        private Shader GetPreviewShader()
        {
            Shader shader = null;

            // Check if URP or HDRP is active
            if (UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline != null)
            {
                string rpName = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline.GetType().Name;
                if (rpName.Contains("Universal") || rpName.Contains("URP"))
                {
                    shader = Shader.Find("Universal Render Pipeline/Lit");
                    if (shader == null) shader = Shader.Find("Universal Render Pipeline/Simple Lit");
                }
                else if (rpName.Contains("HDRP") || rpName.Contains("HighDefinition") || rpName.Contains("HDRenderPipeline"))
                {
                    shader = Shader.Find("HDRP/Lit");
                }
            }

            // Fallbacks
            if (shader == null) shader = Shader.Find("Standard");
            if (shader == null) shader = Shader.Find("Diffuse");
            if (shader == null) shader = Shader.Find("Hidden/Internal-Colored");

            return shader;
        }

        private void UpdatePreviewMaterial()
        {
            if (_previewMaterial == null)
            {
                Shader shader = GetPreviewShader();
                _previewMaterial = new Material(shader);
                _previewMaterial.hideFlags = HideFlags.HideAndDontSave;
            }

            bool drape = _texturedPreviewToggle != null ? _texturedPreviewToggle.value : true;
            Texture2D inputTex = drape ? (_inputTextureField.value as Texture2D) : null;

            if (inputTex != null)
            {
                if (_previewMaterial.HasProperty("_MainTex")) _previewMaterial.SetTexture("_MainTex", inputTex);
                if (_previewMaterial.HasProperty("_BaseMap")) _previewMaterial.SetTexture("_BaseMap", inputTex);
                if (_previewMaterial.HasProperty("_Color")) _previewMaterial.SetColor("_Color", Color.white);
                if (_previewMaterial.HasProperty("_BaseColor")) _previewMaterial.SetColor("_BaseColor", Color.white);
            }
            else
            {
                Texture2D checkerboard = GetCheckerboardTexture();
                if (_previewMaterial.HasProperty("_MainTex")) _previewMaterial.SetTexture("_MainTex", checkerboard);
                if (_previewMaterial.HasProperty("_BaseMap")) _previewMaterial.SetTexture("_BaseMap", checkerboard);
                if (_previewMaterial.HasProperty("_Color")) _previewMaterial.SetColor("_Color", Color.gray);
                if (_previewMaterial.HasProperty("_BaseColor")) _previewMaterial.SetColor("_BaseColor", Color.gray);
            }
        }

        private Texture2D GetCheckerboardTexture()
        {
            if (_checkerboardTex == null)
            {
                _checkerboardTex = new Texture2D(16, 16, TextureFormat.RGB24, false, true);
                _checkerboardTex.filterMode = FilterMode.Point;
                _checkerboardTex.hideFlags = HideFlags.HideAndDontSave;
                
                Color32[] colors = new Color32[16 * 16];
                for (int y = 0; y < 16; y++)
                {
                    for (int x = 0; x < 16; x++)
                    {
                        bool isWhite = ((x / 4) + (y / 4)) % 2 == 0;
                        byte val = isWhite ? (byte)220 : (byte)150;
                        colors[y * 16 + x] = new Color32(val, val, val, 255);
                    }
                }
                _checkerboardTex.SetPixels32(colors);
                _checkerboardTex.Apply();
            }
            return _checkerboardTex;
        }

        private void Draw3DPreview(Rect rect)
        {
            if (_adjustedHeightmap == null)
            {
                GUI.Box(rect, "No heightmap generated yet", new GUIStyle(GUI.skin.box)
                {
                    alignment = TextAnchor.MiddleCenter,
                    fontSize = 12,
                    normal = { textColor = Color.gray }
                });
                return;
            }

            if (_previewUtility == null)
            {
                _previewUtility = new PreviewRenderUtility();
                _previewUtility.camera.fieldOfView = 30f;
            }

            Event evt = Event.current;
            if (rect.Contains(evt.mousePosition))
            {
                if (evt.type == EventType.MouseDrag && evt.button == 0)
                {
                    _previewDrag.x += evt.delta.x * 0.8f;
                    _previewDrag.y = Mathf.Clamp(_previewDrag.y + evt.delta.y * 0.8f, -80f, 80f);
                    evt.Use();
                    Repaint();
                }
            }

            _previewUtility.BeginPreview(rect, GUIStyle.none);

            float distance = 1.8f;
            Quaternion camRot = Quaternion.Euler(_previewDrag.y, _previewDrag.x, 0f);
            Vector3 camPos = camRot * new Vector3(0f, 0f, -distance);

            _previewUtility.camera.transform.position = camPos;
            _previewUtility.camera.transform.rotation = camRot;
            _previewUtility.camera.nearClipPlane = 0.05f;
            _previewUtility.camera.farClipPlane = 10f;

            _previewUtility.lights[0].enabled = true;
            _previewUtility.lights[0].transform.rotation = Quaternion.Euler(40f, 40f, 0f);
            _previewUtility.lights[0].intensity = 1.3f;
            _previewUtility.lights[0].color = Color.white;

            if (_previewMesh != null)
            {
                if (_previewMaterial == null)
                {
                    UpdatePreviewMaterial();
                }
                _previewUtility.DrawMesh(_previewMesh, Vector3.zero, Quaternion.identity, _previewMaterial, 0);
            }

            _previewUtility.camera.Render();
            Texture renderedTex = _previewUtility.EndPreview();
            GUI.DrawTexture(rect, renderedTex, ScaleMode.StretchToFill, false);
        }

        private void OnSaveClicked()
        {
            if (_adjustedHeightmap == null) return;

            Texture2D inputTex = _inputTextureField.value as Texture2D;
            if (inputTex == null) return;

            CPritch.DepthForge.Editor.Utils.TextureExporter.ExportFormat format = 
                (CPritch.DepthForge.Editor.Utils.TextureExporter.ExportFormat)_outputFormatField.value;

            string path = CPritch.DepthForge.Editor.Utils.TextureExporter.SaveHeightmap(_adjustedHeightmap, inputTex, format);

            if (path != null)
            {
                bool autoAssign = _autoAssignToggle != null ? _autoAssignToggle.value : false;
                if (autoAssign && IsMicroSplatPresent())
                {
                    AutoAssignToMicroSplat(path, inputTex);
                }
                
                EditorUtility.DisplayDialog("Success", $"Heightmap exported successfully to:\n{path}", "Awesome");
            }
        }

        private void OnPreviewMeshClicked()
        {
            if (_inScenePreviewObject != null)
            {
                CleanupScenePreview();
                if (_previewMeshButton != null) _previewMeshButton.text = "Spawn Preview Mesh";
                return;
            }

            if (_adjustedHeightmap == null) return;

            _inScenePreviewObject = new GameObject("DepthForge_PreviewPlane");
            _inScenePreviewObject.hideFlags = HideFlags.DontSave;

            MeshFilter filter = _inScenePreviewObject.AddComponent<MeshFilter>();
            MeshRenderer renderer = _inScenePreviewObject.AddComponent<MeshRenderer>();

            filter.sharedMesh = _previewMesh;

            Shader shader = GetPreviewShader();
            Material mat = new Material(shader);
            mat.hideFlags = HideFlags.DontSave;

            Texture2D inputTex = _inputTextureField.value as Texture2D;
            if (inputTex != null)
            {
                if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", inputTex);
                if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", inputTex);
            }
            renderer.sharedMaterial = mat;

            if (SceneView.lastActiveSceneView != null && SceneView.lastActiveSceneView.camera != null)
            {
                Vector3 spawnPos = SceneView.lastActiveSceneView.camera.transform.position + SceneView.lastActiveSceneView.camera.transform.forward * 2.0f;
                _inScenePreviewObject.transform.position = spawnPos;
                _inScenePreviewObject.transform.localScale = Vector3.one * 2f;
            }
            
            Selection.activeGameObject = _inScenePreviewObject;

            if (_previewMeshButton != null) _previewMeshButton.text = "Clear Preview Mesh";
        }

        private bool IsMicroSplatPresent()
        {
            return FindTypeInAssemblies("JBooth.MicroSplat.TextureArrayConfig") != null;
        }

        private Type FindTypeInAssemblies(string typeName)
        {
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type t = assembly.GetType(typeName);
                if (t != null) return t;
            }
            return null;
        }

        private void AutoAssignToMicroSplat(string heightmapPath, Texture2D sourceDiffuse)
        {
            string[] guids = AssetDatabase.FindAssets("t:JBooth.MicroSplat.TextureArrayConfig");
            if (guids == null || guids.Length == 0)
            {
                guids = AssetDatabase.FindAssets("t:TextureArrayConfig");
            }

            if (guids == null || guids.Length == 0)
            {
                Debug.LogWarning("[DepthForge] No MicroSplat TextureArrayConfig found in the project. Skipping auto-assignment.");
                return;
            }

            Texture2D heightmapTex = AssetDatabase.LoadAssetAtPath<Texture2D>(heightmapPath);
            if (heightmapTex == null)
            {
                Debug.LogError($"[DepthForge] Failed to load generated heightmap at: {heightmapPath}");
                return;
            }

            bool assignedAny = false;

            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                ScriptableObject config = AssetDatabase.LoadAssetAtPath<ScriptableObject>(path);
                if (config == null) continue;

                Type configType = config.GetType();
                if (configType.FullName != "JBooth.MicroSplat.TextureArrayConfig") continue;

                var sourceTexturesField = configType.GetField("sourceTextures");
                if (sourceTexturesField == null) continue;

                System.Collections.IList sourceTexturesList = sourceTexturesField.GetValue(config) as System.Collections.IList;
                if (sourceTexturesList == null) continue;

                bool configChanged = false;

                foreach (var entry in sourceTexturesList)
                {
                    if (entry == null) continue;
                    Type entryType = entry.GetType();

                    var diffuseField = entryType.GetField("diffuse");
                    if (diffuseField == null) continue;

                    Texture2D diffuseTex = diffuseField.GetValue(entry) as Texture2D;
                    if (diffuseTex == sourceDiffuse)
                    {
                        var heightField = entryType.GetField("height");
                        if (heightField != null)
                        {
                            heightField.SetValue(entry, heightmapTex);
                            configChanged = true;
                            assignedAny = true;
                            Debug.Log($"[DepthForge] Auto-assigned heightmap to MicroSplat Config '{config.name}' for diffuse texture '{sourceDiffuse.name}'");
                        }
                    }
                }

                if (configChanged)
                {
                    EditorUtility.SetDirty(config);
                    AssetDatabase.SaveAssets();

                    Type editorType = FindTypeInAssemblies("JBooth.MicroSplat.TextureArrayConfigEditor");
                    if (editorType != null)
                    {
                        var compileMethod = editorType.GetMethod("CompileConfig", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static, null, new Type[] { configType }, null);
                        if (compileMethod != null)
                        {
                            Debug.Log($"[DepthForge] Compiling MicroSplat Texture Array Config: {config.name}");
                            compileMethod.Invoke(null, new object[] { config });
                        }
                        else
                        {
                            Debug.LogWarning("[DepthForge] MicroSplat compile method 'CompileConfig' not found.");
                        }
                    }
                    else
                    {
                        Debug.LogWarning("[DepthForge] MicroSplat Editor type 'JBooth.MicroSplat.TextureArrayConfigEditor' not found.");
                    }
                }
            }

            if (!assignedAny)
            {
                Debug.LogWarning($"[DepthForge] Could not find any MicroSplat config entry matching diffuse texture '{sourceDiffuse.name}'");
            }
        }
    }
}
