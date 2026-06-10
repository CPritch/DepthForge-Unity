using System;
using System.IO;
using System.Collections.Generic;
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
            // Low min so it can dock narrow (single-column); comfortable default when floated.
            wnd.minSize = new Vector2(360, 480);
            if (wnd.position.width < 760f)
            {
                Rect p = wnd.position;
                p.width = 980f;
                p.height = 640f;
                wnd.position = p;
            }
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

        public enum TilingAlignment
        {
            [InspectorName("Offset Only (Mean Shift)")]
            OffsetOnly,
            [InspectorName("Linear Regression (Full)")]
            LinearRegressionFull,
            [InspectorName("Linear Regression (Downscaled 64x64)")]
            LinearRegressionDownscaled
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
        // Phase 1 (R1): the current source/recipe/state as a Job. The UI still edits live for now;
        // this carries persistence (sidecar) and seeds the future batch queue.
        private CPritch.DepthForge.Editor.Data.Job _currentJob;
        private ObjectField _inputTextureField;
        private EnumField _modelSizeField;
        private EnumField _backendField;
        private EnumField _outputFormatField;
        
        // Sliders & Toggles
        private Slider _strengthSlider;
        private Slider _contrastSlider;
        private Slider _midpointSlider;
        private Slider _flattenSlider;
        private Slider _normalStrengthSlider;
        private Slider _aoStrengthSlider;
        private Toggle _invertToggle;
        private Toggle _exportNormalToggle;
        private Toggle _exportAOToggle;
        private Toggle _autoAssignToggle;
        private Toggle _tiledInferenceToggle;
        private Toggle _letterboxToggle;
        private EnumField _tilingAlignmentField;
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

        // Batch queue (R2)
        private List<CPritch.DepthForge.Editor.Data.Job> _queue = new List<CPritch.DepthForge.Editor.Data.Job>();
        private ListView _queueListView;
        private Button _addToQueueButton;
        private Button _clearQueueButton;
        private Button _batchGenerateButton;
        private Label _batchStatusLabel;
        private BatchProcessor _batch;

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
            _flattenSlider = root.Q<Slider>("flattenSlider");
            _invertToggle = root.Q<Toggle>("invertToggle");
            _tiledInferenceToggle = root.Q<Toggle>("tiledInferenceToggle");
            _letterboxToggle = root.Q<Toggle>("letterboxToggle");
            _tilingAlignmentField = root.Q<EnumField>("tilingAlignmentField");

            if (_tilingAlignmentField != null)
            {
                _tilingAlignmentField.Init(TilingAlignment.LinearRegressionDownscaled);
            }

            if (_tiledInferenceToggle != null)
            {
                _tiledInferenceToggle.RegisterValueChangedCallback(evt =>
                {
                    if (_tilingAlignmentField != null)
                    {
                        _tilingAlignmentField.SetEnabled(evt.newValue);
                    }
                });
                
                if (_tilingAlignmentField != null)
                {
                    _tilingAlignmentField.SetEnabled(_tiledInferenceToggle.value);
                }
            }
            
            // MicroSplat settings
            _exportNormalToggle = root.Q<Toggle>("exportNormalToggle");
            _normalStrengthSlider = root.Q<Slider>("normalStrengthSlider");
            _autoAssignToggle = root.Q<Toggle>("autoAssignToggle");
            _microSplatStatusLabel = root.Q<Label>("microSplatStatusLabel");

            if (_exportNormalToggle != null && _normalStrengthSlider != null)
            {
                _normalStrengthSlider.SetEnabled(_exportNormalToggle.value);
                _exportNormalToggle.RegisterValueChangedCallback(evt => _normalStrengthSlider.SetEnabled(evt.newValue));
            }

            _exportAOToggle = root.Q<Toggle>("exportAOToggle");
            _aoStrengthSlider = root.Q<Slider>("aoStrengthSlider");
            if (_exportAOToggle != null && _aoStrengthSlider != null)
            {
                _aoStrengthSlider.SetEnabled(_exportAOToggle.value);
                _exportAOToggle.RegisterValueChangedCallback(evt => _aoStrengthSlider.SetEnabled(evt.newValue));
            }

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

                if (_rawHeightmap != null)
                {
                    bool proceed = EditorUtility.DisplayDialog(
                        "Unsaved Heightmap Data",
                        "Changing the base texture will discard your current unsaved heightmap. Do you want to proceed?",
                        "Yes",
                        "Cancel"
                    );

                    if (!proceed)
                    {
                        _inputTextureField.SetValueWithoutNotify(evt.previousValue as Texture2D);
                        return;
                    }

                    // Clear heightmap since we are changing texture
                    CleanupPreviewResources();
                    SetActionButtonsEnabled(false);
                }

                if (tex != null)
                {
                    inputPreview.style.backgroundImage = tex;

                    // Phase 1 (R1/R5): wrap the source in a Job and reload any persisted recipe
                    // so prior depth work on this texture is restored.
                    _currentJob = new CPritch.DepthForge.Editor.Data.Job(tex);
                    var loadedRecipe = CPritch.DepthForge.Editor.Data.RecipeSidecar.Load(tex);
                    if (loadedRecipe != null)
                    {
                        _currentJob.recipe = loadedRecipe;

                        // Restore the cached raw depth so revisit is instant AND still editable:
                        // re-derive the adjusted map from raw + recipe, with no re-inference.
                        var cachedRaw = CPritch.DepthForge.Editor.Data.RawCache.LoadRaw(tex);
                        if (cachedRaw != null)
                        {
                            if (_rawHeightmap != null) DestroyImmediate(_rawHeightmap);
                            _rawHeightmap = cachedRaw;
                        }

                        ApplyRecipeToUI(loadedRecipe); // re-derives the adjusted map if a raw is present

                        if (_rawHeightmap != null) SetActionButtonsEnabled(true);
                    }
                }
                else
                {
                    inputPreview.style.backgroundImage = null;
                    _currentJob = null;
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
            if (_flattenSlider != null)
            {
                _flattenSlider.RegisterValueChangedCallback(evt => UpdateAdjustedHeightmap());
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

            // Batch queue (R2)
            _addToQueueButton = root.Q<Button>("addToQueueButton");
            _clearQueueButton = root.Q<Button>("clearQueueButton");
            _batchGenerateButton = root.Q<Button>("batchGenerateButton");
            _batchStatusLabel = root.Q<Label>("batchStatusLabel");
            _queueListView = root.Q<ListView>("queueListView");
            if (_queueListView != null)
            {
                _queueListView.fixedItemHeight = 20;
                _queueListView.itemsSource = _queue;
                _queueListView.makeItem = () => new Label();
                _queueListView.bindItem = (e, i) =>
                {
                    var job = _queue[i];
                    string name = job.source != null ? job.source.name : "(missing)";
                    string status = job.state.ToString();
                    if (job.state == CPritch.DepthForge.Editor.Data.JobState.Error && !string.IsNullOrEmpty(job.error))
                    {
                        status += $": {job.error}";
                    }
                    ((Label)e).text = $"{name}  —  {status}";
                };
                _queueListView.style.minHeight = 80;
            }
            if (_addToQueueButton != null) _addToQueueButton.clicked += AddSelectionToQueue;
            if (_clearQueueButton != null) _clearQueueButton.clicked += ClearQueue;
            if (_batchGenerateButton != null) _batchGenerateButton.clicked += StartBatch;
            UpdateBatchStatus();

            // Responsive: stack the three zones vertically when the window is too narrow for columns.
            var dfMain = root.Q<VisualElement>("dfMain");
            if (dfMain != null)
            {
                dfMain.RegisterCallback<GeometryChangedEvent>(evt =>
                {
                    float w = evt.newRect.width;
                    bool narrow = w > 0f && w < 720f;
                    if (dfMain.ClassListContains("df-narrow") != narrow)
                    {
                        dfMain.EnableInClassList("df-narrow", narrow);
                    }
                });
            }

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

            try
            {
                _runner.Initialize(modelAsset, backend);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Inference init failed: {ex.Message}");
                EditorUtility.DisplayDialog("Generation Error", $"Failed to initialize inference:\n{ex.Message}", "OK");
                return;
            }

            int targetSize = 518;
            bool useTiling = _tiledInferenceToggle != null ? _tiledInferenceToggle.value : false;
            DepthInferenceRunner.TilingAlignment tilingAlignment = _tilingAlignmentField != null ?
                (DepthInferenceRunner.TilingAlignment)_tilingAlignmentField.value :
                DepthInferenceRunner.TilingAlignment.LinearRegressionDownscaled;
            bool letterbox = _letterboxToggle != null ? _letterboxToggle.value : true;

            // Non-blocking: the editor stays responsive while inference runs; results arrive via callback.
            if (_generateButton != null)
            {
                _generateButton.SetEnabled(false);
                _generateButton.text = "Generating...";
            }
            if (_progressBar != null)
            {
                _progressBar.RemoveFromClassList("hidden");
                _progressBar.value = 0f;
                _progressBar.title = "Running depth inference...";
            }

            _runner.ExecuteAsync(inputTex, useTiling, tilingAlignment, letterbox,
                progress =>
                {
                    if (_progressBar != null) _progressBar.value = progress;
                },
                result =>
                {
                    EndGeneratingUI();
                    if (result != null)
                    {
                        if (_rawHeightmap != null) DestroyImmediate(_rawHeightmap);
                        _rawHeightmap = result;
                        UpdateAdjustedHeightmap();
                        SetActionButtonsEnabled(true);
                    }
                    Repaint();
                },
                error =>
                {
                    EndGeneratingUI();
                    Debug.LogError($"Heightmap generation failed: {error}");
                    EditorUtility.DisplayDialog("Generation Error", $"An error occurred during heightmap generation:\n{error}", "OK");
                    Repaint();
                });
        }

        private void EndGeneratingUI()
        {
            if (_progressBar != null) _progressBar.AddToClassList("hidden");
            if (_generateButton != null)
            {
                _generateButton.SetEnabled(true);
                _generateButton.text = "Generate Heightmap";
            }
        }

        // ---- Batch queue (R2) ---------------------------------------------------------------

        private void AddSelectionToQueue()
        {
            // Bulk path: a multi-selection of textures in the Project window. Otherwise fall back to
            // the Base Map picker above, so single adds work without touching the Project window.
            var candidates = new List<Texture2D>();
            foreach (var obj in Selection.objects)
            {
                if (obj is Texture2D selTex) candidates.Add(selTex);
            }
            if (candidates.Count == 0 && _inputTextureField?.value is Texture2D fieldTex)
            {
                candidates.Add(fieldTex);
            }

            if (candidates.Count == 0)
            {
                EditorUtility.DisplayDialog("Add to Queue",
                    "Nothing to add. Set a Base Map above, or select one or more textures in the Project window, then Add to Queue.",
                    "OK");
                return;
            }

            foreach (var tex in candidates)
            {
                if (tex == null) continue;
                if (_queue.Exists(j => j.source == tex)) continue;

                var job = new CPritch.DepthForge.Editor.Data.Job(tex);
                var recipe = CPritch.DepthForge.Editor.Data.RecipeSidecar.Load(tex);
                if (recipe != null) job.recipe = recipe;
                _queue.Add(job);
            }

            _queueListView?.RefreshItems();
            UpdateBatchStatus();
        }

        private void ClearQueue()
        {
            if (_batch != null && _batch.IsRunning) return;
            _queue.Clear();
            _queueListView?.RefreshItems();
            UpdateBatchStatus();
        }

        private void StartBatch()
        {
            if (_batch != null && _batch.IsRunning) return;
            if (_queue.Count == 0)
            {
                EditorUtility.DisplayDialog("Batch", "The queue is empty. Add some textures first.", "OK");
                return;
            }

            ModelSize size = _modelSizeField != null ? (ModelSize)_modelSizeField.value : ModelSize.Small;
            ModelAsset model = ResolveModelAsset(size);
            if (model == null)
            {
                EditorUtility.DisplayDialog("Batch",
                    "The selected model isn't downloaded yet. Run a single Generate once to download it, then batch.",
                    "OK");
                return;
            }

            InferenceBackend backendChoice = _backendField != null ? (InferenceBackend)_backendField.value : InferenceBackend.Auto;
            BackendType backend = ResolveBackend(size, backendChoice);

            _batchGenerateButton?.SetEnabled(false);
            _generateButton?.SetEnabled(false);

            _batch = new BatchProcessor(_runner, _queue, model, backend,
                job =>
                {
                    _queueListView?.RefreshItems();
                    UpdateBatchStatus();
                    Repaint();
                },
                (completed, total) =>
                {
                    if (_batchStatusLabel != null) _batchStatusLabel.text = $"Processing {completed}/{total}…";
                    Repaint();
                },
                () =>
                {
                    _batchGenerateButton?.SetEnabled(true);
                    _generateButton?.SetEnabled(true);
                    _queueListView?.RefreshItems();
                    UpdateBatchStatus();
                    AssetDatabase.Refresh();
                    Repaint();
                });
            _batch.Start();
        }

        private void UpdateBatchStatus()
        {
            if (_batchStatusLabel == null) return;
            if (_queue.Count == 0) { _batchStatusLabel.text = "Queue empty."; return; }

            int done = _queue.FindAll(j => j.state == CPritch.DepthForge.Editor.Data.JobState.Exported).Count;
            int errors = _queue.FindAll(j => j.state == CPritch.DepthForge.Editor.Data.JobState.Error).Count;
            _batchStatusLabel.text = $"{_queue.Count} queued · {done} done" + (errors > 0 ? $" · {errors} failed" : "");
        }

        private ModelAsset ResolveModelAsset(ModelSize size)
        {
            string modelPath = MODEL_SMALL_PATH;
            if (size == ModelSize.Base) modelPath = MODEL_BASE_PATH;
            else if (size == ModelSize.Large) modelPath = MODEL_LARGE_PATH;

            if (!File.Exists(modelPath)) return null;

            ModelAsset asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelPath);
            if (asset == null)
            {
                AssetDatabase.ImportAsset(modelPath);
                asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelPath);
            }
            return asset;
        }

        private BackendType ResolveBackend(ModelSize size, InferenceBackend choice)
        {
            if (choice == InferenceBackend.CPUBurst) return BackendType.CPU;
            if (choice == InferenceBackend.Auto)
            {
                return (size == ModelSize.Base || size == ModelSize.Large) ? BackendType.CPU : BackendType.GPUCompute;
            }
            return BackendType.GPUCompute;
        }

        private void UpdateAdjustedHeightmap()
        {
            if (_rawHeightmap == null) return;

            float contrast = _contrastSlider != null ? _contrastSlider.value : 1f;
            float midpoint = _midpointSlider != null ? _midpointSlider.value : 0.5f;
            bool invert = _invertToggle != null ? _invertToggle.value : true;
            float flatten = _flattenSlider != null ? _flattenSlider.value : 0f;

            Texture2D oldAdjusted = _adjustedHeightmap;
            _adjustedHeightmap = ImageProcessor.ApplyAdjustments(_rawHeightmap, contrast, midpoint, invert, flatten);

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

        // ---- Phase 1 (R1): Recipe <-> UI binding -------------------------------------------
        // The window's nested enums (ModelSize/InferenceBackend/TilingAlignment) mirror the
        // model-layer enums in order, so conversion is a straight int cast at this boundary.

        private CPritch.DepthForge.Editor.Data.Recipe BuildRecipeFromUI()
        {
            var r = new CPritch.DepthForge.Editor.Data.Recipe();
            if (_modelSizeField != null) r.modelSize = (CPritch.DepthForge.Editor.Data.DepthModelSize)(int)(ModelSize)_modelSizeField.value;
            if (_backendField != null) r.backend = (CPritch.DepthForge.Editor.Data.InferenceBackendChoice)(int)(InferenceBackend)_backendField.value;
            if (_tiledInferenceToggle != null) r.tiledInference = _tiledInferenceToggle.value;
            if (_letterboxToggle != null) r.letterbox = _letterboxToggle.value;
            if (_tilingAlignmentField != null) r.tilingAlignment = (CPritch.DepthForge.Editor.Data.TilingAlignment)(int)(TilingAlignment)_tilingAlignmentField.value;
            if (_invertToggle != null) r.invert = _invertToggle.value;
            if (_contrastSlider != null) r.contrast = _contrastSlider.value;
            if (_midpointSlider != null) r.midpoint = _midpointSlider.value;
            if (_flattenSlider != null) r.flatten = _flattenSlider.value;
            if (_exportNormalToggle != null) r.exportNormal = _exportNormalToggle.value;
            if (_normalStrengthSlider != null) r.normalStrength = _normalStrengthSlider.value;
            if (_exportAOToggle != null) r.exportAO = _exportAOToggle.value;
            if (_aoStrengthSlider != null) r.aoStrength = _aoStrengthSlider.value;
            if (_outputFormatField != null) r.format = (CPritch.DepthForge.Editor.Utils.TextureExporter.ExportFormat)_outputFormatField.value;
            if (_autoAssignToggle != null) r.autoAssignMicroSplat = _autoAssignToggle.value;
            // AO fields + preview-only strength are intentionally not bound here yet (AO = Phase 2).
            return r;
        }

        private void ApplyRecipeToUI(CPritch.DepthForge.Editor.Data.Recipe r)
        {
            if (r == null) return;
            if (_modelSizeField != null) _modelSizeField.SetValueWithoutNotify((ModelSize)(int)r.modelSize);
            if (_backendField != null) _backendField.SetValueWithoutNotify((InferenceBackend)(int)r.backend);
            if (_tiledInferenceToggle != null) _tiledInferenceToggle.SetValueWithoutNotify(r.tiledInference);
            if (_letterboxToggle != null) _letterboxToggle.SetValueWithoutNotify(r.letterbox);
            if (_tilingAlignmentField != null)
            {
                _tilingAlignmentField.SetValueWithoutNotify((TilingAlignment)(int)r.tilingAlignment);
                _tilingAlignmentField.SetEnabled(r.tiledInference);
            }
            if (_invertToggle != null) _invertToggle.SetValueWithoutNotify(r.invert);
            if (_contrastSlider != null) _contrastSlider.SetValueWithoutNotify(r.contrast);
            if (_midpointSlider != null) _midpointSlider.SetValueWithoutNotify(r.midpoint);
            if (_flattenSlider != null) _flattenSlider.SetValueWithoutNotify(r.flatten);
            if (_exportNormalToggle != null) _exportNormalToggle.SetValueWithoutNotify(r.exportNormal);
            if (_normalStrengthSlider != null)
            {
                _normalStrengthSlider.SetValueWithoutNotify(r.normalStrength);
                _normalStrengthSlider.SetEnabled(r.exportNormal);
            }
            if (_exportAOToggle != null) _exportAOToggle.SetValueWithoutNotify(r.exportAO);
            if (_aoStrengthSlider != null)
            {
                _aoStrengthSlider.SetValueWithoutNotify(r.aoStrength);
                _aoStrengthSlider.SetEnabled(r.exportAO);
            }
            if (_outputFormatField != null) _outputFormatField.SetValueWithoutNotify(r.format);
            // Refresh the derived preview only if a heightmap already exists for this source.
            if (_rawHeightmap != null) UpdateAdjustedHeightmap();
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
                // Optionally derive a normal map from the same adjusted heightmap. This gives MicroSplat
                // real normal data rather than letting it synthesize (often poor) normals from the diffuse.
                string normalPath = null;
                bool exportNormal = _exportNormalToggle != null && _exportNormalToggle.value;
                if (exportNormal)
                {
                    float normalStrength = _normalStrengthSlider != null ? _normalStrengthSlider.value : 8f;
                    Texture2D normalTex = ImageProcessor.GenerateNormalMap(_adjustedHeightmap, normalStrength);
                    if (normalTex != null)
                    {
                        normalPath = CPritch.DepthForge.Editor.Utils.TextureExporter.SaveNormalMap(normalTex, inputTex);
                        DestroyImmediate(normalTex);
                    }
                }

                // Optionally derive an ambient-occlusion map from the same adjusted heightmap so the
                // exported map set (Height + Normal + AO) is complete and consistent.
                string aoPath = null;
                bool exportAO = _exportAOToggle != null && _exportAOToggle.value;
                if (exportAO)
                {
                    float aoStrength = _aoStrengthSlider != null ? _aoStrengthSlider.value : 1f;
                    float aoRadius = _currentJob != null ? _currentJob.recipe.aoRadius : 0.02f;
                    Texture2D aoTex = ImageProcessor.GenerateAO(_adjustedHeightmap, aoStrength, aoRadius);
                    if (aoTex != null)
                    {
                        aoPath = CPritch.DepthForge.Editor.Utils.TextureExporter.SaveAOMap(aoTex, inputTex);
                        DestroyImmediate(aoTex);
                    }
                }

                bool autoAssign = _autoAssignToggle != null ? _autoAssignToggle.value : false;
                if (autoAssign && IsMicroSplatPresent())
                {
                    AutoAssignToMicroSplat(path, normalPath, inputTex);
                }

                // Phase 1 (R5): persist the recipe sidecar so this depth work is re-editable later.
                var savedRecipe = BuildRecipeFromUI();
                if (_currentJob == null) _currentJob = new CPritch.DepthForge.Editor.Data.Job(inputTex);
                _currentJob.recipe = savedRecipe;
                _currentJob.state = CPritch.DepthForge.Editor.Data.JobState.Exported;
                CPritch.DepthForge.Editor.Data.RecipeSidecar.Save(inputTex, savedRecipe);
                // Cache the raw (pre-adjustment) depth so revisiting this source restores an
                // editable result without re-running inference.
                CPritch.DepthForge.Editor.Data.RawCache.SaveRaw(inputTex, _rawHeightmap);

                string message = $"Heightmap exported to:\n{path}";
                if (normalPath != null) message += $"\n\nNormal map:\n{normalPath}";
                if (aoPath != null) message += $"\n\nAO map:\n{aoPath}";
                EditorUtility.DisplayDialog("Success", message, "Awesome");
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

        private void AutoAssignToMicroSplat(string heightmapPath, string normalPath, Texture2D sourceDiffuse)
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

            Texture2D normalTex = string.IsNullOrEmpty(normalPath)
                ? null
                : AssetDatabase.LoadAssetAtPath<Texture2D>(normalPath);

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

                        if (normalTex != null)
                        {
                            var normalField = entryType.GetField("normal");
                            if (normalField != null)
                            {
                                normalField.SetValue(entry, normalTex);
                                configChanged = true;
                                assignedAny = true;
                                Debug.Log($"[DepthForge] Auto-assigned normal map to MicroSplat Config '{config.name}' for diffuse texture '{sourceDiffuse.name}'");
                            }
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
