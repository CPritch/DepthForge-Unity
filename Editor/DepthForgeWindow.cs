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

        // Map providers (R11): inference is decoupled behind IMapProvider. _provider is the focused one.
        private System.Collections.Generic.List<IMapProvider> _providers;
        private IMapProvider _provider;
        private DropdownField _providerField;
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
        private Toggle _exportRoughnessToggle;
        private Toggle _autoAssignToggle;
        private Toggle _tiledInferenceToggle;
        private Toggle _letterboxToggle;
        private EnumField _tilingAlignmentField;
        private Toggle _texturedPreviewToggle;
        private Label _microSplatStatusLabel;

        // Tabs & Viewports
        private enum PreviewTab { Source, Height, Normal, Roughness, AO, ThreeD }
        private PreviewTab _activeTab = PreviewTab.Height;
        private Button _tabSource;
        private Button _tabHeight;
        private Button _tabNormal;
        private Button _tabRoughness;
        private Button _tabAO;
        private Button _tab3D;
        private Texture2D _normalPreview;   // derived-from-height normal (depth-style providers)
        private Texture2D _aoPreview;
        // Native model outputs for the focused source (owned here); independent of height tuning.
        private Texture2D _nativeNormal;
        private Texture2D _nativeRoughness;
        private VisualElement _previewControls3D;
        private VisualElement _outputPreview2D;
        private IMGUIContainer _outputPreview3D;

        // Action Buttons
        private Button _generateButton;
        private Button _saveButton;
        private Button _previewMeshButton;
        private ProgressBar _progressBar;

        // Presets (R6)
        private DropdownField _presetDropdown;
        private TextField _presetNameField;
        private Label _presetDirtyMarker;
        private Button _savePresetButton;
        private Button _saveAsPresetButton;
        private Button _deletePresetButton;
        private string _selectedPresetName;

        // Batch queue (R2)
        private List<CPritch.DepthForge.Editor.Data.Job> _queue = new List<CPritch.DepthForge.Editor.Data.Job>();
        private ListView _queueListView;
        private Button _addToQueueButton;
        private Button _clearQueueButton;
        private Button _batchGenerateButton;
        private Label _batchStatusLabel;
        private BatchProcessor _batch;

        // 3D Preview State
        private Texture2D _rawHeightmap;
        private Texture2D _adjustedHeightmap;
        private PreviewRenderUtility _previewUtility;
        private Mesh _previewMesh;
        private Material _previewMaterial;
        private Texture2D _checkerboardTex;
        private Vector2 _previewDrag = new Vector2(45f, 45f);
        // (legacy _show3DPreview removed — preview is now tab-driven via _activeTab)

        // Scene preview state
        private GameObject _inScenePreviewObject;

        private void OnEnable()
        {
            _providers = ProviderRegistry.CreateAll();
            _provider = ProviderRegistry.Default(_providers);
            Selection.selectionChanged += OnSelectionChanged;
        }

        private void OnDisable()
        {
            if (_providers != null)
            {
                foreach (var p in _providers) p?.Dispose();
                _providers = null;
                _provider = null;
            }
            Selection.selectionChanged -= OnSelectionChanged;

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

            if (_normalPreview != null)
            {
                DestroyImmediate(_normalPreview);
                _normalPreview = null;
            }

            if (_aoPreview != null)
            {
                DestroyImmediate(_aoPreview);
                _aoPreview = null;
            }

            if (_nativeNormal != null)
            {
                DestroyImmediate(_nativeNormal);
                _nativeNormal = null;
            }

            if (_nativeRoughness != null)
            {
                DestroyImmediate(_nativeRoughness);
                _nativeRoughness = null;
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
            _providerField = root.Q<DropdownField>("providerField");
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

            _exportRoughnessToggle = root.Q<Toggle>("exportRoughnessToggle");

            // Tabs & Preview
            _tabSource = root.Q<Button>("tabSource");
            _tabHeight = root.Q<Button>("tabHeight");
            _tabNormal = root.Q<Button>("tabNormal");
            _tabRoughness = root.Q<Button>("tabRoughness");
            _tabAO = root.Q<Button>("tabAO");
            _tab3D = root.Q<Button>("tab3D");
            _previewControls3D = root.Q<VisualElement>("previewControls3D");
            _outputPreview2D = root.Q<VisualElement>("outputPreview2D");
            _outputPreview3D = root.Q<IMGUIContainer>("outputPreview3D");
            _texturedPreviewToggle = root.Q<Toggle>("texturedPreviewToggle");

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

            // Provider dropdown (R11): lists the available providers; reference (non-commercial) ones
            // carry that label and are compiled out of shippable builds.
            if (_providerField != null && _providers != null)
            {
                var names = new List<string>(_providers.Count);
                foreach (var p in _providers) names.Add(p.Info.displayName);
                _providerField.choices = names;
                _providerField.SetValueWithoutNotify(_provider != null ? _provider.Info.displayName : (names.Count > 0 ? names[0] : null));
                _providerField.RegisterValueChangedCallback(evt => OnProviderSelected(evt.newValue));
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
                    _currentJob = null;
                }
                RefreshActiveTab();
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
            // Map-strength changes invalidate the cached map preview so the active tab refreshes.
            if (_normalStrengthSlider != null)
            {
                _normalStrengthSlider.RegisterValueChangedCallback(evt =>
                {
                    if (_normalPreview != null) { DestroyImmediate(_normalPreview); _normalPreview = null; }
                    if (_activeTab == PreviewTab.Normal) RefreshActiveTab();
                });
            }
            if (_aoStrengthSlider != null)
            {
                _aoStrengthSlider.RegisterValueChangedCallback(evt =>
                {
                    if (_aoPreview != null) { DestroyImmediate(_aoPreview); _aoPreview = null; }
                    if (_activeTab == PreviewTab.AO) RefreshActiveTab();
                });
            }

            // Tab navigation callbacks
            if (_tabSource != null) _tabSource.clicked += () => SetPreviewTab(PreviewTab.Source);
            if (_tabHeight != null) _tabHeight.clicked += () => SetPreviewTab(PreviewTab.Height);
            if (_tabNormal != null) _tabNormal.clicked += () => SetPreviewTab(PreviewTab.Normal);
            if (_tabRoughness != null) _tabRoughness.clicked += () => SetPreviewTab(PreviewTab.Roughness);
            if (_tabAO != null) _tabAO.clicked += () => SetPreviewTab(PreviewTab.AO);
            if (_tab3D != null) _tab3D.clicked += () => SetPreviewTab(PreviewTab.ThreeD);

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
                    var label = (Label)e;
                    string name = job.source != null ? job.source.name : "(missing)";
                    string status = job.state.ToString();
                    if (job.state == CPritch.DepthForge.Editor.Data.JobState.Error && !string.IsNullOrEmpty(job.error))
                    {
                        status += $": {job.error}";
                    }
                    label.text = $"{name}  —  {status}";

                    switch (job.state)
                    {
                        case CPritch.DepthForge.Editor.Data.JobState.Exported:
                            label.style.color = new Color(0.45f, 0.78f, 0.42f); break;
                        case CPritch.DepthForge.Editor.Data.JobState.Error:
                            label.style.color = new Color(0.85f, 0.42f, 0.42f); break;
                        case CPritch.DepthForge.Editor.Data.JobState.Generating:
                            label.style.color = new Color(0.85f, 0.70f, 0.35f); break;
                        default:
                            label.style.color = StyleKeyword.Null; break;
                    }
                };
                _queueListView.style.minHeight = 80;
                _queueListView.selectionType = SelectionType.Single;
                _queueListView.selectionChanged += OnQueueSelectionChanged;
            }
            if (_addToQueueButton != null) _addToQueueButton.clicked += AddSelectionToQueue;
            if (_clearQueueButton != null) _clearQueueButton.clicked += ClearQueue;
            if (_batchGenerateButton != null) _batchGenerateButton.clicked += StartBatch;
            UpdateBatchStatus();

            // Presets (R6)
            _presetDropdown = root.Q<DropdownField>("presetDropdown");
            _presetNameField = root.Q<TextField>("presetNameField");
            _presetDirtyMarker = root.Q<Label>("presetDirtyMarker");
            _savePresetButton = root.Q<Button>("savePresetButton");
            _saveAsPresetButton = root.Q<Button>("saveAsPresetButton");
            _deletePresetButton = root.Q<Button>("deletePresetButton");

            SetButtonIcon(_savePresetButton, "SaveActive", "Overwrite this preset with the current settings");
            SetButtonIcon(_saveAsPresetButton, "SaveAs", "Save the current settings as a new preset");
            SetButtonIcon(_deletePresetButton, "TreeEditor.Trash", "Delete this preset");

            if (_presetDropdown != null)
                _presetDropdown.RegisterValueChangedCallback(evt => OnPresetSelected(evt.newValue));
            if (_savePresetButton != null) _savePresetButton.clicked += OverwritePreset;
            if (_saveAsPresetButton != null) _saveAsPresetButton.clicked += SaveAsNewPreset;
            if (_deletePresetButton != null) _deletePresetButton.clicked += DeletePreset;

            // Any change to a tuning control re-evaluates the dirty marker against the selected preset.
            _invertToggle?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _contrastSlider?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _midpointSlider?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _flattenSlider?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _exportNormalToggle?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _normalStrengthSlider?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _exportAOToggle?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _aoStrengthSlider?.RegisterValueChangedCallback(_ => RecomputePresetDirty());
            _outputFormatField?.RegisterValueChangedCallback(_ => RecomputePresetDirty());

            RefreshPresetDropdown(CPritch.DepthForge.Editor.Data.PresetStore.DefaultName);

            // Responsive: stack the three zones vertically when the window is too narrow for columns.
            var dfMain = root.Q<VisualElement>("dfMain");
            if (dfMain != null)
            {
                dfMain.RegisterCallback<GeometryChangedEvent>(evt =>
                {
                    float w = evt.newRect.width;
                    bool narrow = w > 0f && w < 600f;            // stack into one column
                    bool medium = w >= 600f && w < 820f;         // compressed three columns
                    if (dfMain.ClassListContains("df-narrow") != narrow) dfMain.EnableInClassList("df-narrow", narrow);
                    if (dfMain.ClassListContains("df-medium") != medium) dfMain.EnableInClassList("df-medium", medium);
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

            UpdateProviderDependentUI();
        }

        // ---- Providers (R11) ----------------------------------------------------------------

        private void OnProviderSelected(string displayName)
        {
            if (_providers == null) return;
            foreach (var p in _providers)
            {
                if (p.Info.displayName == displayName) { _provider = p; break; }
            }
            UpdateProviderDependentUI();
        }

        /// <summary>Shows/hides controls that only make sense for certain providers: the Roughness
        /// preview tab + export toggle (native-roughness providers only), and the Depth-Anything-only
        /// tiling/letterbox/model-size controls.</summary>
        private void UpdateProviderDependentUI()
        {
            bool hasRoughness = _provider != null && (_provider.NativeMaps & MapKinds.Roughness) != 0;
            bool isDepthStyle = _provider != null && _provider.NativeMaps == MapKinds.Height; // Depth Anything

            ShowElement(_tabRoughness, hasRoughness);
            ShowRow(_exportRoughnessToggle, hasRoughness);

            // DA-specific generation controls; the material model handles sizing/tiling internally.
            ShowRow(_modelSizeField, isDepthStyle);
            ShowRow(_tiledInferenceToggle, isDepthStyle);
            ShowRow(_letterboxToggle, isDepthStyle);
            ShowRow(_tilingAlignmentField, isDepthStyle);

            // If the hidden Roughness tab was active, fall back to Height.
            if (!hasRoughness && _activeTab == PreviewTab.Roughness) SetPreviewTab(PreviewTab.Height);
        }

        private static void ShowElement(VisualElement el, bool show)
        {
            if (el != null) el.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
        }

        // EnumField/Toggle rows carry their label on the element itself, so toggle the element directly.
        private static void ShowRow(VisualElement el, bool show) => ShowElement(el, show);

        private void SetActionButtonsEnabled(bool enabled)
        {
            if (_saveButton != null) _saveButton.SetEnabled(enabled);
            if (_previewMeshButton != null) _previewMeshButton.SetEnabled(enabled);
        }

        private void SetPreviewTab(PreviewTab tab)
        {
            _activeTab = tab;

            _tabSource?.EnableInClassList("active-tab", tab == PreviewTab.Source);
            _tabHeight?.EnableInClassList("active-tab", tab == PreviewTab.Height);
            _tabNormal?.EnableInClassList("active-tab", tab == PreviewTab.Normal);
            _tabRoughness?.EnableInClassList("active-tab", tab == PreviewTab.Roughness);
            _tabAO?.EnableInClassList("active-tab", tab == PreviewTab.AO);
            _tab3D?.EnableInClassList("active-tab", tab == PreviewTab.ThreeD);

            if (tab == PreviewTab.ThreeD)
            {
                _outputPreview2D?.AddToClassList("hidden");
                _outputPreview3D?.RemoveFromClassList("hidden");
                _previewControls3D?.RemoveFromClassList("hidden");
            }
            else
            {
                _outputPreview2D?.RemoveFromClassList("hidden");
                _outputPreview3D?.AddToClassList("hidden");
                _previewControls3D?.AddToClassList("hidden");
                RefreshActiveTab();
            }
            Repaint();
        }

        /// <summary>Sets the 2D preview image to match the active map tab, generating map previews lazily.</summary>
        private void RefreshActiveTab()
        {
            if (_activeTab == PreviewTab.ThreeD || _outputPreview2D == null) return;

            Texture2D tex = null;
            switch (_activeTab)
            {
                case PreviewTab.Source: tex = _inputTextureField?.value as Texture2D; break;
                case PreviewTab.Height: tex = _adjustedHeightmap; break;
                case PreviewTab.Normal: tex = EnsureNormalPreview(); break;
                case PreviewTab.Roughness: tex = _nativeRoughness; break;
                case PreviewTab.AO: tex = EnsureAOPreview(); break;
            }
            _outputPreview2D.style.backgroundImage = tex;
        }

        private Texture2D EnsureNormalPreview()
        {
            // Prefer the provider's native normal (independent of the height tuning sliders); only
            // depth-style providers (Depth Anything) fall through to a derived-from-height normal.
            if (_nativeNormal != null) return _nativeNormal;
            if (_adjustedHeightmap == null) return null;
            if (_normalPreview == null)
            {
                float strength = _normalStrengthSlider != null ? _normalStrengthSlider.value : 8f;
                _normalPreview = ImageProcessor.GenerateNormalMap(_adjustedHeightmap, strength);
            }
            return _normalPreview;
        }

        private Texture2D EnsureAOPreview()
        {
            if (_adjustedHeightmap == null) return null;
            if (_aoPreview == null)
            {
                float strength = _aoStrengthSlider != null ? _aoStrengthSlider.value : 1f;
                float radius = _currentJob != null ? _currentJob.recipe.aoRadius : 0.02f;
                _aoPreview = ImageProcessor.GenerateAO(_adjustedHeightmap, strength, radius);
            }
            return _aoPreview;
        }

        private void InvalidateMapPreviews()
        {
            if (_normalPreview != null) { DestroyImmediate(_normalPreview); _normalPreview = null; }
            if (_aoPreview != null) { DestroyImmediate(_aoPreview); _aoPreview = null; }
        }

        private void OnGenerateClicked()
        {
            Texture2D inputTex = _inputTextureField.value as Texture2D;

            if (inputTex == null)
            {
                EditorUtility.DisplayDialog("Missing references", "Please assign a Base Map texture.", "OK");
                return;
            }
            if (_provider == null)
            {
                EditorUtility.DisplayDialog("No model", "No inference provider is available.", "OK");
                return;
            }

            CPritch.DepthForge.Editor.Data.Recipe recipe = BuildRecipeFromUI();

            // The provider owns availability + (its own) download. Drive its progress through our
            // non-modal bar, then run inference once it reports ready.
            if (_generateButton != null) _generateButton.SetEnabled(false);
            if (_progressBar != null)
            {
                _progressBar.RemoveFromClassList("hidden");
                _progressBar.value = 0f;
                _progressBar.title = "Preparing model...";
            }

            _provider.PrepareAsync(recipe,
                (p, status) =>
                {
                    if (_progressBar != null) { _progressBar.value = p; _progressBar.title = status; }
                    if (_generateButton != null) _generateButton.text = status;
                },
                () => RunInference(inputTex, recipe),
                error =>
                {
                    EndGeneratingUI();
                    EditorUtility.DisplayDialog("Model Unavailable", error, "OK");
                });
        }

        private void RunInference(Texture2D inputTex, CPritch.DepthForge.Editor.Data.Recipe recipe)
        {
            try
            {
                _provider.Initialize(recipe);
            }
            catch (Exception ex)
            {
                EndGeneratingUI();
                Debug.LogError($"Inference init failed: {ex.Message}");
                EditorUtility.DisplayDialog("Generation Error", $"Failed to initialize inference:\n{ex.Message}", "OK");
                return;
            }

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
                _progressBar.title = "Running inference...";
            }

            _provider.ExecuteAsync(inputTex, recipe,
                progress =>
                {
                    if (_progressBar != null) _progressBar.value = progress;
                },
                maps =>
                {
                    EndGeneratingUI();
                    OnInferenceDone(maps);
                    Repaint();
                },
                error =>
                {
                    EndGeneratingUI();
                    Debug.LogError($"Map generation failed: {error}");
                    EditorUtility.DisplayDialog("Generation Error", $"An error occurred during generation:\n{error}", "OK");
                    Repaint();
                });
        }

        /// <summary>Takes ownership of a provider's native <see cref="MapSet"/> and refreshes the workspace.</summary>
        private void OnInferenceDone(MapSet maps)
        {
            if (maps == null) return;

            // Height is always native (DA emits it; DepthForge bakes integration into the export).
            if (_rawHeightmap != null) DestroyImmediate(_rawHeightmap);
            _rawHeightmap = maps.height;

            // Adopt any native normal/roughness; drop stale derived previews.
            if (_nativeNormal != null) { DestroyImmediate(_nativeNormal); _nativeNormal = null; }
            if (_nativeRoughness != null) { DestroyImmediate(_nativeRoughness); _nativeRoughness = null; }
            if (maps.Has(MapKinds.Normal)) _nativeNormal = maps.normal;
            if (maps.Has(MapKinds.Roughness)) _nativeRoughness = maps.roughness;

            UpdateAdjustedHeightmap();
            SetActionButtonsEnabled(true);
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

        private void OnQueueSelectionChanged(IEnumerable<object> selected)
        {
            foreach (var o in selected)
            {
                if (o is CPritch.DepthForge.Editor.Data.Job job)
                {
                    FocusJob(job);
                    break;
                }
            }
        }

        /// <summary>
        /// Makes a queue Job the active focus: its source fills the Base Map picker, its recipe
        /// populates the inspector, and its cached raw restores the preview — all without re-running
        /// inference. The previously focused job keeps its in-progress UI edits (in memory).
        /// </summary>
        private void FocusJob(CPritch.DepthForge.Editor.Data.Job job)
        {
            if (job == null || job.source == null) return;
            if (_currentJob == job) return;

            // Preserve the outgoing job's in-flight UI edits before switching away.
            if (_currentJob != null)
            {
                _currentJob.recipe = BuildRecipeFromUI();
            }

            // Drop the current working heightmaps; the new focus loads its own. Native model maps
            // aren't cached, so a revisit re-derives normal/AO from the cached height (DA-style).
            if (_rawHeightmap != null) { DestroyImmediate(_rawHeightmap); _rawHeightmap = null; }
            if (_adjustedHeightmap != null) { DestroyImmediate(_adjustedHeightmap); _adjustedHeightmap = null; }
            if (_nativeNormal != null) { DestroyImmediate(_nativeNormal); _nativeNormal = null; }
            if (_nativeRoughness != null) { DestroyImmediate(_nativeRoughness); _nativeRoughness = null; }
            InvalidateMapPreviews();

            _currentJob = job;

            // Reflect the source in the picker + input preview without re-triggering the field reload.
            _inputTextureField?.SetValueWithoutNotify(job.source);

            ApplyRecipeToUI(job.recipe);

            // Restore the cached raw depth so the preview shows the result without re-inference.
            var cachedRaw = CPritch.DepthForge.Editor.Data.RawCache.LoadRaw(job.source);
            if (cachedRaw != null)
            {
                _rawHeightmap = cachedRaw;
                UpdateAdjustedHeightmap();
                SetActionButtonsEnabled(true);
            }
            else
            {
                SetActionButtonsEnabled(false);
                RefreshActiveTab();
            }

            UpdatePreviewMaterial();
            Repaint();
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

            if (_provider == null)
            {
                EditorUtility.DisplayDialog("Batch", "No inference provider is available.", "OK");
                return;
            }
            if (!_provider.IsAvailable(out string reason))
            {
                EditorUtility.DisplayDialog("Batch", reason, "OK");
                return;
            }

            // One provider + one model load for the whole batch; each job's own recipe drives its
            // tuning/maps. The representative recipe (current UI) selects the model + backend.
            var initRecipe = BuildRecipeFromUI();

            _batchGenerateButton?.SetEnabled(false);
            _generateButton?.SetEnabled(false);

            _batch = new BatchProcessor(_provider, _queue, initRecipe,
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
                    // R10: assign every exported job's maps to MicroSplat in one pass (compile once per config).
                    TryAssignBatchToMicroSplat();
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

        // ---- Presets (R6) -------------------------------------------------------------------
        // Presets carry only the tuning half of a recipe (adjustments + maps + format); the
        // generation half (model/backend/tiling/letterbox) is left as the focused job has it, so a
        // "Rock" look applies no matter which model is downloaded.

        private void SetButtonIcon(Button button, string iconName, string tooltip)
        {
            if (button == null) return;
            button.text = string.Empty;
            button.tooltip = tooltip;

            // Prefer the dark-skin ("d_") variant when appropriate; fall back to the base icon.
            GUIContent content = null;
            if (EditorGUIUtility.isProSkin) content = EditorGUIUtility.IconContent("d_" + iconName);
            if (content == null || content.image == null) content = EditorGUIUtility.IconContent(iconName);
            if (content?.image is Texture2D tex)
            {
                button.style.backgroundImage = new StyleBackground(tex);
            }
        }

        private void RefreshPresetDropdown(string selectName)
        {
            if (_presetDropdown == null) return;
            var all = CPritch.DepthForge.Editor.Data.PresetStore.GetAll();
            var names = new List<string>(all.Count);
            foreach (var p in all) names.Add(p.name);
            _presetDropdown.choices = names;

            string sel = (!string.IsNullOrEmpty(selectName) && names.Contains(selectName)) ? selectName
                       : (names.Count > 0 ? names[0] : null);
            _presetDropdown.SetValueWithoutNotify(sel);
            _selectedPresetName = sel;

            UpdateDeletePresetButton();
            RecomputePresetDirty();
        }

        private void UpdateDeletePresetButton()
        {
            if (_deletePresetButton == null) return;
            bool deletable = !string.IsNullOrEmpty(_selectedPresetName)
                && !CPritch.DepthForge.Editor.Data.PresetStore.IsBuiltin(_selectedPresetName)
                && CPritch.DepthForge.Editor.Data.PresetStore.Find(_selectedPresetName) != null;
            _deletePresetButton.SetEnabled(deletable);
        }

        private void OnPresetSelected(string name)
        {
            var preset = CPritch.DepthForge.Editor.Data.PresetStore.Find(name);
            if (preset == null) return;
            _selectedPresetName = name;
            ApplyPreset(preset);
            SetPresetDirty(false); // just applied — matches the preset
            UpdateDeletePresetButton();
        }

        private void ApplyPreset(CPritch.DepthForge.Editor.Data.Preset preset)
        {
            if (preset?.recipe == null) return;
            var r = preset.recipe;

            // Overlay only the tuning fields — generation settings stay as the user set them.
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

            // aoRadius has no UI control — carry it onto the focused job so it survives export.
            if (_currentJob != null) _currentJob.recipe.aoRadius = r.aoRadius;

            // Re-derive the adjusted preview (which also invalidates cached Normal/AO maps).
            if (_rawHeightmap != null) UpdateAdjustedHeightmap();
            else { InvalidateMapPreviews(); RefreshActiveTab(); }
            Repaint();
        }

        /// <summary>Compares the live tuning UI against the selected preset and shows/hides the * marker.</summary>
        private void RecomputePresetDirty()
        {
            var preset = CPritch.DepthForge.Editor.Data.PresetStore.Find(_selectedPresetName);
            bool dirty = preset != null && !UiTuningMatches(preset.recipe);
            SetPresetDirty(dirty);
        }

        private void SetPresetDirty(bool dirty)
        {
            if (_presetDirtyMarker != null) _presetDirtyMarker.EnableInClassList("hidden", !dirty);

            // Overwrite only makes sense for a dirty *user* preset; built-ins are read-only.
            bool isUserPreset = !string.IsNullOrEmpty(_selectedPresetName)
                && !CPritch.DepthForge.Editor.Data.PresetStore.IsBuiltin(_selectedPresetName)
                && CPritch.DepthForge.Editor.Data.PresetStore.Find(_selectedPresetName) != null;
            _savePresetButton?.SetEnabled(dirty && isUserPreset);
        }

        private bool UiTuningMatches(CPritch.DepthForge.Editor.Data.Recipe r)
        {
            if (r == null) return false;
            const float eps = 0.0001f;
            if (_invertToggle != null && _invertToggle.value != r.invert) return false;
            if (_contrastSlider != null && Mathf.Abs(_contrastSlider.value - r.contrast) > eps) return false;
            if (_midpointSlider != null && Mathf.Abs(_midpointSlider.value - r.midpoint) > eps) return false;
            if (_flattenSlider != null && Mathf.Abs(_flattenSlider.value - r.flatten) > eps) return false;
            if (_exportNormalToggle != null && _exportNormalToggle.value != r.exportNormal) return false;
            if (_normalStrengthSlider != null && Mathf.Abs(_normalStrengthSlider.value - r.normalStrength) > eps) return false;
            if (_exportAOToggle != null && _exportAOToggle.value != r.exportAO) return false;
            if (_aoStrengthSlider != null && Mathf.Abs(_aoStrengthSlider.value - r.aoStrength) > eps) return false;
            if (_outputFormatField != null &&
                (CPritch.DepthForge.Editor.Utils.TextureExporter.ExportFormat)_outputFormatField.value != r.format) return false;
            return true;
        }

        /// <summary>Save icon — overwrites the currently selected (user) preset with the live settings.</summary>
        private void OverwritePreset()
        {
            if (string.IsNullOrEmpty(_selectedPresetName) || CPritch.DepthForge.Editor.Data.PresetStore.IsBuiltin(_selectedPresetName))
            {
                EditorUtility.DisplayDialog("Save Preset",
                    "Built-in presets can't be overwritten. Use Save As (+) to create your own.", "OK");
                return;
            }
            var recipe = BuildRecipeFromUI();
            CPritch.DepthForge.Editor.Data.PresetStore.SaveUserPreset(_selectedPresetName, recipe);
            RefreshPresetDropdown(_selectedPresetName); // recompute → no longer dirty
        }

        /// <summary>Save-as (+) icon — saves the live settings as a new named user preset.</summary>
        private void SaveAsNewPreset()
        {
            string name = _presetNameField != null ? _presetNameField.value : null;
            if (string.IsNullOrWhiteSpace(name))
            {
                EditorUtility.DisplayDialog("Save As New Preset", "Enter a name in the Save As field first.", "OK");
                return;
            }
            name = name.Trim();
            if (CPritch.DepthForge.Editor.Data.PresetStore.IsBuiltin(name))
            {
                EditorUtility.DisplayDialog("Save As New Preset",
                    $"'{name}' is a built-in preset name. Choose a different name.", "OK");
                return;
            }
            if (CPritch.DepthForge.Editor.Data.PresetStore.Find(name) != null &&
                !EditorUtility.DisplayDialog("Save As New Preset",
                    $"A preset named '{name}' already exists. Overwrite it?", "Overwrite", "Cancel"))
            {
                return;
            }

            var recipe = BuildRecipeFromUI();
            if (!CPritch.DepthForge.Editor.Data.PresetStore.SaveUserPreset(name, recipe))
            {
                EditorUtility.DisplayDialog("Save As New Preset", "Could not save the preset.", "OK");
                return;
            }
            _presetNameField?.SetValueWithoutNotify(string.Empty);
            RefreshPresetDropdown(name);
        }

        private void DeletePreset()
        {
            if (string.IsNullOrEmpty(_selectedPresetName) || CPritch.DepthForge.Editor.Data.PresetStore.IsBuiltin(_selectedPresetName))
            {
                EditorUtility.DisplayDialog("Delete Preset",
                    "Built-in presets can't be deleted. Select a saved preset to delete.", "OK");
                return;
            }
            if (!EditorUtility.DisplayDialog("Delete Preset", $"Delete the preset '{_selectedPresetName}'?", "Delete", "Cancel")) return;

            CPritch.DepthForge.Editor.Data.PresetStore.DeleteUserPreset(_selectedPresetName);
            RefreshPresetDropdown(CPritch.DepthForge.Editor.Data.PresetStore.DefaultName);
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

            // The adjusted heightmap changed, so cached Normal/AO previews are stale.
            InvalidateMapPreviews();
            RefreshActiveTab();

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
            // Start from the focused job's recipe so fields without a UI control (e.g. aoRadius,
            // carried by presets) survive the round-trip; UI values overlay on top.
            var r = _currentJob != null ? _currentJob.recipe.Clone() : new CPritch.DepthForge.Editor.Data.Recipe();
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
            if (_exportRoughnessToggle != null) r.exportRoughness = _exportRoughnessToggle.value;
            if (_autoAssignToggle != null) r.autoAssignMicroSplat = _autoAssignToggle.value;
            r.providerId = _provider != null ? _provider.Info.id : r.providerId;
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
            if (_exportRoughnessToggle != null) _exportRoughnessToggle.SetValueWithoutNotify(r.exportRoughness);

            // Reflect the recipe's provider (if it names one that's available in this build).
            if (!string.IsNullOrEmpty(r.providerId) && _providers != null)
            {
                var p = ProviderRegistry.ById(_providers, r.providerId);
                if (p != null)
                {
                    _provider = p;
                    _providerField?.SetValueWithoutNotify(p.Info.displayName);
                    UpdateProviderDependentUI();
                }
            }

            // Refresh the derived preview only if a heightmap already exists for this source.
            if (_rawHeightmap != null) UpdateAdjustedHeightmap();
            // The loaded recipe may diverge from the selected preset — reflect that in the * marker.
            RecomputePresetDirty();
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
                var savedRecipe = BuildRecipeFromUI();
                var maps = CurrentMapSet();

                // Normal: use the provider's native normal when present (better than synthesised);
                // otherwise derive one from the adjusted height (the Depth Anything path).
                string normalPath = null;
                if (savedRecipe.exportNormal)
                {
                    bool derived;
                    Texture2D normalTex = MapPipeline.BuildNormal(maps, savedRecipe, _adjustedHeightmap, out derived);
                    if (normalTex != null)
                    {
                        normalPath = CPritch.DepthForge.Editor.Utils.TextureExporter.SaveNormalMap(normalTex, inputTex);
                        if (derived) DestroyImmediate(normalTex);
                    }
                }

                // Roughness: native only (absent for depth-style providers).
                string roughnessPath = null;
                Texture2D nativeRough = MapPipeline.NativeRoughness(maps);
                if (savedRecipe.exportRoughness && nativeRough != null)
                {
                    roughnessPath = CPritch.DepthForge.Editor.Utils.TextureExporter.SaveRoughness(nativeRough, inputTex);
                }

                // AO is always derived from the adjusted height.
                string aoPath = null;
                if (savedRecipe.exportAO)
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
                    AssignToMicroSplat(new List<MicroSplatAssignment>
                    {
                        new MicroSplatAssignment { diffuse = inputTex, heightPath = path, normalPath = normalPath }
                    });
                }

                // Phase 1 (R5): persist the recipe sidecar so this depth work is re-editable later.
                if (_currentJob == null) _currentJob = new CPritch.DepthForge.Editor.Data.Job(inputTex);
                _currentJob.recipe = savedRecipe;
                _currentJob.state = CPritch.DepthForge.Editor.Data.JobState.Exported;
                _currentJob.heightmapPath = path;
                _currentJob.normalPath = normalPath;
                _currentJob.roughnessPath = roughnessPath;
                CPritch.DepthForge.Editor.Data.RecipeSidecar.Save(inputTex, savedRecipe);
                // Cache the raw (pre-adjustment) depth so revisiting this source restores an
                // editable result without re-running inference.
                CPritch.DepthForge.Editor.Data.RawCache.SaveRaw(inputTex, _rawHeightmap);

                string message = $"Heightmap exported to:\n{path}";
                if (normalPath != null) message += $"\n\nNormal map:\n{normalPath}";
                if (roughnessPath != null) message += $"\n\nRoughness map:\n{roughnessPath}";
                if (aoPath != null) message += $"\n\nAO map:\n{aoPath}";
                EditorUtility.DisplayDialog("Success", message, "Awesome");
            }
        }

        /// <summary>A MapSet view over the focused source's current working maps (native height +
        /// any native normal/roughness), for MapPipeline-driven export.</summary>
        private MapSet CurrentMapSet()
        {
            var native = MapKinds.Height;
            if (_nativeNormal != null) native |= MapKinds.Normal;
            if (_nativeRoughness != null) native |= MapKinds.Roughness;
            return new MapSet { height = _rawHeightmap, normal = _nativeNormal, roughness = _nativeRoughness, native = native };
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

        // One diffuse → its freshly-exported height/normal asset paths. Batching these lets us assign a
        // whole queue and compile each MicroSplat config only once (R10 "assign-all").
        private struct MicroSplatAssignment
        {
            public Texture2D diffuse;
            public string heightPath;
            public string normalPath;
        }

        /// <summary>After a batch export, push every exported job's maps into MicroSplat in one pass.</summary>
        private void TryAssignBatchToMicroSplat()
        {
            if (!IsMicroSplatPresent()) return;

            var assignments = new List<MicroSplatAssignment>();
            foreach (var job in _queue)
            {
                if (job == null || job.source == null) continue;
                if (job.state != CPritch.DepthForge.Editor.Data.JobState.Exported) continue;
                if (!job.recipe.autoAssignMicroSplat) continue;
                if (string.IsNullOrEmpty(job.heightmapPath)) continue;
                assignments.Add(new MicroSplatAssignment
                {
                    diffuse = job.source,
                    heightPath = job.heightmapPath,
                    normalPath = job.normalPath
                });
            }

            if (assignments.Count > 0) AssignToMicroSplat(assignments);
        }

        /// <summary>
        /// Assigns one-or-many (diffuse → height/normal) sets into every matching MicroSplat
        /// TextureArrayConfig entry, then saves + compiles each touched config exactly once.
        /// </summary>
        private void AssignToMicroSplat(List<MicroSplatAssignment> assignments)
        {
            if (assignments == null || assignments.Count == 0) return;

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

            var texCache = new Dictionary<string, Texture2D>();
            Texture2D LoadTex(string p)
            {
                if (string.IsNullOrEmpty(p)) return null;
                if (!texCache.TryGetValue(p, out var t)) { t = AssetDatabase.LoadAssetAtPath<Texture2D>(p); texCache[p] = t; }
                return t;
            }

            var changedConfigs = new List<ScriptableObject>();
            Type changedConfigType = null;
            int assignedCount = 0;

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
                    if (diffuseTex == null) continue;

                    // Find the assignment whose diffuse matches this config entry.
                    int matchIdx = assignments.FindIndex(a => a.diffuse == diffuseTex);
                    if (matchIdx < 0) continue;
                    var match = assignments[matchIdx];

                    Texture2D heightTex = LoadTex(match.heightPath);
                    if (heightTex != null)
                    {
                        var heightField = entryType.GetField("height");
                        if (heightField != null)
                        {
                            heightField.SetValue(entry, heightTex);
                            configChanged = true;
                            assignedCount++;
                            Debug.Log($"[DepthForge] Assigned heightmap to MicroSplat '{config.name}' for '{diffuseTex.name}'");
                        }
                    }

                    Texture2D normalTex = LoadTex(match.normalPath);
                    if (normalTex != null)
                    {
                        var normalField = entryType.GetField("normal");
                        if (normalField != null)
                        {
                            normalField.SetValue(entry, normalTex);
                            configChanged = true;
                            Debug.Log($"[DepthForge] Assigned normal map to MicroSplat '{config.name}' for '{diffuseTex.name}'");
                        }
                    }
                }

                if (configChanged)
                {
                    changedConfigs.Add(config);
                    changedConfigType = configType;
                }
            }

            if (changedConfigs.Count == 0)
            {
                Debug.LogWarning("[DepthForge] No MicroSplat config entry matched the exported diffuse texture(s).");
                return;
            }

            // Save all touched configs, then compile each once (compile is the expensive part).
            foreach (var cfg in changedConfigs) EditorUtility.SetDirty(cfg);
            AssetDatabase.SaveAssets();

            Type editorType = FindTypeInAssemblies("JBooth.MicroSplat.TextureArrayConfigEditor");
            if (editorType != null && changedConfigType != null)
            {
                var compileMethod = editorType.GetMethod("CompileConfig",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static,
                    null, new Type[] { changedConfigType }, null);
                if (compileMethod != null)
                {
                    foreach (var cfg in changedConfigs)
                    {
                        Debug.Log($"[DepthForge] Compiling MicroSplat Texture Array Config: {cfg.name}");
                        compileMethod.Invoke(null, new object[] { cfg });
                    }
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

            Debug.Log($"[DepthForge] Auto-assigned maps to MicroSplat for {assignedCount} entr{(assignedCount == 1 ? "y" : "ies")} across {changedConfigs.Count} config(s).");
        }
    }
}
