// =============================================================================
// THROWAWAY SPIKE — DepthForge R3 (non-blocking inference) feasibility test.
// Purpose: prove the editor stays responsive while inference runs, using
// Inference Engine's ScheduleIterable (budgeted, cooperative scheduling) plus
// async readback (ReadbackRequest / IsReadbackRequestDone), versus the current
// blocking Schedule + DownloadToArray path.
//
// DELETE this file (and the Spikes folder) once R3 is validated — it is not part
// of the shipping product.
//
// How to run:
//   1. Tools ▸ DepthForge ▸ Spikes ▸ Async Inference Spike
//   2. Assign an imported ONNX depth ModelAsset + any readable Texture2D.
//   3. Click "Run BLOCKING" — watch the Heartbeat freeze; note "Worst frame gap".
//      Try dragging another editor window during the run: it will stall.
//   4. Click "Run ASYNC" — Heartbeat keeps ticking; worst frame gap stays small;
//      other windows stay interactive. Output min/max should match the blocking run.
//
// If an API name differs in your exact 2.2.x patch, the only three calls to adjust
// are: worker.ScheduleIterable(...), tensor.ReadbackRequest(), tensor.IsReadbackRequestDone().
// =============================================================================

using System;
using System.Collections;
using System.Diagnostics;
using UnityEditor;
using UnityEngine;
using Unity.InferenceEngine;
using CPritch.DepthForge.Editor.Inference;
using Debug = UnityEngine.Debug;

namespace CPritch.DepthForge.Editor.Spikes
{
    public class AsyncInferenceSpike : EditorWindow
    {
        private ModelAsset _modelAsset;
        private Texture2D _inputTexture;
        private BackendType _backend = BackendType.GPUCompute;
        private const int T = 518;
        private const double PumpBudgetMs = 6.0; // per-tick compute budget for the cooperative pump

        // Heartbeat / responsiveness instrumentation (driven by EditorApplication.update).
        private long _heartbeat;
        private double _lastTickTime;
        private double _worstFrameGapMs;
        private bool _measuring;

        // Async run state machine.
        private enum AsyncPhase { Idle, Scheduling, AwaitingReadback }
        private AsyncPhase _phase = AsyncPhase.Idle;
        private Worker _worker;
        private Tensor<float> _input;
        private IEnumerator _schedule;
        private Tensor<float> _pendingOutput;
        private Stopwatch _runTimer;
        private int _inputRank = 4;

        private string _lastResult = "—";

        [MenuItem("Tools/DepthForge/Spikes/Async Inference Spike")]
        public static void Open()
        {
            var wnd = GetWindow<AsyncInferenceSpike>();
            wnd.titleContent = new GUIContent("R3 Async Spike");
            wnd.minSize = new Vector2(380, 320);
        }

        private void OnEnable()
        {
            _lastTickTime = EditorApplication.timeSinceStartup;
            EditorApplication.update += Tick;
        }

        private void OnDisable()
        {
            EditorApplication.update -= Tick;
            CleanupRun();
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("R3 — Non-blocking inference spike", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Compare a blocking run vs a cooperative async run. Watch the Heartbeat and the " +
                "'Worst frame gap' — a frozen heartbeat / large gap means the editor stalled.",
                MessageType.None);

            _modelAsset = (ModelAsset)EditorGUILayout.ObjectField("Depth ModelAsset", _modelAsset, typeof(ModelAsset), false);
            _inputTexture = (Texture2D)EditorGUILayout.ObjectField("Input Texture", _inputTexture, typeof(Texture2D), false);
            _backend = (BackendType)EditorGUILayout.EnumPopup("Backend", _backend);

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(_phase != AsyncPhase.Idle))
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Run BLOCKING (baseline)", GUILayout.Height(30))) RunBlocking();
                    if (GUILayout.Button("Run ASYNC (ScheduleIterable)", GUILayout.Height(30))) StartAsync();
                }
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Heartbeat (ticks while idle/async, freezes while blocking):", _heartbeat.ToString());
            EditorGUILayout.LabelField("Worst frame gap during last run:", $"{_worstFrameGapMs:F1} ms");
            EditorGUILayout.LabelField("Phase:", _phase.ToString());
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Last result:", EditorStyles.miniBoldLabel);
            EditorGUILayout.SelectableLabel(_lastResult, EditorStyles.wordWrappedMiniLabel, GUILayout.Height(48));
        }

        // ---- Responsiveness pump + async state machine -------------------------------------

        private void Tick()
        {
            double now = EditorApplication.timeSinceStartup;
            double gapMs = (now - _lastTickTime) * 1000.0;
            _lastTickTime = now;

            _heartbeat++;
            if (_measuring && gapMs > _worstFrameGapMs) _worstFrameGapMs = gapMs;
            Repaint(); // keep the heartbeat visibly ticking

            if (_phase == AsyncPhase.Scheduling) PumpSchedule();
            else if (_phase == AsyncPhase.AwaitingReadback) PollReadback();
        }

        private void PumpSchedule()
        {
            // Cooperative: advance the model a few layers per tick, bounded by a time budget,
            // so we never block the editor for a whole inference.
            var budget = Stopwatch.StartNew();
            try
            {
                while (budget.Elapsed.TotalMilliseconds < PumpBudgetMs)
                {
                    if (!_schedule.MoveNext())
                    {
                        // Compute scheduled; request a non-blocking GPU->CPU readback.
                        _pendingOutput = _worker.PeekOutput() as Tensor<float>;
                        _pendingOutput.ReadbackRequest();
                        _phase = AsyncPhase.AwaitingReadback;
                        return;
                    }
                }
            }
            catch (Exception ex)
            {
                FailRun($"ASYNC schedule failed: {ex.Message}");
            }
        }

        private void PollReadback()
        {
            try
            {
                if (!_pendingOutput.IsReadbackRequestDone()) return; // still copying; editor stays free

                float[] data = _pendingOutput.DownloadToArray(); // now instant — data already on CPU
                _runTimer.Stop();
                _measuring = false;
                ReportRun("ASYNC", data, _runTimer.Elapsed.TotalMilliseconds);
                CleanupRun();
                _phase = AsyncPhase.Idle;
            }
            catch (Exception ex)
            {
                FailRun($"ASYNC readback failed: {ex.Message}");
            }
        }

        // ---- Runs --------------------------------------------------------------------------

        private void StartAsync()
        {
            if (!Validate()) return;
            try
            {
                BeginMeasuring();
                var model = ModelLoader.Load(_modelAsset);
                _inputRank = (model.inputs != null && model.inputs.Count > 0) ? model.inputs[0].shape.rank : 4;
                _worker = new Worker(model, _backend);
                _input = ImageProcessor.PreprocessTexture(_inputTexture, T, T, _inputRank);
                _runTimer = Stopwatch.StartNew();
                _schedule = _worker.ScheduleIterable(_input);
                _phase = AsyncPhase.Scheduling;
                Debug.Log("[R3 Spike] ASYNC run started — try interacting with other editor windows now.");
            }
            catch (Exception ex)
            {
                FailRun($"ASYNC start failed: {ex.Message}");
            }
        }

        private void RunBlocking()
        {
            if (!Validate()) return;
            Worker worker = null;
            Tensor<float> input = null;
            try
            {
                BeginMeasuring();
                var model = ModelLoader.Load(_modelAsset);
                int rank = (model.inputs != null && model.inputs.Count > 0) ? model.inputs[0].shape.rank : 4;
                worker = new Worker(model, _backend);
                input = ImageProcessor.PreprocessTexture(_inputTexture, T, T, rank);

                var sw = Stopwatch.StartNew();
                worker.Schedule(input);                                   // dispatch
                var output = worker.PeekOutput() as Tensor<float>;
                float[] data = output.DownloadToArray();                  // BLOCKING sync readback
                sw.Stop();

                _measuring = false;
                ReportRun("BLOCKING", data, sw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                _measuring = false;
                FailRun($"BLOCKING failed: {ex.Message}");
            }
            finally
            {
                input?.Dispose();
                worker?.Dispose();
            }
        }

        // ---- Helpers -----------------------------------------------------------------------

        private bool Validate()
        {
            if (_modelAsset == null || _inputTexture == null)
            {
                _lastResult = "Assign both a ModelAsset and an input Texture2D first.";
                return false;
            }
            if (!_inputTexture.isReadable)
            {
                _lastResult = "Input texture is not readable — enable Read/Write in its import settings.";
                return false;
            }
            return true;
        }

        private void BeginMeasuring()
        {
            _worstFrameGapMs = 0.0;
            _lastTickTime = EditorApplication.timeSinceStartup;
            _measuring = true;
        }

        private void ReportRun(string mode, float[] data, double totalMs)
        {
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            _lastResult = $"{mode}: {totalMs:F0} ms total · worst frame gap {_worstFrameGapMs:F1} ms · " +
                          $"out {data.Length} vals · depth min {min:F3} / max {max:F3}";
            Debug.Log($"[R3 Spike] {_lastResult}");
        }

        private void FailRun(string msg)
        {
            _lastResult = msg;
            _measuring = false;
            Debug.LogError($"[R3 Spike] {msg}");
            CleanupRun();
            _phase = AsyncPhase.Idle;
        }

        private void CleanupRun()
        {
            _input?.Dispose();
            _input = null;
            _worker?.Dispose();
            _worker = null;
            _schedule = null;
            _pendingOutput = null;
            _runTimer = null;
        }
    }
}
