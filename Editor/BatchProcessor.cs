using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using Unity.InferenceEngine;
using CPritch.DepthForge.Editor.Data;
using CPritch.DepthForge.Editor.Inference;
using CPritch.DepthForge.Editor.Utils;

namespace CPritch.DepthForge.Editor
{
    /// <summary>
    /// Drives a queue of <see cref="Job"/>s through inference + adjustments + map export sequentially,
    /// reusing a single Worker (one model load for the whole batch — VRAM cost paid once). Non-blocking:
    /// each job's inference runs via the runner's async path, and jobs are chained across editor ticks
    /// so the editor stays responsive. The batch uses one model (chosen by the caller) for all jobs;
    /// each job's own recipe drives its adjustments, maps, and export.
    /// </summary>
    public class BatchProcessor
    {
        private readonly DepthInferenceRunner _runner;
        private readonly List<Job> _jobs;
        private readonly ModelAsset _model;
        private readonly BackendType _backend;
        private readonly Action<Job> _onJobChanged;
        private readonly Action<int, int> _onProgress;
        private readonly Action _onAllDone;

        private int _index;
        public bool IsRunning { get; private set; }

        public BatchProcessor(DepthInferenceRunner runner, List<Job> jobs, ModelAsset model, BackendType backend,
                              Action<Job> onJobChanged, Action<int, int> onProgress, Action onAllDone)
        {
            _runner = runner;
            _jobs = jobs;
            _model = model;
            _backend = backend;
            _onJobChanged = onJobChanged;
            _onProgress = onProgress;
            _onAllDone = onAllDone;
        }

        public void Start()
        {
            if (_jobs == null || _jobs.Count == 0) { _onAllDone?.Invoke(); return; }

            try
            {
                _runner.Initialize(_model, _backend); // load the model / Worker once for the whole batch
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DepthForge] Batch init failed: {ex.Message}");
                _onAllDone?.Invoke();
                return;
            }

            IsRunning = true;
            _index = 0;
            ProcessNext();
        }

        private void ProcessNext()
        {
            if (_index >= _jobs.Count)
            {
                IsRunning = false;
                _onAllDone?.Invoke();
                return;
            }

            Job job = _jobs[_index];
            if (job == null || job.source == null)
            {
                if (job != null) { job.state = JobState.Error; job.error = "Missing source."; }
                AdvanceAfter(job);
                return;
            }

            job.state = JobState.Generating;
            job.error = null;
            _onJobChanged?.Invoke(job);

            bool useTiling = job.recipe.tiledInference;
            var alignment = (DepthInferenceRunner.TilingAlignment)(int)job.recipe.tilingAlignment;
            bool letterbox = job.recipe.letterbox;

            _runner.ExecuteAsync(job.source, useTiling, alignment, letterbox,
                _ => { },
                raw => CompleteJob(job, raw),
                err => { job.state = JobState.Error; job.error = err; AdvanceAfter(job); });
        }

        private void CompleteJob(Job job, Texture2D raw)
        {
            try
            {
                if (raw == null) { job.state = JobState.Error; job.error = "No inference result."; AdvanceAfter(job); return; }

                Recipe r = job.recipe;
                Texture2D adjusted = ImageProcessor.ApplyAdjustments(raw, r.contrast, r.midpoint, r.invert, r.flatten);

                // Record export paths so the Window can do a single MicroSplat assign-all afterwards.
                job.normalPath = null;
                job.heightmapPath = TextureExporter.SaveHeightmap(adjusted, job.source, r.format);

                if (r.exportNormal)
                {
                    Texture2D n = ImageProcessor.GenerateNormalMap(adjusted, r.normalStrength);
                    if (n != null) { job.normalPath = TextureExporter.SaveNormalMap(n, job.source); UnityEngine.Object.DestroyImmediate(n); }
                }
                if (r.exportAO)
                {
                    Texture2D a = ImageProcessor.GenerateAO(adjusted, r.aoStrength, r.aoRadius);
                    if (a != null) { TextureExporter.SaveAOMap(a, job.source); UnityEngine.Object.DestroyImmediate(a); }
                }

                RecipeSidecar.Save(job.source, r);
                RawCache.SaveRaw(job.source, raw);
                job.state = JobState.Exported;

                UnityEngine.Object.DestroyImmediate(adjusted);
            }
            catch (Exception ex)
            {
                job.state = JobState.Error;
                job.error = ex.Message;
                Debug.LogError($"[DepthForge] Batch job '{(job.source != null ? job.source.name : "?")}' failed: {ex.Message}");
            }
            finally
            {
                if (raw != null) UnityEngine.Object.DestroyImmediate(raw);
            }

            AdvanceAfter(job);
        }

        private void AdvanceAfter(Job job)
        {
            if (job != null) _onJobChanged?.Invoke(job);
            _index++;
            _onProgress?.Invoke(_index, _jobs.Count);
            // Yield to the editor between jobs so the UI stays responsive.
            EditorApplication.delayCall += ProcessNext;
        }
    }
}
