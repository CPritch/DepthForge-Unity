using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using CPritch.DepthForge.Editor.Data;
using CPritch.DepthForge.Editor.Inference;
using CPritch.DepthForge.Editor.Utils;

namespace CPritch.DepthForge.Editor
{
    /// <summary>
    /// Drives a queue of <see cref="Job"/>s through an <see cref="IMapProvider"/> + adjustments + map
    /// export sequentially, loading the model once for the whole batch. Non-blocking: each job's
    /// inference runs via the provider's async path, and jobs are chained across editor ticks so the
    /// editor stays responsive. The batch uses one provider/model (chosen by the caller via the
    /// representative recipe) for all jobs; each job's own recipe drives its tuning, maps, and export.
    /// Native vs. derived maps are resolved by <see cref="MapPipeline"/>, identical to single export.
    /// </summary>
    public class BatchProcessor
    {
        private readonly IMapProvider _provider;
        private readonly List<Job> _jobs;
        private readonly Recipe _initRecipe;
        private readonly Action<Job> _onJobChanged;
        private readonly Action<int, int> _onProgress;
        private readonly Action _onAllDone;

        private int _index;
        public bool IsRunning { get; private set; }

        public BatchProcessor(IMapProvider provider, List<Job> jobs, Recipe initRecipe,
                              Action<Job> onJobChanged, Action<int, int> onProgress, Action onAllDone)
        {
            _provider = provider;
            _jobs = jobs;
            _initRecipe = initRecipe;
            _onJobChanged = onJobChanged;
            _onProgress = onProgress;
            _onAllDone = onAllDone;
        }

        public void Start()
        {
            if (_jobs == null || _jobs.Count == 0) { _onAllDone?.Invoke(); return; }

            try
            {
                _provider.Initialize(_initRecipe); // load the model / worker once for the whole batch
            }
            catch (Exception ex)
            {
                // TODO(batch-error-channel, higher priority): this only logs to the console — the user
                // sees nothing in-window when batch init fails (e.g. the selected model size isn't
                // downloaded yet). Add an onError callback so the window can surface it (dialog/status row).
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

            _provider.ExecuteAsync(job.source, job.recipe,
                _ => { },
                maps => CompleteJob(job, maps),
                err => { job.state = JobState.Error; job.error = err; AdvanceAfter(job); });
        }

        private void CompleteJob(Job job, MapSet maps)
        {
            Texture2D adjusted = null;
            try
            {
                if (maps == null || maps.height == null)
                {
                    job.state = JobState.Error; job.error = "No inference result."; AdvanceAfter(job); return;
                }

                Recipe r = job.recipe;
                adjusted = MapPipeline.BuildAdjustedHeight(maps, r);

                job.heightmapPath = TextureExporter.SaveHeightmap(adjusted, job.source, r.format);
                job.normalPath = null;
                job.roughnessPath = null;

                if (r.exportNormal)
                {
                    Texture2D n = MapPipeline.BuildNormal(maps, r, adjusted, out bool derived);
                    if (n != null)
                    {
                        job.normalPath = TextureExporter.SaveNormalMap(n, job.source);
                        if (derived) UnityEngine.Object.DestroyImmediate(n);
                    }
                }

                Texture2D nativeRough = MapPipeline.NativeRoughness(maps);
                if (r.exportRoughness && nativeRough != null)
                {
                    job.roughnessPath = TextureExporter.SaveRoughness(nativeRough, job.source);
                }

                if (r.exportAO)
                {
                    Texture2D a = ImageProcessor.GenerateAO(adjusted, r.aoStrength, r.aoRadius);
                    if (a != null) { TextureExporter.SaveAOMap(a, job.source); UnityEngine.Object.DestroyImmediate(a); }
                }

                RecipeSidecar.Save(job.source, r);
                RawCache.SaveRaw(job.source, maps.height);
                job.state = JobState.Exported;
            }
            catch (Exception ex)
            {
                job.state = JobState.Error;
                job.error = ex.Message;
                Debug.LogError($"[DepthForge] Batch job '{(job.source != null ? job.source.name : "?")}' failed: {ex.Message}");
            }
            finally
            {
                if (adjusted != null) UnityEngine.Object.DestroyImmediate(adjusted);
                // Native maps are transient per job; release them now that export is done.
                if (maps != null)
                {
                    if (maps.height != null) UnityEngine.Object.DestroyImmediate(maps.height);
                    if (maps.normal != null) UnityEngine.Object.DestroyImmediate(maps.normal);
                    if (maps.roughness != null) UnityEngine.Object.DestroyImmediate(maps.roughness);
                }
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
