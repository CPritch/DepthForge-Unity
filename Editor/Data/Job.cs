using UnityEngine;
using UnityEditor;

namespace CPritch.DepthForge.Editor.Data
{
    public enum JobState { Queued, Generating, Ready, Edited, Exported, Error }

    /// <summary>
    /// One source texture moving through the depth pipeline: its Source + Recipe + runtime state +
    /// generated results. The <see cref="Recipe"/> is what persists (sidecar); the textures here are
    /// transient and regenerated. This is the unit the future batch queue holds a list of.
    /// </summary>
    public class Job
    {
        public string sourceGuid;
        public Texture2D source;
        public Recipe recipe = new Recipe();
        public JobState state = JobState.Queued;
        public string error;

        // Transient results — regenerated, never serialized.
        public Texture2D rawHeightmap;
        public Texture2D adjustedHeightmap;

        // Last-exported asset paths (transient) — used by batch MicroSplat assign-all.
        public string heightmapPath;
        public string normalPath;

        public Job() { }

        public Job(Texture2D source)
        {
            SetSource(source);
        }

        public void SetSource(Texture2D tex)
        {
            source = tex;
            sourceGuid = null;
            if (tex != null)
            {
                string path = AssetDatabase.GetAssetPath(tex);
                if (!string.IsNullOrEmpty(path))
                {
                    sourceGuid = AssetDatabase.AssetPathToGUID(path);
                }
            }
        }
    }
}
