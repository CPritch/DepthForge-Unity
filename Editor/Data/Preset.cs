using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace CPritch.DepthForge.Editor.Data
{
    /// <summary>
    /// A named recipe template (R6). Presets only carry the *tuning* half of a recipe
    /// (adjustments + maps + format); the generation half (model / backend / tiling / letterbox)
    /// stays whatever the focused job already uses — see <see cref="OverlayTuning"/>. That keeps a
    /// "Rock" look applicable regardless of which model the user has downloaded.
    /// </summary>
    [Serializable]
    public class Preset
    {
        public string name;
        public bool builtin;
        public Recipe recipe;

        public Preset() { }

        public Preset(string name, Recipe recipe, bool builtin)
        {
            this.name = name;
            this.recipe = recipe;
            this.builtin = builtin;
        }

        /// <summary>
        /// Copies this preset's tuning fields onto <paramref name="target"/>, leaving the target's
        /// generation settings (modelSize/backend/tiledInference/tilingAlignment/letterbox) untouched.
        /// </summary>
        public void OverlayTuning(Recipe target)
        {
            if (target == null || recipe == null) return;
            target.invert = recipe.invert;
            target.contrast = recipe.contrast;
            target.midpoint = recipe.midpoint;
            target.flatten = recipe.flatten;
            target.exportNormal = recipe.exportNormal;
            target.normalStrength = recipe.normalStrength;
            target.exportAO = recipe.exportAO;
            target.aoStrength = recipe.aoStrength;
            target.aoRadius = recipe.aoRadius;
            target.format = recipe.format;
        }
    }

    [Serializable]
    internal class PresetCollection
    {
        public List<Preset> presets = new List<Preset>();
    }

    /// <summary>
    /// Persists user-defined presets in <see cref="EditorPrefs"/> (project-scoped key) and exposes the
    /// built-in starting points. Built-ins are regenerated each call so an app update can refine them.
    /// </summary>
    public static class PresetStore
    {
        private const string PrefsKey = "DepthForge.UserPresets";

        public const string DefaultName = "Default";

        public static List<Preset> GetBuiltins()
        {
            return new List<Preset>
            {
                Make(DefaultName, contrast: 1.0f, midpoint: 0.5f, flatten: 0.0f, normal: 8f, ao: 1.0f),
                // Sharp, high-relief surfaces.
                Make("Rock",   contrast: 1.3f, midpoint: 0.5f, flatten: 0.10f, normal: 12f, ao: 1.4f),
                // Deep directional grooves.
                Make("Bark",   contrast: 1.5f, midpoint: 0.5f, flatten: 0.0f,  normal: 14f, ao: 1.6f),
                // Soft weave — flatten the broad shape, keep gentle surface.
                Make("Fabric", contrast: 0.8f, midpoint: 0.5f, flatten: 0.40f, normal: 5f,  ao: 0.7f),
                // Near-flat panel with fine micro-detail only.
                Make("Metal",  contrast: 1.1f, midpoint: 0.5f, flatten: 0.60f, normal: 4f,  ao: 0.5f),
            };
        }

        private static Preset Make(string name, float contrast, float midpoint, float flatten, float normal, float ao)
        {
            var r = new Recipe
            {
                contrast = contrast,
                midpoint = midpoint,
                flatten = flatten,
                normalStrength = normal,
                aoStrength = ao,
            };
            return new Preset(name, r, builtin: true);
        }

        public static List<Preset> GetUserPresets()
        {
            string json = EditorPrefs.GetString(PrefsKey, string.Empty);
            if (string.IsNullOrEmpty(json)) return new List<Preset>();
            try
            {
                var coll = JsonUtility.FromJson<PresetCollection>(json);
                if (coll?.presets != null)
                {
                    foreach (var p in coll.presets) p.builtin = false;
                    return coll.presets;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"DepthForge: failed to read user presets — {ex.Message}");
            }
            return new List<Preset>();
        }

        /// <summary>Built-ins first, then user presets, in save order.</summary>
        public static List<Preset> GetAll()
        {
            var all = GetBuiltins();
            all.AddRange(GetUserPresets());
            return all;
        }

        public static Preset Find(string name)
        {
            if (string.IsNullOrEmpty(name)) return null;
            return GetAll().Find(p => p.name == name);
        }

        public static bool IsBuiltin(string name)
        {
            var p = GetBuiltins().Find(x => x.name == name);
            return p != null;
        }

        /// <summary>
        /// Saves (or overwrites) a user preset. Returns false if the name collides with a built-in
        /// (those are reserved). The stored recipe is a tuning-only clone of <paramref name="recipe"/>.
        /// </summary>
        public static bool SaveUserPreset(string name, Recipe recipe)
        {
            if (string.IsNullOrWhiteSpace(name) || recipe == null) return false;
            name = name.Trim();
            if (IsBuiltin(name)) return false;

            var users = GetUserPresets();
            users.RemoveAll(p => p.name == name);
            users.Add(new Preset(name, recipe.Clone(), builtin: false));
            Persist(users);
            return true;
        }

        public static void DeleteUserPreset(string name)
        {
            if (string.IsNullOrEmpty(name) || IsBuiltin(name)) return;
            var users = GetUserPresets();
            users.RemoveAll(p => p.name == name);
            Persist(users);
        }

        private static void Persist(List<Preset> users)
        {
            var coll = new PresetCollection { presets = users };
            EditorPrefs.SetString(PrefsKey, JsonUtility.ToJson(coll));
        }
    }
}
