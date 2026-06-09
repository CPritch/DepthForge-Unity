using UnityEngine;
using UnityEditor;
using System.IO;

namespace CPritch.DepthForge.Editor.Utils
{
    public static class TextureExporter
    {
        public enum ExportFormat
        {
            PNG_8Bit,
            EXR_16Bit,
            // MicroSplat often uses packed textures. We'll add standard formats for now
            // and can add specific channel packing later.
            MicroSplat_Height
        }

        public static string SaveHeightmap(Texture2D generatedTexture, Texture2D sourceTexture, ExportFormat format)
        {
            if (generatedTexture == null || sourceTexture == null)
            {
                Debug.LogError("Cannot save heightmap: texture is null.");
                return null;
            }

            string sourcePath = AssetDatabase.GetAssetPath(sourceTexture);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogError("Source texture is not saved in the project.");
                return null;
            }

            string directory = Path.GetDirectoryName(sourcePath);
            string filename = Path.GetFileNameWithoutExtension(sourcePath) + "_Height";
            
            string extension = format == ExportFormat.EXR_16Bit ? ".exr" : ".png";
            string savePath = Path.Combine(directory, filename + extension);
            
            // Normalize path separators
            savePath = savePath.Replace("\\", "/");

            byte[] bytes;
            if (format == ExportFormat.EXR_16Bit)
            {
                // Unity's EncodeToEXR fails or outputs flat black if the texture is not an HDR format
                Texture2D hdrTex = new Texture2D(generatedTexture.width, generatedTexture.height, TextureFormat.RGBAHalf, false, true);
                hdrTex.SetPixels(generatedTexture.GetPixels());
                hdrTex.Apply();
                bytes = ImageConversion.EncodeToEXR(hdrTex, Texture2D.EXRFlags.CompressZIP);
                UnityEngine.Object.DestroyImmediate(hdrTex);
            }
            else
            {
                // Both PNG_8Bit and MicroSplat_Height will use PNG for now
                // Later we can implement custom channel packing for MicroSplat if required.
                bytes = ImageConversion.EncodeToPNG(generatedTexture);
            }

            File.WriteAllBytes(savePath, bytes);
            AssetDatabase.Refresh();

            // Configure importer settings
            TextureImporter importer = AssetImporter.GetAtPath(savePath) as TextureImporter;
            if (importer != null)
            {
                importer.sRGBTexture = false; // Heightmaps should always be linear
                importer.mipmapEnabled = true;
                importer.textureCompression = TextureImporterCompression.Uncompressed; // Or high quality
                importer.SaveAndReimport();
            }

            Debug.Log($"Heightmap saved to: {savePath}");
            return savePath;
        }

        /// <summary>
        /// Saves a generated normal map next to the source texture as "&lt;name&gt;_Normal.png" and
        /// configures its importer as a proper Unity NormalMap (linear, mip-mapped). This gives
        /// MicroSplat real normal data instead of having it synthesize normals from the diffuse.
        /// </summary>
        public static string SaveNormalMap(Texture2D normalTexture, Texture2D sourceTexture)
        {
            if (normalTexture == null || sourceTexture == null)
            {
                Debug.LogError("Cannot save normal map: texture is null.");
                return null;
            }

            string sourcePath = AssetDatabase.GetAssetPath(sourceTexture);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogError("Source texture is not saved in the project.");
                return null;
            }

            string directory = Path.GetDirectoryName(sourcePath);
            string filename = Path.GetFileNameWithoutExtension(sourcePath) + "_Normal";
            string savePath = Path.Combine(directory, filename + ".png").Replace("\\", "/");

            byte[] bytes = ImageConversion.EncodeToPNG(normalTexture);
            File.WriteAllBytes(savePath, bytes);
            AssetDatabase.Refresh();

            TextureImporter importer = AssetImporter.GetAtPath(savePath) as TextureImporter;
            if (importer != null)
            {
                importer.textureType = TextureImporterType.NormalMap; // forces linear + correct sampling
                importer.mipmapEnabled = true;
                importer.SaveAndReimport();
            }

            Debug.Log($"Normal map saved to: {savePath}");
            return savePath;
        }

        /// <summary>
        /// Saves a generated ambient-occlusion map next to the source as "&lt;name&gt;_AO.png" and
        /// configures its importer as linear, mip-mapped grayscale.
        /// </summary>
        public static string SaveAOMap(Texture2D aoTexture, Texture2D sourceTexture)
        {
            if (aoTexture == null || sourceTexture == null)
            {
                Debug.LogError("Cannot save AO map: texture is null.");
                return null;
            }

            string sourcePath = AssetDatabase.GetAssetPath(sourceTexture);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogError("Source texture is not saved in the project.");
                return null;
            }

            string directory = Path.GetDirectoryName(sourcePath);
            string filename = Path.GetFileNameWithoutExtension(sourcePath) + "_AO";
            string savePath = Path.Combine(directory, filename + ".png").Replace("\\", "/");

            byte[] bytes = ImageConversion.EncodeToPNG(aoTexture);
            File.WriteAllBytes(savePath, bytes);
            AssetDatabase.Refresh();

            TextureImporter importer = AssetImporter.GetAtPath(savePath) as TextureImporter;
            if (importer != null)
            {
                importer.sRGBTexture = false; // AO is linear data
                importer.mipmapEnabled = true;
                importer.SaveAndReimport();
            }

            Debug.Log($"AO map saved to: {savePath}");
            return savePath;
        }
    }
}
