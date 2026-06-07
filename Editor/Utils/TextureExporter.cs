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
                bytes = ImageConversion.EncodeToEXR(generatedTexture, Texture2D.EXRFlags.CompressZIP);
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
    }
}
