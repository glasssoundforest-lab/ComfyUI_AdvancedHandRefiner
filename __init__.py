from .nodes import (
    AdvancedHandAutoFixer,
    AdvancedHandMaskRefiner,
    AdvancedHandOrientationOptimizer,
    AdvancedHandQualityChecker,
    AdvancedHandSeamlessStitcher,
)

NODE_CLASS_MAPPINGS = {
    "AdvancedHandOrientationOptimizer": AdvancedHandOrientationOptimizer,
    "AdvancedHandMaskRefiner": AdvancedHandMaskRefiner,
    "AdvancedHandSeamlessStitcher": AdvancedHandSeamlessStitcher,
    "AdvancedHandQualityChecker": AdvancedHandQualityChecker,
    "AdvancedHandAutoFixer": AdvancedHandAutoFixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedHandOrientationOptimizer": "👋 Hand Orientation & Crop Optimizer",
    "AdvancedHandMaskRefiner": "✨ Advanced Anatomical Mask Refiner",
    "AdvancedHandSeamlessStitcher": "🪡 Seamless Stitch & Color Matcher",
    "AdvancedHandQualityChecker": "🔍 Advanced Hand Quality Checker",
    "AdvancedHandAutoFixer": "🔁 Advanced Hand Auto Fixer",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
