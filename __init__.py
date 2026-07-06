from .nodes import AdvancedHandOrientationOptimizer, AdvancedHandMaskRefiner, AdvancedHandSeamlessStitcher

NODE_CLASS_MAPPINGS = {
    "AdvancedHandOrientationOptimizer": AdvancedHandOrientationOptimizer,
    "AdvancedHandMaskRefiner": AdvancedHandMaskRefiner,
    "AdvancedHandSeamlessStitcher": AdvancedHandSeamlessStitcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedHandOrientationOptimizer": "👋 Hand Orientation & Crop Optimizer",
    "AdvancedHandMaskRefiner": "✨ Advanced Anatomical Mask Refiner",
    "AdvancedHandSeamlessStitcher": "🪡 Seamless Stitch & Color Matcher"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
