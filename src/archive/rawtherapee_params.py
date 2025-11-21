from typing import Dict, Any

# ==================== PARAMETER GRIDS ====================

# Define parameter grids to test
# You can modify these ranges based on what you want to explore

PARAMETER_GRIDS = {
    # Preprocessing
    'black_level': {
        'values': [0],  # Your value is 93
        'description': 'Black point subtraction (0-255 scale)'
    },
    
    # Exposure and tone
    'exposure_stops': {
        'values': [0.3, 0.5, 0.70, 0.85, 0.90],
        # 'values': [0.7, 0.9, 1.3, 1.7, 2.0],  # Your value is 2.72
        'description': 'Exposure compensation (stops)'
    },
    
    'highlight_threshold': {
        'values': [0.85, 0.90, 0.95],  # Your value is 0.85
        'description': 'Highlight rolloff threshold'
    },
    
    'highlight_smoothness': {
        'values': [0.03, 0.05, 0.10, 0.15, 0.20],
        'description': 'Highlight rolloff smoothness'
    },
    'highlight_strength': {  # NEW!
        'values': [0.5, 1.0, 1.5, 2.0, 3.0],
        'description': 'Highlight compression strength'
    },
    
    'shadow_compression': {
        'values': [25, 53],  # Your value is 53
        # 'values': [25, 53, 75],  # Your value is 53
        'description': 'Shadow lift amount (0-100)'
    },
    
    'shadow_threshold': {
        'values': [0.10, 0.15, 0.20],  # Default is 0.15
        'description': 'Shadow region threshold'
    },
    
    # Creative adjustments
    'brightness': {
        'values': [5, 10, 15, 20, 25, 30],  # Your value is 5
        'description': 'Brightness adjustment (-100 to +100)'
    },
    
    'contrast': {
        'values': [10, 20],  # Your value is 20
        'description': 'Contrast adjustment (-100 to +100)'
    },
    
    'saturation': {
        'values': [15, 20, 25],  # Your value is 15
        'description': 'Saturation adjustment (-100 to +100)'
    },
    
    # Display encoding
    'gamma': {
        'values': [0.6, 0.7, 0.8, 0.95, 1.0],  # Your value is 0.9
        # 'values': [0.85, 0.9, 0.95, 1.0],  # Your value is 0.9
        'description': 'Gamma for display encoding'
    },
    
    # Post-processing
    'vignette_strength': {
        'values': [1.1, 0.7, 0.0],  # Your value is 1.1
        'description': 'Vignetting strength'
    }
}

# ==================== SWEEP CONFIGURATIONS ====================

def get_baseline_params() -> Dict[str, Any]:
    """Get your baseline RawTherapee parameters."""
    return {
        'black_level': 0,
        'exposure_stops': 0.3,
        'highlight_protection': 'rolloff',
        'highlight_threshold': 0.90,
        'highlight_smoothness': 0.10,
        'highlight_strength': 1.0,
        'shadow_compression': 53,
        'shadow_threshold': 0.15,
        'brightness': 30,
        'contrast': 10,
        'saturation': 15,
        'gamma': 0.95,
        'vignette_strength': 1.1, #0.5,
        'vignette_feather': 0, #88,
        'vignette_roundness': 0
    }