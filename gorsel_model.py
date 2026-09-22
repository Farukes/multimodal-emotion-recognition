"""
Visual Emotion Recognition Model Runner (Compatibility Script)

This script is provided for convenience and backward compatibility.
See visual_model.py for detailed architecture, arguments, and implementation.
"""

from visual_model import AttentionBlock, main

__all__ = ["AttentionBlock", "main"]

if __name__ == "__main__":
    main()
