import cv2
import numpy as np
from typing import List, Tuple, Optional

from image_processing_api.constants import PATCH_NAMES

# ============================================================================
# COLORCHECKER PROCESSING FUNCTIONS
# ============================================================================

def isolate_colorchecker(full_image: np.ndarray, 
                        top_left: Tuple[int, int], 
                        bottom_right: Tuple[int, int]) -> np.ndarray:
    """
    Isolate the ColorChecker region from the full image.
    
    Parameters:
        full_image: Full resolution image
        top_left: (x, y) of ColorChecker top-left corner
        bottom_right: (x, y) of ColorChecker bottom-right corner
    
    Returns:
        Cropped ColorChecker region
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    isolated = full_image[y1:y2, x1:x2].copy()
    
    print(f"\nIsolated ColorChecker region:")
    print(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Size: {x2-x1} x {y2-y1} pixels")
    
    return isolated

def extract_patch_colors(isolated_checker: np.ndarray, 
                        patches: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                        log_lines: Optional[List[str]] = None) -> np.ndarray:
    """
    Extract average color from each patch in the isolated ColorChecker region.
    
    Parameters:
        isolated_checker: Isolated ColorChecker region (full resolution)
        patches: List of 24 bounding boxes relative to isolated region
        log_lines: Optional list to append log messages to
    
    Returns:
        patch_colors: Array of shape (24, 3) with average color of each patch
    """
    patch_colors = []
    
    header = "\n" + "="*70 + "\n" + "EXTRACTING COLORS FROM FULL RESOLUTION PATCHES\n" + "="*70
    print(header)
    if log_lines is not None:
        log_lines.append(header)
    
    for i, (top_left, bottom_right) in enumerate(patches):
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Extract patch from full resolution isolated region
        patch = isolated_checker[y1:y2, x1:x2]
        patch_size = patch.shape[0] * patch.shape[1]
        
        # Calculate average color
        avg_color = np.mean(patch.reshape(-1, 3), axis=0)
        patch_colors.append(avg_color)
        
        line = (f"  Patch {i+1:2d} ({PATCH_NAMES[i]:15s}): "
                f"R={avg_color[0]:.3f}, G={avg_color[1]:.3f}, B={avg_color[2]:.3f} "
                f"[{patch_size:,} pixels]")
        print(line)
        if log_lines is not None:
            log_lines.append(line)
    
    return np.array(patch_colors)

def save_patch_visualization(isolated_checker: np.ndarray, 
                            patches: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                            display_scale: float = 1.0,
                            output_path: str = 'selected_patches.png'):
    """Save visualization of selected patches."""
    h, w = isolated_checker.shape[:2]
    vis_image = cv2.resize(isolated_checker, (int(w * display_scale), int(h * display_scale)))
    vis_image = (vis_image * 255).astype(np.uint8).copy()
    
    for i, (top_left, bottom_right) in enumerate(patches):
        x1_s = int(top_left[0] * display_scale)
        y1_s = int(top_left[1] * display_scale)
        x2_s = int(bottom_right[0] * display_scale)
        y2_s = int(bottom_right[1] * display_scale)
        
        cv2.rectangle(vis_image, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
        cv2.putText(vis_image, str(i + 1), 
                   (x1_s + 5, y1_s + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, vis_bgr)
    print(f"✓ Saved visualization: {output_path}")

def save_comparison_image(original: np.ndarray, corrected: np.ndarray, 
                         display_scale: float = 1.0,
                         output_path: str = 'colorchecker_before_after.png'):
    """Save side-by-side comparison."""
    orig_scaled = cv2.resize(original, (0, 0), fx=display_scale, fy=display_scale)
    corr_scaled = cv2.resize(corrected, (0, 0), fx=display_scale, fy=display_scale)
    
    orig_scaled = (orig_scaled * 255).astype(np.uint8)
    corr_scaled = (corr_scaled * 255).astype(np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_scaled, 'Original', (20, 40), font, 1, (255, 255, 255), 2)
    cv2.putText(corr_scaled, 'Corrected', (20, 40), font, 1, (255, 255, 255), 2)
    
    comparison = np.hstack([orig_scaled, corr_scaled])
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, comparison_bgr)
    print(f"✓ Saved comparison: {output_path}")

# ============================================================================
# INTERACTIVE MULTI-PATCH SELECTOR
# ============================================================================
class MultiPatchSelector:
    """Interactive selector for isolated ColorChecker region."""
    
    def __init__(self, isolated_checker: np.ndarray, display_scale: float = 1.0, num_patches: int = 24):
        """
        Parameters:
            isolated_checker: Isolated ColorChecker region (full resolution)
            display_scale: Scale factor for display
            num_patches: Number of patches to select (default 24)
        """
        self.isolated_fullres = isolated_checker
        self.display_scale = display_scale
        self.num_patches = num_patches
        self.patches_in_isolated = []  # Coordinates relative to isolated region
        self.current_patch = 0
        self.drawing = False
        self.start_point = None
        self.temp_end_point = None
        
        # Create display image
        h, w = isolated_checker.shape[:2]
        
        # Automatically determine good display scale
        max_display = 1200  # Max dimension for comfortable viewing
        auto_scale = min(max_display / w, max_display / h, 1.0)
        if display_scale is None:
            display_scale = auto_scale
        self.display_scale = display_scale
        
        display_h = int(h * self.display_scale)
        display_w = int(w * self.display_scale)
        self.display_image = cv2.resize(isolated_checker, (display_w, display_h))
        
        print(f"\nDisplay settings:")
        print(f"  Isolated region: {w}x{h} pixels")
        print(f"  Display size: {display_w}x{display_h} (scale={self.display_scale:.3f})")
        
        # Convert to uint8 BGR for OpenCV
        if self.display_image.max() <= 1.0:
            self.display_image = (self.display_image * 255).astype(np.uint8)
        self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
        
        self.window_name = "Select 24 Patches in ColorChecker"
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Ensure top-left to bottom-right (in display coordinates)
                x1_disp = min(self.start_point[0], end_point[0])
                y1_disp = min(self.start_point[1], end_point[1])
                x2_disp = max(self.start_point[0], end_point[0])
                y2_disp = max(self.start_point[1], end_point[1])
                
                # Convert to coordinates in isolated region (full resolution)
                x1_iso = int(x1_disp / self.display_scale)
                y1_iso = int(y1_disp / self.display_scale)
                x2_iso = int(x2_disp / self.display_scale)
                y2_iso = int(y2_disp / self.display_scale)
                
                # Store coordinates relative to isolated region
                self.patches_in_isolated.append(((x1_iso, y1_iso), (x2_iso, y2_iso)))
                
                print(f"  ✓ Patch {self.current_patch + 1} ({PATCH_NAMES[self.current_patch]}): "
                      f"({x1_iso}, {y1_iso}) to ({x2_iso}, {y2_iso})")
                
                self.current_patch += 1
                self.start_point = None
                self.temp_end_point = None
    
    def draw_interface(self) -> np.ndarray:
        """Draw the current state of selections."""
        display = self.display_image.copy()
        
        # Draw all completed patches
        for i, (tl_iso, br_iso) in enumerate(self.patches_in_isolated):
            # Scale to display coordinates
            tl_disp = (int(tl_iso[0] * self.display_scale), int(tl_iso[1] * self.display_scale))
            br_disp = (int(br_iso[0] * self.display_scale), int(br_iso[1] * self.display_scale))
            
            # Draw rectangle
            cv2.rectangle(display, tl_disp, br_disp, (0, 255, 0), 2)
            
            # Draw patch number
            cv2.putText(display, str(i + 1), 
                       (tl_disp[0] + 5, tl_disp[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw current patch being drawn
        if self.drawing and self.start_point and self.temp_end_point:
            cv2.rectangle(display, self.start_point, self.temp_end_point, (255, 255, 0), 2)
        
        # Draw instructions
        if self.current_patch < self.num_patches:
            instruction = f"Patch {self.current_patch + 1}/{self.num_patches}: {PATCH_NAMES[self.current_patch]}"
            cv2.putText(display, instruction, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Click & drag | U=Undo | R=Reset | ENTER=Done | ESC=Cancel",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(display, "All 24 patches selected! Press ENTER to confirm",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display
    
    def select_patches(self) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Main loop for selecting patches.
        
        Returns:
            List of 24 bounding boxes relative to isolated region, or None if cancelled
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*70)
        print("SELECT 24 COLORCHECKER PATCHES")
        print("="*70)
        print("\nInstructions:")
        print("  - Select patches in order: Row 1 (left→right), Row 2, Row 3, Row 4")
        print("  - Click and drag to draw a box around each patch")
        print("\nControls:")
        print("  U: Undo last selection")
        print("  R: Reset all")
        print("  ENTER: Confirm (when all 24 selected)")
        print("  ESC: Cancel")
        print("="*70 + "\n")
        
        while True:
            display = self.draw_interface()
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(10) & 0xFF
            
            if key == 13:  # ENTER
                if len(self.patches_in_isolated) == self.num_patches:
                    cv2.destroyAllWindows()
                    print(f"\n✓ All {self.num_patches} patches selected!")
                    return self.patches_in_isolated
                else:
                    print(f"Need {self.num_patches} patches, have {len(self.patches_in_isolated)}")
            
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                print("\n✗ Selection cancelled")
                return None
            
            elif key == ord('u') or key == ord('U'):
                if len(self.patches_in_isolated) > 0:
                    self.patches_in_isolated.pop()
                    self.current_patch -= 1
                    print(f"  ↩ Undid patch {self.current_patch + 1}")
            
            elif key == ord('r') or key == ord('R'):
                self.patches_in_isolated = []
                self.current_patch = 0
                print("  ↻ Reset all selections")

# ============================================================================
# PATCH CLIPPING DIAGNOSIS
# ============================================================================

def diagnose_patch_clipping(isolated_checker: np.ndarray, 
                            patches: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                            log_lines: Optional[List[str]] = None) -> List[int]:
    """
    Check which patches are clipped/overexposed.
    
    Returns:
        List of clipped patch indices
    """
    header = "\n" + "="*70 + "\n" + "CHECKING FOR CLIPPED/OVEREXPOSED PATCHES\n" + "="*70
    print(header)
    if log_lines is not None:
        log_lines.append(header)
    
    clipped_patches = []
    
    for i, (top_left, bottom_right) in enumerate(patches):
        x1, y1 = top_left
        x2, y2 = bottom_right
        patch = isolated_checker[y1:y2, x1:x2]
        
        max_val = np.max(patch)
        mean_val = np.mean(patch)
        clipped_pixels = np.sum(patch >= 0.99)  # Check for near-saturation
        total_pixels = patch.size
        clipped_pct = (clipped_pixels / total_pixels) * 100
        
        if clipped_pct > 1:
            status = "⚠ CLIPPED!"
            clipped_patches.append(i)
        else:
            status = "✓ OK"
        
        line = (f"  Patch {i+1:2d} ({PATCH_NAMES[i]:15s}): "
                f"Max={max_val:.3f}, Mean={mean_val:.3f}, "
                f"Clipped={clipped_pct:.1f}% {status}")
        print(line)
        if log_lines is not None:
            log_lines.append(line)
    
    return clipped_patches

def analyze_color_matrix(color_matrix: np.ndarray):
    """Analyze color correction matrix for issues."""
    print("\n" + "="*70)
    print("COLOR MATRIX ANALYSIS")
    print("="*70)
    print("\nMatrix:")
    print(color_matrix)
    
    # Check for extreme values
    max_val = np.max(np.abs(color_matrix))
    min_val = np.min(np.abs(color_matrix[color_matrix != 0]))
    
    print(f"\nMax absolute value: {max_val:.3f}")
    print(f"Min absolute value: {min_val:.3f}")
    print(f"Ratio (max/min): {max_val/min_val:.3f}")
    
    if max_val > 3.0:
        print("⚠️  WARNING: Matrix has very large values (>3.0)")
        print("   This can cause over-amplification of certain channels")
    
    if max_val / min_val > 10:
        print("⚠️  WARNING: Matrix is poorly conditioned (ratio >10)")
        print("   Consider recalibrating with more carefully selected patches")
    
    # Check diagonal dominance (diagonal should be largest in each row)
    for i in range(3):
        diag = abs(color_matrix[i, i])
        off_diag = [abs(color_matrix[i, j]) for j in range(3) if j != i]
        max_off = max(off_diag)
        if diag < max_off:
            print(f"⚠️  WARNING: Row {i} is not diagonally dominant")
            print(f"   Diagonal: {diag:.3f}, Max off-diagonal: {max_off:.3f}")