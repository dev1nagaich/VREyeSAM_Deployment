import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import io
import sys
import os

# Add segment-anything-2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "segment-anything-2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Page config
st.set_page_config(
    page_title="VREyeSAM - Non-frontal Iris Segmentation",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 16px;
    }
    .result-box {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the VREyeSAM model"""
    try:
        model_cfg = "configs/sam2/sam2_hiera_s.yaml"
        sam2_checkpoint = "segment-anything-2/checkpoints/sam2_hiera_small.pt"
        fine_tuned_weights = "segment-anything-2/checkpoints/VREyeSAM_uncertainity_best.torch"
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.model.load_state_dict(torch.load(fine_tuned_weights, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def read_and_resize_image(image):
    """Read and resize image for processing"""
    img = np.array(image)
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize if needed
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    if r < 1:
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    
    return img

def segment_iris(predictor, image):
    """Perform iris segmentation"""
    # Generate random points for inference
    num_samples = 30
    input_points = np.random.randint(0, min(image.shape[:2]), (num_samples, 1, 2))
    
    # Inference
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    
    # Convert to numpy
    np_masks = np.array(masks[:, 0]).astype(np.float32)
    np_scores = scores[:, 0]
    
    # Normalize scores
    score_sum = np.sum(np_scores)
    if score_sum > 0:
        normalized_scores = np_scores / score_sum
    else:
        normalized_scores = np.ones_like(np_scores) / len(np_scores)
    
    # Generate probabilistic mask
    prob_mask = np.sum(np_masks * normalized_scores[:, None, None], axis=0)
    prob_mask = np.clip(prob_mask, 0, 1)
    
    # Threshold to get binary mask
    binary_mask = (prob_mask > 0.2).astype(np.uint8)
    
    return binary_mask, prob_mask

def extract_iris_strip(image, binary_mask):
    """Extract iris region and create a rectangular strip"""
    # Find contours in binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get the largest contour (assumed to be the iris)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add some padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Extract the iris region
    iris_region = image[y:y+h, x:x+w]
    
    # Create a rectangular strip (normalize height)
    strip_height = 150
    aspect_ratio = w / h
    strip_width = int(strip_height * aspect_ratio)
    
    iris_strip = cv2.resize(iris_region, (strip_width, strip_height))
    
    return iris_strip

def overlay_mask_on_image(image, binary_mask, color=(0, 255, 0), alpha=0.5):
    """Overlay binary mask on original image"""
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[binary_mask > 0] = color
    
    # Blend
    result = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
    
    return result

# Main App
def main():
    st.title("👁️ VREyeSAM: Non-Frontal Iris Segmentation")
    st.markdown("""
    Upload a non-frontal iris image captured in VR/AR environments, and VREyeSAM will segment the iris region 
    using a fine-tuned SAM2 model with uncertainty-weighted loss.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About VREyeSAM")
        st.markdown("""
        **VREyeSAM** is a robust iris segmentation framework designed for images captured under:
        - Varying gaze directions
        - Partial occlusions
        - Inconsistent lighting conditions
        
        **Model Performance:**
        - Precision: 0.751
        - Recall: 0.870
        - F1-Score: 0.806
        - Mean IoU: 0.647
        
        
        """)
        
        st.header("Settings")
        show_overlay = st.checkbox("Show mask overlay", value=True)
        show_probabilistic = st.checkbox("Show probabilistic mask", value=False)
    
    # Load model
    with st.spinner("Loading VREyeSAM model..."):
        predictor = load_model()
    
    if predictor is None:
        st.error("Failed to load model. Please check the setup.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an iris image (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        help="Upload a non-frontal iris image for segmentation"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🔍 Segment Iris", type="primary"):
            with st.spinner("Segmenting iris..."):
                # Prepare image
                img_array = read_and_resize_image(image)
                
                # Perform segmentation
                binary_mask, prob_mask = segment_iris(predictor, img_array)
                
                # Extract iris strip
                 ## iris_strip = extract_iris_strip(img_array, binary_mask)
                
                with col2:
                    st.subheader("🎯 Binary Mask")
                    binary_mask_img = (binary_mask * 255).astype(np.uint8)
                    st.image(binary_mask_img, use_container_width=True)
                
                # Additional results
                st.markdown("---")
                st.subheader("📊 Segmentation Results")
                
                result_cols = st.columns(3)
                
                with result_cols[0]:
                    if show_overlay:
                        st.markdown("**Overlay View**")
                        overlay = overlay_mask_on_image(img_array, binary_mask)
                        st.image(overlay, use_container_width=True)
                
                with result_cols[1]:
                    if show_probabilistic:
                        st.markdown("**Probabilistic Mask**")
                        prob_mask_img = (prob_mask * 255).astype(np.uint8)
                        st.image(prob_mask_img, use_container_width=True)
                
        #        with result_cols[2]:
        #           if iris_strip is not None:
        #                st.markdown("**Extracted Iris Strip**")
        #                st.image(iris_strip, use_container_width=True)
        #           else:
        #                st.warning("No iris region detected")
                
                # Download options
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                download_cols = st.columns(3)
                
                with download_cols[0]:
                    # Binary mask download
                    binary_pil = Image.fromarray(binary_mask_img)
                    buf = io.BytesIO()
                    binary_pil.save(buf, format="PNG")
                    st.download_button(
                        label="Download Binary Mask",
                        data=buf.getvalue(),
                        file_name="binary_mask.png",
                        mime="image/png"
                    )
                
                with download_cols[1]:
                    if show_overlay:
                        # Overlay download
                        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                        buf = io.BytesIO()
                        overlay_pil.save(buf, format="PNG")
                        st.download_button(
                            label="Download Overlay",
                            data=buf.getvalue(),
                            file_name="overlay.png",
                            mime="image/png"
                        )
                
        #        with download_cols[2]:
        #           if iris_strip is not None:
        #                # Iris strip download
        #                strip_pil = Image.fromarray(cv2.cvtColor(iris_strip, cv2.COLOR_BGR2RGB))
        #                buf = io.BytesIO()
        #                strip_pil.save(buf, format="PNG")
        #                st.download_button(
        #                    label="Download Iris Strip",
        #                    data=buf.getvalue(),
        #                    file_name="iris_strip.png",
        #                    mime="image/png"
        #                )
                
                # Statistics
                st.markdown("---")
                st.subheader("📈 Segmentation Statistics")
                stats_cols = st.columns(4)
                
                mask_area = np.sum(binary_mask > 0)
                total_area = binary_mask.shape[0] * binary_mask.shape[1]
                coverage = (mask_area / total_area) * 100
                
                with stats_cols[0]:
                    st.metric("Mask Coverage", f"{coverage:.2f}%")
                with stats_cols[1]:
                    st.metric("Image Size", f"{img_array.shape[1]}x{img_array.shape[0]}")
                with stats_cols[2]:
                    st.metric("Mask Area (pixels)", f"{mask_area:,}")
        #        with stats_cols[3]:
        #            if iris_strip is not None:
        #                st.metric("Strip Size", f"{iris_strip.shape[1]}x{iris_strip.shape[0]}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>VREyeSAM</strong> - Virtual Reality Non-Frontal Iris Segmentation</p>
        <p>🔗 <a href='https://github.com/GeetanjaliGTZ/VREyeSAM'>GitHub</a> | 
        📧 <a href='mailto:geetanjalisharma546@gmail.com'>Contact</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()