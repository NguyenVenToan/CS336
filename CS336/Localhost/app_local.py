import os
import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
@st.cache_resource  # Cache để không tải lại model mỗi lần
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

# Load ALIGN model
@st.cache_resource
def load_align_model():
    model = AutoModel.from_pretrained("kakaobrain/align-base")
    processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
    return model, processor

# Hàm tìm kiếm ảnh
def search_image(text_query, model, processor, text_embeddings, image_embeddings, image_paths, top_k=5):
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).numpy()

    similarities = cosine_similarity(text_features, image_embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    result_images = [image_paths[i] for i in top_k_indices]
    return result_images

# Load embeddings và đường dẫn ảnh
@st.cache_data
def load_data(model_name):
    if model_name == "CLIP":
        text_embeddings = np.load("text_embeddings.npy")
        image_embeddings = np.load("image_embeddings.npy")
    elif model_name == "ALIGN":
        text_embeddings = np.load("text_embeddings_align.npy")
        image_embeddings = np.load("image_embeddings_align.npy")

    image_filenames = np.load("image_filenames.npy", allow_pickle=True)
    images_dir = "images"  # Thay thế bằng đường dẫn thư mục của bạn
    image_paths = [os.path.join(images_dir, filename) for filename in image_filenames]

    return text_embeddings, image_embeddings, image_paths

# Giao diện Streamlit
st.set_page_config(
    page_title="Image Search with CLIP & ALIGN",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📷 Image Search from Text using CLIP and ALIGN")
st.markdown(
    """
    Welcome to the **Image Search** application! 🎨  
    Simply enter a text description, choose a model, and let AI find the most relevant images for you! 🚀
    """
)

# Sidebar for model selection
st.sidebar.header("Model Settings")
model_name = st.sidebar.radio("Choose a model:", ["CLIP", "ALIGN"])
st.sidebar.write("Select the preferred model to process your query.")

# Main input section
st.markdown("### 🔍 Search for Images")
text_query = st.text_input("Enter a description (e.g., 'A sunset over the ocean'):", "")
top_k = st.slider("Number of images to retrieve:", min_value=1, max_value=20, value=5)

# Search button
if st.button("Search"):
    if text_query:
        # Load selected model
        if model_name == "CLIP":
            model, processor = load_clip_model()
        elif model_name == "ALIGN":
            model, processor = load_align_model()

        # Load embeddings and paths
        text_embeddings, image_embeddings, image_paths = load_data(model_name)

        # Perform image search
        result_images = search_image(text_query, model, processor, text_embeddings, image_embeddings, image_paths, top_k=top_k)

        # Display results
        st.markdown("### 🎯 Search Results")
        cols = st.columns(3)
        for i, img_path in enumerate(result_images):
            col = cols[i % 3]
            try:
                with col:
                    st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
            except FileNotFoundError:
                st.error(f"Image not found: {img_path}")
    else:
        st.warning("Please enter a text description to start the search!")

# Footer
st.markdown(
    """
    ---
    **Developed by Nguyễn Vẹn Toàn, Đào Văn Tuân, Vũ Anh Tuấn**  
    """
)
