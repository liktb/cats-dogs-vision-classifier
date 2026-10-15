import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# ---------------------------------------------------------
# 1. Page Configuration & Header
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Cats vs. Dogs Classifier")
st.write("Upload an image of a cat or a dog, and the AI model will predict what it is!")

# ---------------------------------------------------------
# 2. Model Loading Function
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Loads and caches the trained PyTorch model.
    Replace 'models/model.pth' with the actual path to your saved model weights.
    """
    try:
        # NOTE: Once Teammate 2 gives you the model structure and saved weights file,
        # uncomment and update the loading logic below:
        
        # model = MyModelClass() # Import your model architecture here
        # model.load_state_dict(torch.load('models/model.pth', map_location=torch.device('cpu')))
        # model.eval()
        # return model
        
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

# ---------------------------------------------------------
# 3. Image Preprocessing Pipeline
# ---------------------------------------------------------
def transform_image(image):
    """
    Transforms the input image into the format expected by the model.
    Adjust resize dimensions and normalization parameters to match Teammate 1/2's setup.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Standard ImageNet means
            std=[0.229, 0.224, 0.225]   # Standard ImageNet stds
        )
    ])
    return preprocess(image).unsqueeze(0) # Add batch dimension

# ---------------------------------------------------------
# 4. User Upload & Prediction Interface
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image uploaded by the user
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify button
    if st.button("Classify Image 🚀"):
        with st.spinner("Analyzing image..."):
            
            # Preprocess the image
            input_tensor = transform_image(image)

            # --- PREDICTION LOGIC ---
            if model is not None:
                # Actual Model Inference (runs when model weights are connected)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    
                    # Assuming class 0 is Cat and class 1 is Dog
                    cat_prob = probabilities[0].item()
                    dog_prob = probabilities[1].item()
                    
                    if dog_prob > cat_prob:
                        label = "Dog 🐶"
                        confidence = dog_prob
                    else:
                        label = "Cat 🐱"
                        confidence = cat_prob
            else:
                # Temporary Mock Result (used until model weights file is added)
                import random
                label = random.choice(["Cat 🐱", "Dog 🐶"])
                confidence = round(random.uniform(0.85, 0.99), 2)
                st.info("ℹ️ Running in demo mode using sample output (Model weights not connected yet).")

            # Output results to UI
            st.success(f"**Prediction:** {label}")
            st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")