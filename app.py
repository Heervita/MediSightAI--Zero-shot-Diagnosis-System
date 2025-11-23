import streamlit as st
import time
from src import MedicalModel, RAGEngine, process_image
from src.utils import get_reference_images

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediSight AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD SYSTEM (CACHED) ---
# @st.cache_resource prevents the model from reloading on every button click
@st.cache_resource
def load_resources():
    model = MedicalModel()
    rag = RAGEngine()
    return model, rag

# Load the heavy resources once
with st.spinner("Booting up MediSight AI Systems... (Approx. 20s)"):
    model, rag = load_resources()

# --- SIDEBAR: CONTROLS & DATABASE ---
with st.sidebar:
    st.title("⚙️ Diagnostic Panel")
    
    st.divider()
    st.subheader("1. Define Conditions")
    # Default list of diseases to check against
    default_conditions = "Pneumonia, Covid-19, Fracture, Tumor, Healthy"
    conditions_input = st.text_area("Possible Diagnoses (comma-separated):", default_conditions)
    condition_list = [c.strip() for c in conditions_input.split(",") if c.strip()]

    st.divider()
    st.subheader("2. Knowledge Base")
    # A button to force the system to learn from the 'data/reference_images' folder
    import re  # Add this import at the very top of your file if not there

    # ... inside your sidebar code ...

    if st.button("🔄 Update Medical Database"):
        count = 0
        progress_bar = st.progress(0)
        
        ref_images = list(get_reference_images())
        total_imgs = len(ref_images)
        
        if total_imgs == 0:
            st.error("No images found in 'data/reference_images/'")
        else:
            for i, (img_path, filename) in enumerate(ref_images):
                try:
                    # --- NEW: SMART FILENAME PARSER ---
                    clean_name = filename.lower()
                    description = "Unknown Condition"
                    
                    # 1. Check for NORMAL cases
                    if "normal" in clean_name:
                        description = "Healthy / Normal Lung Scan"
                        
                    # 2. Check for VIRAL Pneumonia
                    elif "virus" in clean_name:
                        # Try to find the patient number (e.g., person1 -> Patient 1)
                        match = re.search(r'person(\d+)', clean_name)
                        patient_id = match.group(1) if match else "Unknown"
                        description = f"Confirmed Viral Pneumonia (Patient #{patient_id})"
                        
                    # 3. Check for BACTERIAL Pneumonia
                    elif "bacteria" in clean_name:
                        match = re.search(r'person(\d+)', clean_name)
                        patient_id = match.group(1) if match else "Unknown"
                        description = f"Confirmed Bacterial Pneumonia (Patient #{patient_id})"
                        
                    # 4. Fallback (just make it look nicer)
                    else:
                        description = filename.replace("_", " ").replace("-", " ").split(".")[0].title()
                    # ----------------------------------

                    img = process_image(img_path)
                    if img:
                        vec = model.encode_image(img)
                        # We now save the CLEAN description, not the filename
                        rag.add_case(vec, description, metadata={"filename": filename})
                        count += 1
                    
                    progress_bar.progress((i + 1) / total_imgs)
                except Exception as e:
                    st.error(f"Failed to index {filename}: {e}")
            
            st.success(f"Database updated! {count} cases are now neatly labeled.")

# --- MAIN INTERFACE ---
st.title("🩺 MediSight: Zero-Shot Diagnostic Assistant")
st.markdown("Upload a medical scan to analyze it against **new** or **rare** conditions instantly.")

col1, col2 = st.columns([1, 1])

# --- COLUMN 1: INPUT & ANALYSIS ---
with col1:
    st.header("1. Patient Scan")
    uploaded_file = st.file_uploader("Upload X-Ray / MRI", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Display the uploaded image
        patient_image = process_image(uploaded_file)
        if patient_image:
            st.image(patient_image, caption="Current Patient Scan", use_column_width=True)

            # Run Analysis Button
            analyze_btn = st.button("🔍 Run Diagnostic Analysis", type="primary")

# --- COLUMN 2: RESULTS ---
with col2:
    st.header("2. AI Findings")
    
    if uploaded_file and analyze_btn:
        # A. ZERO-SHOT DIAGNOSIS
        st.subheader("📊 Probability Analysis (Zero-Shot)")
        
        with st.spinner("Analyzing visual patterns..."):
            # 1. Convert Patient Image to Vector
            patient_vector = model.encode_image(patient_image)
            
            # --- FIX: BETTER PROMPT ENGINEERING ---
            # Instead of just "Pneumonia", we send "A chest X-ray showing Pneumonia"
            # This helps the model understand the context better.
            
            formatted_conditions = [f"A chest X-ray showing {c}" for c in condition_list]
            
            # 2. Encode the formatted text
            text_vectors = model.encode_text(formatted_conditions)
            
            # 3. Compare
            probs = model.compute_similarity(patient_vector, text_vectors)
            
            # Display Results
            results = list(zip(condition_list, probs))
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 4. Smart Display Logic
            top_match = results[0][0]
            top_score = results[0][1]
            
            if top_score < 0.2:
                st.warning("⚠️ Low Confidence: The model is unsure. Rely on RAG results.")
            
            for condition, score in results:
                # If the score is tiny, don't show it to avoid confusion
                if score > 0.01: 
                    st.markdown(f"**{condition}**")
                    
                    # Green bar for Healthy, Red for Disease
                    if "Healthy" in condition or "No Findings" in condition:
                        st.progress(float(score)) # Streamlit default is blue, good enough
                    else:
                        st.progress(float(score))
                        
                    st.caption(f"Confidence: {score*100:.2f}%")

        st.divider()

        # B. RAG RETRIEVAL (EVIDENCE)
        st.subheader("📚 Similar Historical Cases (RAG)")
        
        with st.spinner("Searching hospital records..."):
            # Search the database using the patient's vector
            similar_cases = rag.search_similar(patient_vector, n_results=3)
            
            if not similar_cases or similar_cases[0] == "Database is empty. Add trusted cases to data/reference_images first.":
                st.warning("⚠️ No historical matches found. (Did you click 'Update Medical Database'?)")
            else:
                for i, case_desc in enumerate(similar_cases):
                    st.info(f"**Case Match #{i+1}:** {case_desc}")