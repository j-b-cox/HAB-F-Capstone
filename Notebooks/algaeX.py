
import os
import streamlit as st
from pathlib import Path
import algaexutils
from algaexutils import plot_chla_image_from_dataset
from algaexutils import plot_chla_true_pred_dataset
import tarfile
import tempfile



st.title("AlgaeX")

st.write("Prediction Engine trained on data from Jan 2017 through Oct 2023")

# Storage to save uploaded datasets
UPLOAD_DIR = "uploaded_dataset"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Refresh button
if st.button("🔄 Refresh Dataset List"):
    st.session_state["refresh_datasets"] = True

# Initial setup of session state
if "refresh_datasets" not in st.session_state:
    st.session_state["refresh_datasets"] = True

# Update file list only if refresh is triggered
if st.session_state["refresh_datasets"]:
    uploaded_sets = list(Path(UPLOAD_DIR).glob("*"))
    st.session_state["uploaded_sets"] = uploaded_sets
    st.session_state["refresh_datasets"] = False
else:
    uploaded_sets = st.session_state.get("uploaded_sets", [])

#uploaded_sets = list(Path(UPLOAD_DIR).glob("*"))

selected_file = st.selectbox(
    "Select a dataset to view its content",
    [file.name for file in uploaded_sets] if uploaded_sets else []
)
st.write(f"Total Datasets: {len(uploaded_sets)}")

if selected_file:
    file_path = Path(UPLOAD_DIR) / selected_file

    # Set your target directory
    directory = file_path

# Prompt the user
choice = st.radio("Display all frames in dataset?", ("No", "Yes"))

if choice == "Yes":
    plotimg = plot_chla_image_from_dataset(directory, min_thresh=2.5, max_thresh=7.0)
    
    with open(plotimg, "rb") as f:
        st.image(f.read(), caption=f'Sequence from {selected_file}', use_container_width=True)
else:
    pass

# Prompt the user
choice = st.radio("Do you want to see the predicted and true output?", ("No", "Yes"))

# Conditionally show content
if choice == "Yes":
    #st.subheader("Here are the extra options!")
    #st.write("You selected **Yes**. Here's some more content...")
    plotimg, ssim, psnr = plot_chla_true_pred_dataset(directory, min_thresh=2.5, max_thresh=7.0)
    
    with open(plotimg, "rb") as f:
        st.image(f.read(), caption=f'Predicted and True from {selected_file}', use_container_width=True)
    
    st.markdown("#### Quantitative Metrics")
    st.write(f"Structural similarity Index Measure: {ssim}")
    st.write(f"Peak Signal to Noise Ratio: {psnr}")

else:
    pass
    #st.write("You selected **No**. Nothing more to show.")



import tarfile
import os

st.subheader("Upload new dataset")
# File uploader
uploaded_file = st.file_uploader("Upload a .tar.gz or .tar dataset", type=["tar", "gz"])

# Specify your desired output folder
output_dir = f"{UPLOAD_DIR}/"

if uploaded_file is not None:
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the uploaded file to disk
    tar_path = os.path.join(output_dir, uploaded_file.name)
    with open(tar_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract the tar file
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            # List file names
            file_names = tar.getnames()
            st.write("Files to be extracted:")
            for name in file_names:
                st.write(name)
            
            tar.extractall(path=output_dir)
            st.success(f"File extracted to: {output_dir}")

            # List extracted files
            #st.subheader("Extracted Files:")
            #for root, dirs, files in os.walk(output_dir):
            #    for file in files:
            #        st.write(os.path.join(root, file))
    except tarfile.TarError as e:
        st.error(f"Error extracting tar file: {e}")
    print(f'Removing {tar_path}')
    os.remove(tar_path)

#uploaded_file = st.file_uploader(
#    "Choose a dataset directory to upload",
#    type=["png"]
#)

#if uploaded_file is not None:
#    file_path = Path(UPLOAD_DIR) / uploaded_file.name
#    with open(file_path, "wb") as f:
#        f.write(uploaded_file.getbuffer())
#    st.success(f"File saved to {file_path}")
#    st.image(file_path, caption="Uploaded Image", use_container_width=True)# Directory to save uploaded images
#UPLOAD_DIR = "uploaded_images"
#os.makedirs(UPLOAD_DIR, exist_ok=True)

