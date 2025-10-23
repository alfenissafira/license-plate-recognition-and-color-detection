import streamlit as st
import numpy as np
import os
from PIL import Image
import base64
from io import BytesIO
from yolo_predictions import YOLO_Pred
import webbrowser
from streamlit_option_menu import option_menu
import requests
import pandas as pd
from zipfile import ZipFile

# Fungsi untuk mendapatkan path folder berdasarkan pilihan pengguna
def get_folder_path(selected_folder):
    base_folder = "E:/Dataset"  # Ganti dengan path folder utama
    return os.path.join(base_folder, selected_folder)
# Fungsi untuk mendapatkan nama-nama folder
def get_folder_names():
    return ['Dataset_Plat']

# Fungsi untuk mendapatkan data nama file dan format dalam tabel
def get_table_data(folder_path, valid_image_extensions):
    image_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)) and any(file.lower().endswith(ext) for ext in valid_image_extensions)]
    table_data = {"Nama File": [file.split(".")[0] for file in image_files],
                  "Format": [file.split(".")[-1] for file in image_files]}
    return pd.DataFrame(table_data)

# Fungsi untuk mendapatkan data nama file dan format dalam tabel
def get_table_data(folder_path, valid_image_extensions):
    image_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)) and any(file.lower().endswith(ext) for ext in valid_image_extensions)]
    table_data = {"Nama File": [file.split(".")[0] for file in image_files],
                  "Format": [file.split(".")[-1] for file in image_files]}
    return pd.DataFrame(table_data)
# Opsi di sidebar
with st.sidebar:
    selected = option_menu(
        "Menu",
        ['Home', 'Upload Dataset', 'Proses Training',  'Prediksi' ],
        icons=['cloud-upload', 'database-fill-check', 'images',  'gear', 'kanban' ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "padding-top": "0px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
        }
    )

if selected == "Home":
    st.title("Upload Dataset")
    
    # Allow users to upload a zip file containing images
    uploaded_folder_zip = st.file_uploader("Pilih folder (zip) yang berisi gambar...", type=["zip"])

    # Input untuk nama folder
    default_folder_name = uploaded_folder_zip.name.split(".")[0] if uploaded_folder_zip else ""
    folder_name = st.text_input("Nama Folder (tanpa ekstensi)", default_folder_name)

    if uploaded_folder_zip:
        # Extract the contents of the uploaded zip file
        folder_path = get_folder_path(folder_name)
        zip_data = BytesIO(uploaded_folder_zip.read())
        with ZipFile(zip_data, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        st.success(f"Gambar berhasil diunggah dan disimpan di folder {folder_name}")

elif selected == "Upload Dataset":
    st.title("Upload Dataset")

    # Tampilkan daftar gambar yang dapat diupdate atau dihapus
    update_delete_folder = st.selectbox("Pilih Folder untuk Update dan Delete", get_folder_names())

    folder_path_for_update_delete = get_folder_path(update_delete_folder)
    
    # Menyertakan valid_image_extensions untuk mendapatkan daftar file gambar
    valid_image_extensions = ['.jpg', '.jpeg', '.png']
    all_image_files = [file for file in os.listdir(folder_path_for_update_delete) if os.path.isfile(os.path.join(folder_path_for_update_delete, file)) and any(file.lower().endswith(ext) for ext in valid_image_extensions)]

    # Pilihan gambar untuk update atau delete
    page_number = st.number_input("Pilih Halaman", min_value=1, max_value=(len(all_image_files) // 10) + 1, value=1, step=1)

    start_index = (page_number - 1) * 10
    end_index = min(page_number * 10, len(all_image_files))

    table_data = {"Nama File": [file.split(".")[0] for file in all_image_files[start_index:end_index]],
                  "Format": [file.split(".")[-1] for file in all_image_files[start_index:end_index]]}

    # Tampilkan data nama file dan format dalam tabel dengan indeks dimulai dari 1
    st.table(pd.DataFrame(table_data).reset_index(drop=True))

    # Pilihan aksi: Update atau Delete
    action = st.selectbox("Pilih Aksi", ["Update", "Delete"])

    if action == "Update":
        st.text("Form Update Gambar")
        # Tambahkan form atau fungsi update di sini

        # Pilih gambar untuk diupdate
        selected_image_for_update = st.selectbox("Pilih Gambar untuk Update", table_data["Nama File"])

        # Form untuk mengupdate nama file
        new_file_name = st.text_input("Nama File Baru (tanpa ekstensi)", selected_image_for_update)

        # Tampilkan tombol update
        update_button = st.button("Update Gambar")

        if update_button:
            # Lakukan pembaruan nama file
            old_file_path = os.path.join(folder_path_for_update_delete, f"{selected_image_for_update}.jpg")
            new_file_path = os.path.join(folder_path_for_update_delete, f"{new_file_name}.jpg")

            os.rename(old_file_path, new_file_path)
            st.success(f"Gambar {selected_image_for_update} berhasil diupdate menjadi {new_file_name}.jpg")

    elif action == "Delete":
        # Pilihan gambar untuk update atau delete
        selected_image_for_update_delete = st.selectbox("Pilih Gambar untuk Update atau Delete", table_data["Nama File"])

        # Tampilkan tombol delete
        delete_button = st.button("Hapus Gambar Terpilih")

        if delete_button:
            # Hapus gambar jika tombol di tekan
            os.remove(os.path.join(folder_path_for_update_delete, selected_image_for_update_delete + ".jpg"))
            st.success(f"Gambar {selected_image_for_update_delete} berhasil dihapus.")

elif selected == "Proses Training":
    st.title("Training Citra Kendaraan")
    train_button = st.button("Mulai Training")

    if train_button:
        # Panggil fungsi pelatihan dari training.py
        colab_url = "https://colab.research.google.com/drive/1d9x0Y0xzJqHSeYzhaXQZ3khu7DxpUgMy#scrollTo=VzL56RhsvFkb"
        webbrowser.open_new_tab(colab_url)

        st.success("Silahkan beralih ke Google Colab")

elif selected == "Prediksi":
    st.title("Prediksi")
    
    # Load YOLO model
    with st.spinner('Please wait while your model is being loaded'):
        try:
            yolo = YOLO_Pred(onnx_model='yolo_v5/YOLO_Model/weights/best.onnx',
                             data_yaml='yolo_v5/data.yaml')
        except Exception as e:
            st.error('Error loading YOLO model. Please check your file paths.')
            raise e
    
    st.write('Please upload image(s) or a folder containing image(s) of license plates for detection and text extraction')

    def upload_file_or_folder():
        uploaded_objects = st.file_uploader(label='Upload Image or Folder', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        if uploaded_objects is not None and len(uploaded_objects) > 0:
            if isinstance(uploaded_objects[0], BytesIO):  # Uploaded object is a single image
                try:
                    images = [Image.open(obj) for obj in uploaded_objects]
                except Exception as e:
                    st.error('Error opening image. Please check your files.')
                    return None

                size_mb = sum(obj.getbuffer().nbytes for obj in uploaded_objects) / (1024**2)
                file_details = {
                    'filename': 'Uploaded_Images',
                    'filetype': 'Images',
                    'filesize': f'{size_mb:.2f} MB'
                }

                return {"files": uploaded_objects, "details": file_details, "objects": images}

            elif isinstance(uploaded_objects[0], str):  # Uploaded object is a folder
                folder = uploaded_objects[0]
                file_list = [os.path.join(folder, file) for file in os.listdir(folder)]
                image_files = [file for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if not image_files:
                    st.error('No valid image files found in the folder. Upload only, png, jpg, jpeg files.')
                    return None

                return {"folder": folder, "image_files": image_files}

        return None


    def main():
        object = upload_file_or_folder()

        if object:
            if "files" in object:
                process_images(object)
            elif "folder" in object:
                process_images_in_folder(object)


    def process_images(object):
        results = []
        files = object["files"]
        images = object["objects"]

        for file, image in zip(files, images):
            result = process_single_image({"file": file, "details": object["details"], "object": image})
            results.append(result)

        save_path = f'{object["details"]["filename"]}_predictions.txt'
        with open(save_path, 'w') as file:
            for result in results:
                file.write(result)
                file.write('\n\n')

        st.success(f'All predictions saved to {save_path}')
        st.markdown(get_download_link(save_path), unsafe_allow_html=True)


    def process_single_image(object):
        prediction = False
        image_obj = object['object']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info('Preview of Image')
            st.image(image_obj, use_column_width=True)  # Ensure that the image fits within the column width

        with col2:
            try:
                with st.spinner("""Running your model"""):
                    # Convert to array
                    image_array = np.array(image_obj)
                    pred_img, license_text = yolo.predictions(image_array)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True
            except Exception as e:
                st.error('Error processing image. Please check your image file.')
                raise e

            if prediction:
                st.info("Predicted Image")
                # Convert predicted image to RGB mode
                pred_img_obj = pred_img_obj.convert('RGB')
                st.image(pred_img_obj, use_column_width=True)

        with col3:
            if prediction:
                st.info("Predicted OCR Text Extraction")
                st.subheader(license_text)

                result = f"Predictions for {object['details']['filename']}:\n" \
                        f"License Text: {license_text}"

                # Save predictions to a text file
                return result


    def process_images_in_folder(object):
        results = []
        folder = object["folder"]
        image_files = object["image_files"]

        for image_file in image_files:
            try:
                image_obj = Image.open(image_file)
            except Exception as e:
                st.error(f'Error opening image {image_file}. Please check your file.')
                continue

            col1, col2, col3 = st.columns(3)

            with col1:
                st.info(f'Preview of Image: {os.path.basename(image_file)}')
                st.image(image_obj, use_column_width=True)  # Ensure that the image fits within the column width

            with col2:
                try:
                    with st.spinner("Running your model"):
                        image_array = np.array(image_obj)
                        pred_img, license_text = yolo.predictions(image_array)
                        pred_img_obj = Image.fromarray(pred_img)
                except Exception as e:
                    st.error(f'Error processing image {image_file}. Please check your image file.')
                    continue

                st.info(f"Predicted Image for {os.path.basename(image_file)}")
                pred_img_obj = pred_img_obj.convert('RGB')
                st.image(pred_img_obj, use_column_width=True)

            with col3:
                st.info(f"Predicted OCR Text Extraction for {os.path.basename(image_file)}")
                st.subheader(license_text)

                result = f"Predictions for {os.path.basename(image_file)}:\n" \
                        f"License Text: {license_text}"
                results.append(result)

        save_path = f'{object["folder"]}_predictions.txt'
        with open(save_path, 'w') as file:
            for result in results:
                file.write(result)
                file.write('\n\n')

        st.success(f'All predictions saved to {save_path}')
        st.markdown(get_download_link(save_path), unsafe_allow_html=True)


    def get_download_link(file_path):
        with open(file_path, 'r') as file:
            text = file.read()
        b64 = base64.b64encode(text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(file_path)}">Download All Results</a>'
        return href
    
    if __name__ == "__main__":
        main()