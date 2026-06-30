import os
import json
import shutil
import numpy as np
import nibabel as nib

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
# Te sugiero crear una carpeta nueva para no mezclar este experimento con el otro
INPUT_DIR = "C:/Users/matij/Desktop/STU-Net-main/STU-Net-main/data/final-formatted/images/abdomen"
OUTPUT_DIR = "C:/Users/matij/Desktop/STU-Net-main/STU-Net-main/data/nnUNet_raw/Dataset501_AbdomenTumor"

# ==========================================
# 2. EL MAPEO CLÍNICO (T, N, M)
# ==========================================
# Asignamos IDs basados estrictamente en el origen del tumor
ID_TUMOR_PRIMARIO = 1  # 't,'
ID_METASTASIS = 2      # 'm,'
ID_ADENOPATIA = 3      # 'n,'

def apply_windowing(image, L=40, W=350):
    """Aplica el windowing de Hounsfield Units para resaltar los tejidos."""
    min_val = L - (W / 2)
    max_val = L + (W / 2)
    windowed = np.clip(image, min_val, max_val)
    return (windowed - min_val) / (max_val - min_val)

def procesar_split(split_name):
    """Procesa la carpeta train o test"""
    in_images = os.path.join(INPUT_DIR, split_name, "images")
    in_masks = os.path.join(INPUT_DIR, split_name, "masks")
    in_labels = os.path.join(INPUT_DIR, split_name, "labels")
    
    # Dónde guardamos (nnU-Net usa 'Tr' para Train y 'Ts' para Test)
    out_images = os.path.join(OUTPUT_DIR, f"images{split_name.capitalize()[:2]}") 
    out_masks = os.path.join(OUTPUT_DIR, f"labels{split_name.capitalize()[:2]}")  

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    # Iteramos sobre todos los JSONs
    for json_file in os.listdir(in_labels):
        if not json_file.endswith(".json"): continue
        
        base_name = json_file.replace(".json", "") # Ej: 001_1.3.12...
        
        # 1. Leer el JSON del paciente
        with open(os.path.join(in_labels, json_file), 'r') as f:
            local_mapping = json.load(f)
            
        # 2. Cargar la máscara NIfTI original
        mask_path = os.path.join(in_masks, f"{base_name}.nii.gz")
        img_nifti = nib.load(mask_path)
        mask_data = img_nifti.get_fdata()
        
        # Crear una máscara vacía (llena de ceros) del mismo tamaño
        new_mask_data = np.zeros_like(mask_data)

        # 3. Traducir cualquier tumor a ID 1, fondo a 0
        # mask_data contiene los IDs locales de tu segmentación original
        new_mask_data = np.zeros_like(mask_data, dtype=np.uint8)
        
        # Cualquier valor que no sea el fondo (0) en la máscara original, se convierte en 1
        new_mask_data[mask_data > 0] = 1
                
        # 4. Guardar la nueva máscara en la carpeta de nnU-Net
        new_img_nifti = nib.Nifti1Image(new_mask_data.astype(np.uint8), img_nifti.affine, img_nifti.header)
        out_mask_path = os.path.join(out_masks, f"{base_name}.nii.gz")
        nib.save(new_img_nifti, out_mask_path)

        # 5. Copiar, procesar y renombrar la tomografía (Añadir _0000.nii.gz)
        in_image_path = os.path.join(in_images, f"{base_name}.nii.gz")
        out_image_path = os.path.join(out_images, f"{base_name}_0000.nii.gz")
        
        # Cargar la imagen original
        img_nifti_obj = nib.load(in_image_path)
        images_data = img_nifti_obj.get_fdata().astype(np.float32)

        # El resultado se guarda en 'processed_data'
        processed_data = apply_windowing(images_data)
        
        # Crear un nuevo objeto NIfTI con los datos normalizados
        # Es vital mantener el affine y header original para que no pierda la geolocalización médica
        new_img_nifti = nib.Nifti1Image(processed_data.astype(np.float32), 
                                        img_nifti_obj.affine, 
                                        img_nifti_obj.header)
        
        # Guardar
        nib.save(new_img_nifti, out_image_path)      
        print(f"Procesado exitosamente: {base_name}")

# Ejecutar para train y test
print("Procesando Entrenamiento...")
procesar_split("train")
print("Procesando Test...")
procesar_split("test")