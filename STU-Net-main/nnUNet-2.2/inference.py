import torch
import numpy
import os
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    torch.serialization.add_safe_globals([
        numpy._core.multiarray.scalar,
        numpy.dtype
    ])
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        allow_tqdm=True
    )

    # Apuntamos a tus resultados del Fine-Tuning
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset501_AbdomenTumor/STUNetTrainer_small_ft__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth', # O 'checkpoint_best.pth' según cómo lo guardaste en v2
    )

    # Corremos la inferencia de la carpeta completa
    predictor.predict_from_files(
        join(nnUNet_raw, 'Dataset501_AbdomenTumor/imagesTs'),
        '/home/chloe/Desktop/DL-STU-net-fine-tuning/STU-Net-main/result',
        save_probabilities=False, 
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )

if __name__=="__main__":
    main()