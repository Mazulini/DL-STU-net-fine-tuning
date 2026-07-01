import os
from batchgenerators.utilities.file_and_folder_operations import join

# 1. Aseguramos que la ruta base de nnUNet esté accesible en el path si es necesario
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2

def main():
    # 2. Definición de rutas absolutas
    gt_folder = '/home/chloe/Desktop/DL-STU-net-fine-tuning/STU-Net-main/data/nnUNet_raw/Dataset501_AbdomenTumor/labelsTs'
    pred_folder = '/home/chloe/Desktop/DL-STU-net-fine-tuning/STU-Net-main/result'
    
    dataset_json_file = '/home/chloe/Desktop/DL-STU-net-fine-tuning/STU-Net-main/data/nnUNet_raw/Dataset501_AbdomenTumor/dataset.json'
    
    # Apuntamos a los planes que definieron el entrenamiento
    plans_file = '/home/chloe/Desktop/DL-STU-net-fine-tuning/STU-Net-main/data/nnUNet_results/Dataset501_AbdomenTumor/STUNetTrainer_small_ft__nnUNetPlans__3d_fullres/plans.json'
    
    # Destino del reporte final
    output_file = join(pred_folder, 'summary.json')

    print("Iniciando la evaluación cuantitativa de las inferencias...")
    
    # 3. Llamada directa a la función nativa de nnU-Net
    compute_metrics_on_folder2(
        folder_ref=gt_folder,
        folder_pred=pred_folder,
        dataset_json_file=dataset_json_file,
        plans_file=plans_file,
        output_file=output_file,
        num_processes=1,       # Evitamos problemas de RAM / OOM Killer
        chill=False            # Se asegura de que no falte ningún caso entre el GT y tus predicciones
    )
    
    print(f"¡Evaluación completada con éxito! Resultados guardados en: {output_file}")

if __name__ == '__main__':
    main()