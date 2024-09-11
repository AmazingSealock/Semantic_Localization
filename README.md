


# Inference

1. Prepare an image and a pcd (point cloud) file.
2. Modify the path in the inference file
   ```python
    if __name__ == "__main__":
        trainer = MirrorTrainer(env_type=env,
                                max_epochs=max_epoch,
                                batch_size=batch_size,
                                device=device,
                                logdir=logdir,
                                val_every=val_every,
                                num_gpus=num_gpus,
                                master_port=17752,
                                training_script=__file__)
        
        trainer.run_single_file("/home/xingzhaohu/point_cloud_project/2024_0624_data_test/img/KIN_3828.png", 
                                "/home/xingzhaohu/point_cloud_project/2024_0624_data_test/pcd/KIN_3828.pcd")
    
   ```
3. Run the inference code 
   ```python
   python test_point_cloud_0620_single.py
   ```