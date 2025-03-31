# Template

python3 ray_preprocess.py --pipeline-config image_gen.io.datasets.coca_001 --pipeline-module image_gen  --conda-env CIFAR --remote-host ray://localhost:10001 

# Training
```
python3 ray_train.py --pipeline-config image_gen.pipeline_configs.simple_cnn_arch --pipeline-module image_gen --remote-host ray://localhost:10001 --conda-env CIFAR
```

python3 ray_train.py --pipeline-config image_gen.pipeline_configs.gnn_arch_fmnist --pipeline-module image_gen --conda-env CIFAR