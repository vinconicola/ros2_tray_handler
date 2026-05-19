import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "src/training/kpconv_semantickitti.yaml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.KPFCNN(**cfg.model)  # ← fixed: match model to config

cfg.dataset['dataset_path'] = "/home/nicola/ros2_ws/data"
dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "KPFCNN_SemanticKITTI_torch/checkpoint/ckpt_00300.pth"  # ← also fixed double "logs/"

pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("test")
data = test_split.get_data(0)

data['feat'] = None
result = pipeline.run_inference(data)
pipeline.run_test()