import torch
ckpt = torch.load("./finetune_model_dir/D_150000.pth", map_location="cpu")

# check the model's keys
if "model" in ckpt:
    keys = list(ckpt["model"].keys())
else:
    keys = list(ckpt.keys())

for k in keys:
    print(k)