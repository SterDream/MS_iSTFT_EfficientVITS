import torch
from text.phonemize.symbols import symbols
from models import EFTS2
import utils

def load_pretrained_mb_istft_vits(checkpoint_path, config_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # mb_istft_vitsの保存形式に合わせて取り出す
    if 'model' in ckpt:
        pretrained_dict = ckpt['model']
    else:
        pretrained_dict = ckpt

    hps = utils.get_hparams_from_file(config_path)
    model = EFTS2(
      len(symbols), 
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model
    )

    model_dict = model.state_dict()
    matched, skipped_shape, skipped_missing = [], [], []
    new_state_dict = {}

    for k, v in model_dict.items():
        if k in pretrained_dict:
            if pretrained_dict[k].size() == v.size():
                new_state_dict[k] = pretrained_dict[k]
                matched.append(k)
            else:
                new_state_dict[k] = v  # ランダム初期化のまま
                skipped_shape.append(f"{k}: ckpt={pretrained_dict[k].size()} vs model={v.size()}")
        else:
            new_state_dict[k] = v  # ランダム初期化のまま
            skipped_missing.append(k)

    model.load_state_dict(new_state_dict)
    print(f"\n流用成功: {len(matched)} layers")
    print(f"スキップ: {len(skipped_shape)} layers")

    for s in skipped_shape:
        print(f"   {s}")

    print(f"新規初期化: {len(skipped_missing)} layers")
    
    for s in skipped_missing:
        print(f"   {s}")
    return model


model = load_pretrained_mb_istft_vits(
    config_path="./finetune_model_dir/pretrained_MS-iSTFT-VITS_ddp-config.json",
    checkpoint_path="./finetune_model_dir/pretrained_MS-iSTFT-VITS_ddp.pth"
)
torch.save(model.state_dict(), "./finetune_model_dir/pretrained_MS-iSTFT-VITS_ddp_efficient.pth")
print("saved!")