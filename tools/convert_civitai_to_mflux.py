import torch
import safetensors
from diffusers import ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from mlx.core import bfloat16
from transformers import Qwen3Model,Qwen3ForCausalLM

civitaiUnetFile="/Users/Gang.Jin/EnjoyingGreen/ComfyUI/models/unet/zImage/DarkBeastZ6-BlitZ-F32-ComfyUI.safetensors"
orgUnetDit=safetensors.torch.load_file(civitaiUnetFile)
corrected_dict = { k.replace('model.diffusion_model.', ''): v for k, v in orgUnetDit.items() }

civitaiUnet = ZImageTransformer2DModel.from_single_file(corrected_dict)

#text_encoder=Qwen3Model.from_pretrained("huihui-ai/Qwen3-4B-abliterated")

pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo",transformer=None)
pipe.transformer=civitaiUnet
#pipe.text_encoder=text_encoder
"""
redcraftRedzimageUpdatedDEC03_redzimage15AIO.safetensors
/Users/Gang.Jin/EnjoyingGreen/ComfyUI/models/unet/zImage/unstableRevolution_Bf16.safetensors

Tongyi-MAI/Z-Image-Turbo
print(f"\nPipeline components keys: {pipe.components.keys()}")
for k,v in pipe.components.items():
    print(f"\n{k},{v}")
"""
#pipe.to("mps")

# Optionally, set the attention backend to flash-attn 2 or 3, default is SDPA in PyTorch.
# (1) Use flash attention 2
# pipe.transformer.set_attention_backend("flash")
# (2) Use flash attention 3
# pipe.transformer.set_attention_backend("_flash_3")

prompt = """一个中国女人，25岁，凌乱中短发，身高172厘米，超模身材，胸部硕大。
裸体，乳头和私处暴露，肩披白色毛巾，腕带健身手环，脚着白色运动鞋。
流汗，皮肤湿润。坐在长椅上休息，前倾放松，双腿叉开。右手拿着一瓶矿泉水，左手微抬，左手手指指向身后的镜子。女人的脚边有几支小哑铃，背后的镜子映照出她的背影。
场景为白天，室内，健身房，四周是各种健身器材。
社交媒体照片，色调平衡，光照凸显人物身材。"""
"""
image = pipe(
    prompt,
    height=1280,
    width=720,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("mps").manual_seed(456),
).images[0]
image.save("zimage.png")
"""
pipe.save_pretrained("/Users/Gang.Jin/EnjoyingGreen/darkBeast/Z-Image-Turbo")
