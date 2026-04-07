from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImage

model = ZImage(
    model_config=ModelConfig.z_image(),
    model_path="Tongyi-MAI/Z-Image",
)
image = model.generate_image(
    seed=42,
    prompt="Two smiling friends posing for a casual indoor portrait, soft natural light, shallow depth of field.",
    num_inference_steps=50,
    width=720,
    height=1280,
    guidance=4.0,
    negative_prompt="",
)
image.save("z_image_base.png")