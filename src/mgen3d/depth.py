import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, DPTForDepthEstimation
import yaml


class DPT(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device

        self.image_processor = AutoImageProcessor.from_pretrained(
            config["components"]["depth_key"]
        )
        self.depth_model = DPTForDepthEstimation.from_pretrained(
            config["components"]["depth_key"]
        )
        
    def preprocess_image(self, image, scale=1.0):
        # Define the transformations
        transform = transforms.Compose(
            [
                transforms.Resize((int(scale * image.size[1]), 
                                   int(scale * image.size[0]))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor_image = transform(image)
        return tensor_image

    def get_depth(self, image, scale=1.0):
        # prepare image for the model
        inputs = self.image_processor(images=image, return_tensors="pt")

        # forward pass
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[1:],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return prediction


def main():
    import sys
    import absl.flags as flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string("config_path", None, "Path to the config file")
    flags.mark_flag_as_required("config_path")
    flags.FLAGS(sys.argv)

    with open(FLAGS.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    output_path = config["workspace"]["path"]
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(config["workspace"]["input_path"])
    
    scale = 1.0
    
    depth_model = DPT(config, device)
    image = depth_model.preprocess_image(image, scale)
    depth_prediction = depth_model.get_depth(image)
    output = depth_prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    # Save the PIL image
    depth.save(os.path.join(output_path, "depth_test.png"))


if __name__ == "__main__":
    main()
