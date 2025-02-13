import torch
import re
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

class MolmoModel:
    def __init__(self, model_path="../Molmo-7B-O-0924"):
        """
        Load the Molmo model and processor from the specified path.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        ).to(self.device)

        self.model.eval()
        print("✅ Model and processor loaded successfully!")

    def generate_response(self, image_path: str, text_prompt: str):
        """
        Generate a response from the model given an image and a text prompt.
        
        :param image_path: Path to the JPG image.
        :param text_prompt: Text prompt for the model.
        :return: Generated text response.
        """
        # Load and process image
        image = Image.open(image_path).convert("RGB")

        # Process inputs
        inputs = self.processor.process(
            images=[image],
            text=text_prompt
        )

        # Move inputs to model's device
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # Decode generated text
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def parse_coordinates(self, output_text: str):
        """
        Parse the output text for coordinates in the format:
        <point x="54.3" y="53.7" alt="teapot">teapot</point>
        
        :param output_text: The text generated by the model.
        :return: Tuple (x, y) as floats if found; otherwise None.
        """
        # Use regex to extract x and y values from the output
        match = re.search(r'<point\s+.*?x="([\d\.]+)"\s+y="([\d\.]+)".*?>', output_text)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            return (x, y)
        else:
            return None
