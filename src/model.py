# src/model_loader.py
import torch
import open_clip
from PIL import Image

class MedicalModel:
    def __init__(self):
        """
        Initializes the BiomedCLIP model. 
        This is heavy (GBs), so we only want to do this once.
        """
        print("Loading BiomedCLIP model... (This may take up to 30 seconds)")
        
        # Check if GPU is available for faster processing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model and the image pre-processor (transforms)
        # We use 'create_model_and_transforms' to get the validation transform (preprocess_val)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
            device=self.device
        )
        
        # Load the text tokenizer
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # Set model to evaluation mode (we are not training it)
        self.model.eval()
        print(f"Model loaded successfully on {self.device.upper()}")

    def encode_image(self, image: Image.Image):
        """
        Converts a PIL Image into a normalized vector embedding.
        """
        # 1. Preprocess image (resize, normalize) and move to GPU/CPU
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 2. Run through the Vision Transformer
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            
            # 3. Normalize the vector (Crucial for Cosine Similarity!)
            features /= features.norm(dim=-1, keepdim=True)
            
        return features

    def encode_text(self, text_list: list):
        """
        Converts a list of disease names into vector embeddings.
        """
        # 1. Tokenize text
        text_tokens = self.tokenizer(text_list).to(self.device)
        
        # 2. Run through PubMedBERT
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
            # 3. Normalize
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def compute_similarity(self, image_features, text_features):
        """
        Calculates the probability of the image matching each text prompt.
        """
        # Matrix multiplication to find cosine similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity.cpu().numpy()[0]