import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import numpy as np
import re

class BeakerVolumePredictor:
    """Inference class for beaker volume prediction"""
    
    def __init__(self, model_path, model_type="qwen2vl"):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_type} model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        if model_type == "qwen2vl":
            base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
        elif model_type == "florence2":
            base_model_name = "microsoft/Florence-2-base"
        
        # Load base model
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def extract_volume_from_text(self, text):
        """Extract numerical volume from generated text"""
        matches = re.findall(r'\d+\.?\d*', text)
        if matches:
            try:
                return float(matches[0])
            except:
                return None
        return None
    
    def predict(self, image):
        """Predict volume from beaker image"""
        if image is None:
            return "Please upload an image", None
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Prepare inputs based on model type
        if self.model_type == "qwen2vl":
            prompt = "What is the liquid volume in this beaker in mL? Provide only the numerical value."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
        
        elif self.model_type == "florence2":
            prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            task_prompt = f"{prompt} What is the liquid volume in mL?"
            
            inputs = self.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                do_sample=False
            )
        
        # Decode output
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract volume
        volume = self.extract_volume_from_text(generated_text)
        
        if volume is not None:
            result_text = f"**Predicted Volume:** {volume} mL"
            return result_text, volume
        else:
            return f"**Raw Output:** {generated_text}\n\n*Could not extract numerical volume*", None


def create_demo():
    """Create Gradio interface"""
    
    # Initialize predictors for both models (if available)
    qwen_predictor = None
    florence_predictor = None
    
    try:
        qwen_predictor = BeakerVolumePredictor("qwen2vl_model", "qwen2vl")
    except Exception as e:
        print(f"Could not load Qwen2-VL model: {e}")
    
    try:
        florence_predictor = BeakerVolumePredictor("florence2_model", "florence2")
    except Exception as e:
        print(f"Could not load Florence2 model: {e}")
    
    def predict_qwen(image):
        if qwen_predictor is None:
            return "Qwen2-VL model not loaded", None
        return qwen_predictor.predict(image)
    
    def predict_florence(image):
        if florence_predictor is None:
            return "Florence2 model not loaded", None
        return florence_predictor.predict(image)
    
    # Create Gradio interface
    with gr.Blocks(title="Beaker Volume Prediction", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔬 Beaker Liquid Volume Prediction
            
            Upload an image of a beaker with liquid to predict the volume in milliliters (mL).
            
            **Models Available:**
            - **Qwen2-VL-2B**: Vision-language model fine-tuned for volume prediction
            - **Florence2-Base**: Microsoft's vision foundation model adapted for this task
            
            Both models are trained on 2000 beaker images with normal and cluttered backgrounds.
            """
        )
        
        with gr.Tabs():
            # Qwen2-VL Tab
            with gr.Tab("Qwen2-VL Model"):
                with gr.Row():
                    with gr.Column():
                        qwen_input = gr.Image(type="pil", label="Upload Beaker Image")
                        qwen_btn = gr.Button("Predict Volume", variant="primary")
                    
                    with gr.Column():
                        qwen_output = gr.Markdown(label="Prediction Result")
                        qwen_volume = gr.Number(label="Volume (mL)", precision=2)
                
                qwen_btn.click(
                    fn=predict_qwen,
                    inputs=qwen_input,
                    outputs=[qwen_output, qwen_volume]
                )
                
                gr.Examples(
                    examples=[
                        # Add example image paths if available
                    ],
                    inputs=qwen_input,
                    label="Example Images"
                )
            
            # Florence2 Tab
            with gr.Tab("Florence2 Model"):
                with gr.Row():
                    with gr.Column():
                        florence_input = gr.Image(type="pil", label="Upload Beaker Image")
                        florence_btn = gr.Button("Predict Volume", variant="primary")
                    
                    with gr.Column():
                        florence_output = gr.Markdown(label="Prediction Result")
                        florence_volume = gr.Number(label="Volume (mL)", precision=2)
                
                florence_btn.click(
                    fn=predict_florence,
                    inputs=florence_input,
                    outputs=[florence_output, florence_volume]
                )
                
                gr.Examples(
                    examples=[
                        # Add example image paths if available
                    ],
                    inputs=florence_input,
                    label="Example Images"
                )
            
            # Compare Tab
            with gr.Tab("Compare Both Models"):
                with gr.Row():
                    compare_input = gr.Image(type="pil", label="Upload Beaker Image")
                
                compare_btn = gr.Button("Compare Predictions", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Qwen2-VL Prediction")
                        compare_qwen_output = gr.Markdown()
                        compare_qwen_volume = gr.Number(label="Volume (mL)", precision=2)
                    
                    with gr.Column():
                        gr.Markdown("### Florence2 Prediction")
                        compare_florence_output = gr.Markdown()
                        compare_florence_volume = gr.Number(label="Volume (mL)", precision=2)
                
                def compare_predictions(image):
                    qwen_text, qwen_vol = predict_qwen(image)
                    florence_text, florence_vol = predict_florence(image)
                    return qwen_text, qwen_vol, florence_text, florence_vol
                
                compare_btn.click(
                    fn=compare_predictions,
                    inputs=compare_input,
                    outputs=[compare_qwen_output, compare_qwen_volume, 
                            compare_florence_output, compare_florence_volume]
                )
        
        gr.Markdown(
            """
            ---
            ### 📊 Model Information
            
            **Training Details:**
            - Dataset: 2000 beaker images (70% train, 15% validation, 15% test)
            - Fine-tuning: LoRA (Low-Rank Adaptation)
            - Evaluation Metrics: MAE, RMSE, R²
            
            **Backgrounds:**
            - Normal: Clean, uncluttered backgrounds
            - Cluttered: Complex backgrounds with various objects
            
            **Note:** Model performance may vary based on image quality, lighting, and background complexity.
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
