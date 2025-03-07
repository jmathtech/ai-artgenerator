import torch
from diffusers import StableDiffusionPipeline

def generate_art(prompt, output_path="output.png", steps=50, guidance_scale=7.5, seed=None):
    """
    Generates AI art using Stable Diffusion.
    
    Args:
        prompt (str): The description of the image to generate.
        output_path (str): File path to save the generated image.
        steps (int): Number of inference steps.
        guidance_scale (float): Affects creativity. Higher values focus on the prompt more.
        seed (int, optional): Seed for reproducibility.
    
    Returns:
        None
    """
    # Set the random seed for reproducibility
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None

    # Load the pre-trained Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    # Generate the image
    print("Generating art...")
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator).images[0]

    # Save the image
    image.save(output_path)
    print(f"Art saved to {output_path}")

if __name__ == "__main__":
    # Input parameters
    user_prompt = input("Enter a description of the art you want to generate: ")
    generate_art(
        prompt=user_prompt,
        output_path="generated_art.png",
        steps=50,
        guidance_scale=7.5,
        seed=42
    )
