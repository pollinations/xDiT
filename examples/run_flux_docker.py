import logging
import time
import torch
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
import os

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    
    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path="FLUX.1-schnell",
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    pipe = pipe.to("cuda")
    
    # Create results directory if it doesn't exist
    os.makedirs("/outputs", exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    # Get prompt from environment variable or use default
    prompt = os.getenv('FLUX_PROMPT', "a beautiful mountain landscape at sunset, high quality, detailed")
    
    output = pipe(
        height=int(os.getenv('FLUX_HEIGHT', '1024')),
        width=int(os.getenv('FLUX_WIDTH', '1024')),
        prompt=prompt,
        num_inference_steps=int(os.getenv('FLUX_STEPS', '50')),
        output_type="pil",
        guidance_scale=float(os.getenv('FLUX_GUIDANCE_SCALE', '0.0')),
        generator=torch.Generator(device="cuda").manual_seed(int(os.getenv('FLUX_SEED', '42'))),
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"Inference time: {elapsed_time:.2f} seconds")
    print(f"Peak GPU memory: {peak_memory/1e9:.2f} GB")
    print(f"Generated prompt: {prompt}")

    # Save the generated image
    output_path = os.path.join('/outputs', f'flux_output_{int(time.time())}.png')
    output.images[0].save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()
