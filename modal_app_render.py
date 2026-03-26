import modal
import subprocess
import os
import shutil

app = modal.App("hunyuan-world-mirror")

# Secrets
secret = modal.Secret.from_name(
    "r2-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
)
hf_secret = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"]
)

# Volume for temp data
volume = modal.Volume.from_name("temp-data-volume")

# Image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(["git", "wget", "vim", "software-properties-common", "build-essential", "curl", "ca-certificates", "libssl-dev", "libffi-dev", "ninja-build"])
    .run_commands("apt update && apt upgrade -y")
    .apt_install(["python3.10", "python3-pip"])
    .run_commands("curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && rm Miniforge3-Linux-x86_64.sh")
    .run_commands("bash -c 'source /opt/conda/bin/activate && conda create -n hunyuanworld-mirror python=3.10 cmake=3.14.0 -y'")
    .run_commands("bash -c 'source /opt/conda/bin/activate hunyuanworld-mirror && pip install --upgrade pip setuptools wheel ninja'")
    .run_commands("bash -c 'source /opt/conda/bin/activate hunyuanworld-mirror && pip install jaxtyping'")
    .run_commands("bash -c 'source /opt/conda/bin/activate hunyuanworld-mirror && pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124'")
    
)

@app.function(
    image=image,
    secrets=[secret, hf_secret],
    volumes={
        "/mnt/temp-data-volume": volume
        # "/data": modal.CloudBucketMount(
        #     bucket_name="iterative-gs-data",
        #     bucket_endpoint_url="https://927ca89d44254374e2e39138b71b1e34.r2.cloudflarestorage.com",
        #     secret=secret,
        #     read_only=False
        # )
    },
    timeout=86400,  # Adjust as needed for long-running inference
    ephemeral_disk=524288, 
    cpu=4,  # Adjust based on requirements
    # gpu="A100-80GB",
    gpu = "A100-40GB",
    memory=16384,  # 16GB, adjust as needed
)
def run_inference():
    os.chdir("/tmp")
    
    # Clone the main repository
    print("Cloning HunyuanWorld-Mirror repository...")
    subprocess.run(["git", "clone", "https://github.com/Iterative-GS/HunyuanWorld-Mirror.git"], check=True)
    
    # Change to repo directory
    os.chdir("HunyuanWorld-Mirror")
    subprocess.run(["git", "checkout", "test"], check=True)
    # Create a wrapper function to run commands in the conda environment
    def run_in_conda(cmd):
        """Run a command in the hunyuanworld-mirror conda environment"""
        full_cmd = f"bash -c 'source /opt/conda/bin/activate hunyuanworld-mirror && {cmd}'"
        subprocess.run(full_cmd, shell=True, check=True)
    
    # Install requirements
    print("Installing dependencies...")
    run_in_conda("pip install -r requirements.txt")
    run_in_conda("pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124")
    run_in_conda("pip install OpenEXR")
    # Install pycolmap
    print("Installing pycolmap...")
    subprocess.run(["git", "clone", "https://github.com/t5patil007/pycolmap.git"], check=True)
    os.chdir("pycolmap")
    run_in_conda("mv pycolmap/ pycolmap2/")
    run_in_conda("pip install -e .")
    os.chdir("..")
    run_in_conda("cp -r /mnt/temp-data-volume/DL3DV-Splat-Dataset/houtput/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3 ./houtput")
    print("Rendering...")
    run_in_conda("ls ./houtput")
    run_in_conda("python cumulative_render.py --infer_dir ./houtput --render_dir ./saved_renders --height 294 --width 518")
    run_in_conda("rm -rf /mnt/temp-data-volume/saved_renders")
    run_in_conda("mv -f ./saved_renders /mnt/temp-data-volume")

@app.local_entrypoint()
def main():
    # Run the inference function
    run_inference.remote()
