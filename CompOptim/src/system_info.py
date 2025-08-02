import subprocess
import torch

'''
A simple function that collects the available GPU devices from our system.
'''
def get_cuda_devices():

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of CUDA devices
        num_devices = torch.cuda.device_count()
        
        # List each device name
        out_list = []
        for i in range(num_devices):
            out_list.append("cuda:{}".format(i))
        return(out_list)

    else:
        print("CUDA is not available on this machine.")
        return

'''
A simple function that checks if we have an NV-Link in our system, and returns true if we do.
'''
def nvlink_check():

    try:
        # Run nvidia-smi command to get NVLink topology information
        nvlink_output = subprocess.check_output(['nvidia-smi', 'topo', '-m'],stderr=subprocess.STDOUT).decode('utf-8')
        nvlink_output = nvlink_output.split("Legend")[0]

        # If 'NV' (indicating NVLink) is found in the output, NVLink is available
        return 'NV' in nvlink_output

    # Otherwise return False, and print the errors cause.
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        return False

    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure you have NVIDIA drivers installed.")
        return False
