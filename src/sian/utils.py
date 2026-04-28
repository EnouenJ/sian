
import datetime
import json
import os
import torch
import psutil




def gettimestamp():
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y%m%d_%H%M%S")
    return nowstr




def convert_inter_to_saveable(index):
    json_index = []
    for i in index:
        json_index.append(str(i))
    return json_index

def convert_inter_to_loaded(json_index):
    index = []
    for str_i in json_index:
        index.append(int(str_i))
    return tuple(index)

def save_interactions_json(indices, file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    json_indices = [convert_inter_to_saveable(index) for index in indices]
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(json_indices, f, ensure_ascii=False, indent=4)
    # https://www.onlinegdb.com/Gqa_f9hys
    # json_str = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, json_str)

    indent = 4
    json_str = ""
    json_str += "[\n"
    for jj,index in enumerate(json_indices):
        line = ""
        line += " "*indent+"["
        for ii,str_i in enumerate(index):
            line+="\""+str_i+"\""
            if ii!=len(index)-1:
                line+=", "
        line += "]"
        if jj!=len(json_indices)-1:
            line+=","
        json_str += line + "\n"
    json_str += "]"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

def load_interactions_json(file_path):
    with open(file_path) as f:
        json_indices = json.load(f)
    indices = [convert_inter_to_loaded(json_index) for json_index in json_indices]
    return indices




def convert_dict_keys_to_int(old_dict):
    new_dict = {}
    for key_str in old_dict:
        try: #NOTE: care on this because of floats and maybe .isdigit() adapted for negatives is better
            new_dict[int(key_str)] = old_dict[key_str]
        except ValueError:
            new_dict[key_str] = old_dict[key_str]
    return new_dict

def convert_dict_keys_to_str(old_dict):
    new_dict = {}
    for key_nonstr in old_dict:
        try:
            new_dict[str(key_nonstr)] = old_dict[key_nonstr]
        except ValueError:
            new_dict[key_nonstr] = old_dict[key_nonstr]
    return new_dict






def get_device_details(device, VERBOSE=True):
    torch_version = torch.__version__
    if VERBOSE:
        print(f"Using device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        cuda_version = torch.version.cuda
        if VERBOSE:
            print("GPU Name:", gpu_name)
            print("CUDA Version:", cuda_version)
    else:
        gpu_name = "CPU"
        cuda_version = None
        if VERBOSE:
            print("No GPU available.")
            
    return_dictionary = {
        "gpu_name" : gpu_name,
        "cuda_version" : cuda_version,
        "torch_version" : torch_version,
    }
    return return_dictionary

def get_cpu_details(VERBOSE=True):
    cpu_count = os.cpu_count()
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)

    if VERBOSE:
        print("CPU Information:")
        print(f"  Number of CPUs in the system: {cpu_count}")
        print(f"  Physical Cores: {cpu_count_physical}")
        print(f"  Logical Cores: {cpu_count_logical}")

    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024**3)
    if VERBOSE:
        print("\nMemory Information:")
        print(f"  Total Memory: {memory_total:.2f} GB")

    return_dictionary = {
        "memory_total" : memory_total,
        "cpu_count" : cpu_count,
        "cpu_count_physical" : cpu_count_physical,
        "cpu_count_logical" : cpu_count_logical,
    }
    return return_dictionary