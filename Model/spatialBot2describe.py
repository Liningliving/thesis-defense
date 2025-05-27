# -- To do :Deal with walls and floors, which always contain lots of miscellaneous objects in the image
# -- To do:restrict the words used to describe words or relationship given by Spatialbot, for instance these relationship
      #defined by 3DSSG.

#To use this model, Attie need to install the transformers==4.47.0 library(Not the newest).   
import subprocess
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np
from DataProcess.aabb2obb import create_obb_mask, compute_2d_obb
from huggingface_hub import login
import textwrap
import random

# Replace YOUR_TOKEN with your actual access token
login(token="")
# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Set all the seeds:
seed = 1314520
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 2. Force deterministic algorithms:
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



model_name = 'RussRobin/SpatialBot-3B'
offset_bos = 0

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    do_sample=False,
    trust_remote_code=True,
    temperature=0)
model.eval() 
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

def normalize_bb(bb, width, height):
    """
    normalize bb according to the shape of image

    Parameters:
    - bb: Tuple or list of (xmin, ymin, xmax, ymax)
    - width: Width of the original image
    - height: Height of the original image

    Returns:
    - A list [new_xmin, new_ymin, new_xmax, new_ymax]
    """
    return bb[0]/height, bb[1]/width, bb[2]/height, bb[3]/width

def obb_mask(aligned_RGB_path, Depth_path, points):
    '''
    Args:
    points are the 2d coordinates of the object we want to describe in Depth image.
    '''
    aligned_RGB_iamge = Image.open(aligned_RGB_path)  # RGB image
    Depth_image = Image.open(Depth_path)   # Depth image

    #Calculate the obb
    box_pts, angle = compute_2d_obb(points)
    mask = create_obb_mask(box_pts,[172, 224])

    masked_depth = Depth_image * mask

    #increase the dimension of RGB mask
    mask_3d = np.expand_dims(mask, axis=2)  # Add a new axis: (172, 224, 1)
    mask_3d = np.repeat(mask_3d, repeats=3, axis=2)
    masked_RGB = aligned_RGB_iamge * mask_3d

    #change these images from np.array back to image so that it can be described by SpatialBot
    masked_depth = Image.fromarray(masked_depth)

    # Convert dtype from uint16 to uint8
    if masked_RGB.dtype == np.uint16:
        # Scale values to [0, 255] if needed
        if masked_RGB.max() > 255:
            masked_RGB = (masked_RGB / 256).astype(np.uint8)
        else:
            masked_RGB = masked_RGB.astype(np.uint8)
    masked_RGB = Image.fromarray(masked_RGB)

    # Used in debug: Save image if the object is a bathtub, id 11
    # if objectInfo[0] == '11':
    #     aligned_RGB_iamge.save("aligned_RGB_iamge.jpg")
    #     masked_RGB.save("masked_RGB.jpg");masked_depth.save("masked_depth.pgm")

    # print(Depth_path, "\n", aligned_RGB_path)
    return masked_RGB, masked_depth

def rotate_bbox_90_clockwise(bbox, image_width, image_height):
    """
    Rotates a bounding box 90 degrees clockwise.

    Parameters:
    - bbox: Tuple or list of (xmin, ymin, xmax, ymax)
    - image_width: Width of the original image
    - image_height: Height of the original image

    Returns:
    - A list [new_xmin, new_ymin, new_xmax, new_ymax] representing the rotated bounding box
    """
    xmin, ymin, xmax, ymax = bbox

    # Calculate new coordinates after rotation
    new_xmin = image_height - ymax
    new_ymin = xmin
    new_xmax = image_height - ymin
    new_ymax = xmax

    return [new_xmin, new_ymin, new_xmax, new_ymax]


def describe_object(RGB_path, Depth_path,bb, points, object_label,  preprocess_model='aabb', extra_prompt=''):
    '''
    Describe object in the RGB-D using SpatialBot
    '''
    #RGBD data process
    if preprocess_model == 'obb+mask':
        RGB_image, Depth_image = obb_mask(RGB_path, Depth_path, points)
        prompt = 'Descibe the'+ object_label+' and objects that I do not mention, but you can identify. Describe as detail as you can.'
    elif preprocess_model == 'aabb':
        #Without crop version
        RGB_image = Image.open(RGB_path)  # RGB image
        Depth_image = Image.open(Depth_path)   # Depth image
        width, height = Depth_image.size
        # Calculate new bounding box coordinates
        new_xmin, new_ymin, new_xmax, new_ymax = rotate_bbox_90_clockwise(bb, width,height)
        new_xmin, new_ymin, new_xmax, new_ymax = normalize_bb([new_xmin, new_ymin, new_xmax, new_ymax],width, height)#new_xmin/height, new_ymin/width, new_xmax/height, new_ymax/width
        prompt = f'''
        ### Region ID Mapping
        - label: "{object_label}, bbox: <{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}>"
        
        ### Question: Describe the the {object_label} and objects that I do not mention, but you can identify.'''
    elif preprocess_model == 'aabb_cropped':
        RGB_image = Image.open(RGB_path)  # RGB image
        Depth_image = Image.open(Depth_path)   # Depth image
        RGB_image = RGB_image.crop( bb )
        Depth_image = Depth_image.crop( bb )
        prompt = 'Descibe the'+ object_label+' and describe objects that I do not mention, but you can identify.'
    elif preprocess_model == 'aabb_few_shot':
        #Without crop version
        RGB_image = Image.open(RGB_path)  # RGB image
        Depth_image = Image.open(Depth_path)   # Depth image
        width, height = Depth_image.size
        # Calculate new bounding box coordinates
        new_xmin, new_ymin, new_xmax, new_ymax = rotate_bbox_90_clockwise(bb, width,height)
        new_xmin, new_ymin, new_xmax, new_ymax = normalize_bb([new_xmin, new_ymin, new_xmax, new_ymax],width, height)#new_xmin/height, new_ymin/width, new_xmax/height, new_ymax/width
        prompt = f'''
        ### Region ID Mapping
        - label: "{object_label}, bbox: <{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}>"
        
        ### Question: Describe the the {object_label} and objects that I do not mention, but you can identify.
        # For example, you should output:['book','near','wall']'''
    else:
        #This time, the images are the original RGBD
        RGB_image = Image.open(RGB_path)  # RGB image
        Depth_image = Image.open(Depth_path)   # Depth image
        prompt = 'Describe the scene.'
    RGB_image = RGB_image.transpose(Image.Transpose.ROTATE_270)
    Depth_image = Depth_image.transpose(Image.Transpose.ROTATE_270)

    # Convert Depth_image to I,16
    Depth_image = np.array(Depth_image)
    # Convert back to PIL Image in 'I;16' mode
    Depth_image = Image.fromarray(Depth_image, mode='I;16')

    channels = len(Depth_image.getbands())
    if channels == 1:
        img = np.array(Depth_image)
        height, width = img.shape
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8
        Depth_image = Image.fromarray(three_channel_array, 'RGB')

    # text prompt
    text = f"You are good at describe indoor scenes. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
    input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(device)

    # model = model.to("cuda:0")

    image_tensor = model.process_images([RGB_image,Depth_image], model.config).to(dtype=model.dtype, device=device)

    # If 'Expected all tensors to be on the same device' error is thrown, uncomment the following line
    model.get_vision_tower().to('cuda')

    # print("input_ids device:", input_ids.device)
    # print("image_tensor device:", image_tensor.device)
    # print("model.device",model.device)
    # print("vision_parent.vision_tower.device", model.model.get_vision_tower().device)

    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=100,
        use_cache=True,
        repetition_penalty=1.0 # increase this to avoid chattering
    )[0]

    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()


def describe_relationship(RGB_path, Depth_path,bounding_boxs, points, objects_info,  preprocess_model='aabb', extra_prompt=''):
    """
    Describe relationship in the RGB-D using SpatialBot

    Parameters:
    - bounding_boxs: Tuple or list of bounding_box, which format is (xmin, ymin, xmax, ymax)
    - points: the points of the 2 objects
    - objects_info: obj0_id, obj0_label, obj1_id, obj1_label = objects_info.split(',')

    Returns:
    - relationship 
    """

    if preprocess_model == 'aabb':
        #Without crop version
        RGB_image = Image.open(RGB_path)  # RGB image
        Depth_image = Image.open(Depth_path)   # Depth image
        width, height = Depth_image.size

        bb0, bb1 = bounding_boxs
        bb0 = normalize_bb( bb0,width, height )
        bb1 = normalize_bb( bb1,width, height )
        obj0_id, obj0_label, obj1_id, obj1_label = objects_info.split(',')

        prompt = textwrap.dedent( f'''
### Region ID Mapping
- obj0_id: {obj0_id}, bbox: <{bb0[0]}, {bb0[1]}, {bb0[2]}, {bb0[3]}>, label: "{obj0_label}"
- obj1_id: {obj1_id}, bbox: <{bb1[0]}, {bb1[1]}, {bb1[2]}, {bb1[3]}>, label: "{obj1_label}"
### Question: Describe the spatial relationship between {obj0_id} and {obj1_id}. Provide (1) how obj0 is positioned relative to obj1, (2) vice versa, and (3) any non-directional relation.
Your output should follow the physical laws; for example, walls cannot be above ceilings.You should use the id to refer to different objects in the answer explicitly.''' )
        
    else:
        #This time, the images are the original RGBD
        RGB_image = Image.open(RGB_path)  # RGB image
        Depth_image = Image.open(Depth_path)   # Depth image
        prompt = 'Describe the scene.'
    RGB_image = RGB_image.transpose(Image.Transpose.ROTATE_270)
    Depth_image = Depth_image.transpose(Image.Transpose.ROTATE_270)
    RGB_image.save("relationship.jpg")

    # Convert Depth_image to I,16
    Depth_image = np.array(Depth_image)
    # Convert back to PIL Image in 'I;16' mode
    Depth_image = Image.fromarray(Depth_image, mode='I;16')

    channels = len(Depth_image.getbands())
    if channels == 1:
        img = np.array(Depth_image)
        height, width = img.shape
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8
        Depth_image = Image.fromarray(three_channel_array, 'RGB')

    # text prompt
    text = f"You are SpatialBot, an expert at identifying spatial relations. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
    input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(device)

    # model = model.to("cuda:0")

    image_tensor = model.process_images([RGB_image,Depth_image], model.config).to(dtype=model.dtype, device=device)

    # If 'Expected all tensors to be on the same device' error is thrown, uncomment the following line
    model.get_vision_tower().to('cuda')

    # print("input_ids device:", input_ids.device)
    # print("image_tensor device:", image_tensor.device)
    # print("model.device",model.device)
    # print("vision_parent.vision_tower.device", model.model.get_vision_tower().device)

    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=100,
        use_cache=True,
        repetition_penalty=1.0 # increase this to avoid chattering
    )[0]

    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("DF")
    scan_id = "02b33df9-be2b-2d54-9062-1253be3ce186"
    frame_id_string = '1'
    Depth_path = os.path.join('/root/autodl-tmp/AttieCode/masked_depth.pgm')
    aligned_RGB_path = os.path.join('/root/autodl-tmp/AttieCode/masked_RGB.jpg')
    aligned_RGB_iamge = Image.open(aligned_RGB_path)  # RGB image
    Depth_image = Image.open(Depth_path)   # Depth image

    # describe_image(aligned_RGB_iamge, Depth_image)