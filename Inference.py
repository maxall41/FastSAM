import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import os
import numpy as np
from scipy import ndimage
from skimage import measure
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_directory", type=str, default="./images/", help="path to image directory"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    parser.add_argument(
        "--padding_value", 
        type=str,
        default="255,100,255",
        help="RGB padding values for cubic padding (comma-separated)"
    )
    return parser.parse_args()

def sort_slice_files(file_list):
    """Sort slice files numerically"""
    return sorted(file_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

def get_mask_from_results(prompt_process):
    """Extract binary mask from FastSAM results"""
    ann = prompt_process.everything_prompt()
    if len(ann) == 0:
        return None
    
    # Combine all detected objects into one mask
    combined_mask = np.zeros_like(ann[0].cpu().numpy(), dtype=bool)
    for mask in ann:
        combined_mask |= mask.cpu().numpy()
    
    return combined_mask
    
def make_cubic_padded(binary_object, padding_values):
    """
    Convert arbitrary shaped binary object to cubic array with padding
    
    Args:
        binary_object: Original binary object of arbitrary shape
        padding_values: Tuple of (R,G,B) values for padding
        
    Returns:
        Cubic padded array with original object centered
    """
    # Get dimensions
    z, y, x = binary_object.shape
    max_dim = max(z, y, x)
    
    # Calculate padding for each dimension
    pad_z = (max_dim - z) // 2
    pad_y = (max_dim - y) // 2
    pad_x = (max_dim - x) // 2
    
    # Account for odd dimensions
    pad_z_end = max_dim - z - pad_z
    pad_y_end = max_dim - y - pad_y
    pad_x_end = max_dim - x - pad_x
    
    # Create padded array with RGB channels
    padded = np.full((max_dim, max_dim, max_dim, 3), padding_values, dtype=np.uint8)
    
    # Place original object in center
    z_start, z_end = pad_z, pad_z + z
    y_start, y_end = pad_y, pad_y + y
    x_start, x_end = pad_x, pad_x + x
    
    # Set voxels where binary_object is True to 0
    for i in range(3):  # For each RGB channel
        padded[z_start:z_end, y_start:y_end, x_start:x_end, i][binary_object] = 0
        
    return padded, (z_start, y_start, x_start)

def get_minimal_bounding_box(binary_mask):
    """Get the minimal bounding box that contains the object"""
    # Find non-zero indices
    z_indices, y_indices, x_indices = np.nonzero(binary_mask)
    
    if len(z_indices) == 0:
        return None, None
        
    # Get min and max for each dimension
    min_z, max_z = np.min(z_indices), np.max(z_indices) + 1
    min_y, max_y = np.min(y_indices), np.max(y_indices) + 1
    min_x, max_x = np.min(x_indices), np.max(x_indices) + 1
    
    # Extract minimal box
    minimal_box = binary_mask[min_z:max_z, min_y:max_y, min_x:max_x]
    bbox_coords = (min_z, min_y, min_x, max_z, max_y, max_x)
    
    return minimal_box, bbox_coords

def process_3d_objects(binary_stack, min_size=100, padding_values=(255,100,255)):
    """
    Process 3D binary stack to identify and separate unique objects
    Returns: List of dictionaries containing object information including minimal shapes
    """
    # Label connected components in 3D
    labeled_stack, num_features = ndimage.label(binary_stack)
    
    # Measure properties of labeled regions
    objects_3d = []
    
    for label in range(1, num_features + 1):
        # Create binary mask for this object
        object_mask = labeled_stack == label
        
        # Skip if object is too small
        if np.sum(object_mask) < min_size:
            continue
            
        # Get minimal bounding box
        minimal_box, bbox_coords = get_minimal_bounding_box(object_mask)
        if minimal_box is None:
            continue
            
        # Calculate center of mass in global coordinates
        z_indices, y_indices, x_indices = np.nonzero(object_mask)
        center = (
            np.mean(z_indices),
            np.mean(y_indices),
            np.mean(x_indices)
        )
        
        # Create cubic padded version
        padded_object, padding_offset = make_cubic_padded(minimal_box, padding_values)
        
        objects_3d.append({
            'minimal_shape': minimal_box,
            'cubic_padded': padded_object,
            'padding_offset': padding_offset,
            'bbox_coords': bbox_coords,
            'center': center,
            'size': np.sum(minimal_box)
        })
    
    return objects_3d

def save_objects_pickle(objects, output_dir):
    """Save 3D objects and their information as pickle files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a dictionary with all centers and summary information
    summary_dict = {
        'centers': [obj['center'] for obj in objects],
        'sizes': [obj['size'] for obj in objects],
        'bbox_coords': [obj['bbox_coords'] for obj in objects],
        'padding_offsets': [obj['padding_offset'] for obj in objects]
    }
    
    # Save summary information
    with open(os.path.join(output_dir, 'object_summary.pkl'), 'wb') as f:
        pickle.dump(summary_dict, f)
    
    # Save individual objects with their full information
    for i, obj in enumerate(objects):
        obj_dict = {
            'minimal_shape': obj['minimal_shape'],  # Original arbitrary shape
            'cubic_padded': obj['cubic_padded'],   # Cubic padded version
            'padding_offset': obj['padding_offset'],
            'bbox_coords': obj['bbox_coords'],
            'center': obj['center'],
            'size': obj['size']
        }
        with open(os.path.join(output_dir, f'object_{i}.pkl'), 'wb') as f:
            pickle.dump(obj_dict, f)

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Parse padding values
    padding_values = tuple(map(int, args.padding_value.split(',')))
    
    # Load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    
    # Get sorted list of image files
    image_files = sort_slice_files([f for f in os.listdir(args.img_directory) 
                                  if f.endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
    
    # Initialize 3D stack for masks
    first_image = Image.open(os.path.join(args.img_directory, image_files[0]))
    stack_shape = (len(image_files), first_image.size[1], first_image.size[0])
    mask_stack = np.zeros(stack_shape, dtype=bool)
    
    # Process each slice
    for z, image_file in enumerate(image_files):
        img_path = os.path.join(args.img_directory, image_file)
        input_image = Image.open(img_path).convert("RGB")
        
        # Get FastSAM results
        everything_results = model(
            input_image,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou    
        )
        
        # Process results and get mask
        prompt_process = FastSAMPrompt(input_image, everything_results, device=args.device)
        mask = get_mask_from_results(prompt_process)
        
        if mask is not None:
            mask_stack[z] = mask
        
        # Save intermediate visualization if needed
        prompt_process.plot(
            annotations=prompt_process.everything_prompt(),
            output_path=os.path.join(args.output, f"slice_{z:04d}.png"),
            better_quality=args.better_quality,
        )
    
    # Process 3D objects
    objects_3d = process_3d_objects(mask_stack, 
                                  min_size=args.min_object_size,
                                  padding_values=padding_values)
    
    # Save objects with pickle
    save_objects_pickle(objects_3d, os.path.join(args.output, '3d_objects'))
    
    # Print summary
    print(f"\nFound {len(objects_3d)} 3D objects:")
    for i, obj in enumerate(objects_3d):
        print(f"\nObject {i}:")
        print(f"  Size: {obj['size']} voxels")
        print(f"  Center (z,y,x): ({obj['center'][0]:.1f}, {obj['center'][1]:.1f}, {obj['center'][2]:.1f})")
        print(f"  Original shape: {obj['minimal_shape'].shape}")
        print(f"  Cubic padded shape: {obj['cubic_padded'].shape}")
        print(f"  Padding offset: {obj['padding_offset']}")
        print(f"  Bounding box coords: {obj['bbox_coords']}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
