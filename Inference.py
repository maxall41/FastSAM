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
    
def get_original_colors(binary_mask, original_images, bbox_coords):
    """
    Extract original colors from CryoET data for the segmented object
    
    Args:
        binary_mask: 3D binary mask of the object
        original_images: List of original image arrays
        bbox_coords: (min_z, min_y, min_x, max_z, max_y, max_x)
    
    Returns:
        Array of original colors within the bounding box where mask is True
    """
    min_z, min_y, min_x, max_z, max_y, max_x = bbox_coords
    colored_object = np.zeros(binary_mask.shape + (3,), dtype=np.uint8)
    
    for z in range(max_z - min_z):
        if min_z + z >= len(original_images):
            continue
        orig_img = np.array(original_images[min_z + z])
        slice_colors = orig_img[min_y:max_y, min_x:max_x]
        colored_object[z][binary_mask[z]] = slice_colors[binary_mask[z]]
    
    return colored_object

def get_slice_info(binary_mask, start_slice):
    """
    Get information about which slices contain the object
    
    Args:
        binary_mask: 3D binary mask of the object
        start_slice: Global slice index where this object starts
        
    Returns:
        Dictionary with slice information
    """
    slice_info = {}
    for z in range(binary_mask.shape[0]):
        if np.any(binary_mask[z]):
            global_slice = start_slice + z
            slice_mask = binary_mask[z].astype(np.uint8)
            slice_info[global_slice] = {
                'present': True,
                'mask': slice_mask,
                'pixel_count': np.sum(slice_mask)
            }
    return slice_info

def process_3d_objects(binary_stack, original_images, min_size=100):
    """
    Process 3D binary stack to identify and separate unique objects with original colors
    """
    # Label connected components in 3D
    labeled_stack, num_features = ndimage.label(binary_stack)
    
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
        
        # Get slice information
        slice_info = get_slice_info(minimal_box, bbox_coords[0])
        
        # Get original colors for the object
        colored_object = get_original_colors(minimal_box, original_images, bbox_coords)
        
        # Calculate center of mass in global coordinates
        z_indices, y_indices, x_indices = np.nonzero(object_mask)
        center = (
            np.mean(z_indices),
            np.mean(y_indices),
            np.mean(x_indices)
        )
        
        objects_3d.append({
            'minimal_shape': minimal_box,
            'colored_minimal_shape': colored_object,
            'slice_info': slice_info,
            'bbox_coords': bbox_coords,
            'center': center,
            'size': np.sum(minimal_box)
        })
    
    return objects_3d

def save_objects_pickle(objects, output_dir):
    """Save 3D objects and their information as pickle files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a dictionary with summary information
    summary_dict = {
        'centers': [obj['center'] for obj in objects],
        'sizes': [obj['size'] for obj in objects],
        'bbox_coords': [obj['bbox_coords'] for obj in objects],
        'slice_ranges': [(min(obj['slice_info'].keys()), max(obj['slice_info'].keys())) 
                        for obj in objects]
    }
    
    # Save summary information
    with open(os.path.join(output_dir, 'object_summary.pkl'), 'wb') as f:
        pickle.dump(summary_dict, f)
    
    # Save individual objects with their full information
    for i, obj in enumerate(objects):
        obj_dict = {
            'minimal_shape': obj['minimal_shape'],  # Binary mask
            'colored_minimal_shape': obj['colored_minimal_shape'],  # Original colors
            'slice_info': obj['slice_info'],  # Per-slice information
            'bbox_coords': obj['bbox_coords'],
            'center': obj['center'],
            'size': obj['size']
        }
        with open(os.path.join(output_dir, f'object_{i}.pkl'), 'wb') as f:
            pickle.dump(obj_dict, f)

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    try:
        model = FastSAM(args.model_path, weights_only=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    
    # Get sorted list of image files
    image_files = sort_slice_files([f for f in os.listdir(args.img_directory) 
                                  if f.endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
    
    # Initialize 3D stack for masks and store original images
    first_image = Image.open(os.path.join(args.img_directory, image_files[0]))
    stack_shape = (len(image_files), first_image.size[1], first_image.size[0])
    mask_stack = np.zeros(stack_shape, dtype=bool)
    original_images = []
    
    # Process each slice
    for z, image_file in enumerate(image_files):
        try:
            img_path = os.path.join(args.img_directory, image_file)
            input_image = Image.open(img_path).convert("RGB")
            original_images.append(input_image)  # Store original image
            
            everything_results = model(
                input_image,
                device=args.device,
                retina_masks=args.retina,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou    
            )
            
            prompt_process = FastSAMPrompt(input_image, everything_results, device=args.device)
            mask = get_mask_from_results(prompt_process)
            
            if mask is not None:
                mask_stack[z] = mask
            
            # Save intermediate visualization if needed
            if args.save_intermediates:
                prompt_process.plot(
                    annotations=prompt_process.everything_prompt(),
                    output_path=os.path.join(args.output, f"slice_{z:04d}.png"),
                    better_quality=args.better_quality,
                )
        except Exception as e:
            print(f"Error processing slice {z} ({image_file}): {e}")
            continue
    
    # Process 3D objects with original colors
    objects_3d = process_3d_objects(mask_stack, original_images, 
                                  min_size=100)
    
    # Save objects with pickle
    save_objects_pickle(objects_3d, os.path.join(args.output, '3d_objects'))
    
    # Print summary
    print(f"\nFound {len(objects_3d)} 3D objects:")
    for i, obj in enumerate(objects_3d):
        print(f"\nObject {i}:")
        print(f"  Size: {obj['size']} voxels")
        print(f"  Center (z,y,x): ({obj['center'][0]:.1f}, {obj['center'][1]:.1f}, {obj['center'][2]:.1f})")
        print(f"  Present in slices: {min(obj['slice_info'].keys())} to {max(obj['slice_info'].keys())}")
        print(f"  Bounding box coords: {obj['bbox_coords']}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
