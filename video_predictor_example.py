#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# # Video segmentation with SAM 2


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib


# Set the GPU number (e.g., GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[27]:


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ### Loading the SAM 2 video predictor
# #### Here you should specify your model checkpoint path

# In[28]:


from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# So the list is the clip video we want to process, and the video_dir is the directory of the video!
lists = range(0, 5)
root_path = "./rebuttal_data_source/ur5_img_and_language/"
mask_path = "./rebuttal_data_source/ur5_img_and_language_masks/"
# sorted based on the number of the clip id
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


# In[29]:


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Function to save the mask
def save_mask(mask, save_path):
    # Remove the channel dimension if it's there
    mask = mask.squeeze(0)  # shape becomes (256, 256)
    
    # Convert boolean mask to uint8 (0 or 255)
    mask_img = (mask * 255).astype(np.uint8)
    
    # Create an Image object and save it
    mask_image = Image.fromarray(mask_img)
    mask_image.save(save_path)

# Function to merge masks and save the combined mask
def merge_and_save_masks(masks, save_path):
    # Assuming all masks have the same shape, get the shape of one mask
    combined_mask = np.zeros_like(list(masks.values())[0], dtype=np.uint8)
    
    # Merge all masks together
    for mask in masks.values():
        combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))
    
    # Convert to an image and save
    combined_mask_img = Image.fromarray(combined_mask[0] * 255)  # Scale boolean to 0-255
    combined_mask_img.save(save_path)


# #### Select an example video

# We assume that the video is stored as a list of JPEG frames with filenames like `<frame_index>.jpg`.
# 
# For your custom videos, you can extract their JPEG frames using ffmpeg (https://ffmpeg.org/) as follows:
# ```
# ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
# ```
# where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks ffmpeg to start the JPEG file from `00000.jpg`.

def extract_clip_id(file_name):
    return int(file_name.split('_')[-1])

video_dirs = [os.path.join(root_path, sorted(os.listdir(root_path), key=extract_clip_id)[i]) for i in lists]

image_lists = []
for video_dir in video_dirs:
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    for frame_name in frame_names:
        image_lists.append(os.path.join(video_dir, frame_name))

video_dir = image_lists



inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)

# Initialize variables
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
points = []
labels = []

def onclick(event):
        if event.inaxes is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:  # Left click for positive
            label = 1
        elif event.button == 3:  # Right click for negative
            label = 0
        else:
            return
        points.append([x, y])
        labels.append(label)
        ax.plot(x, y, 'ro' if label == 1 else 'bo')
        plt.draw()
        if len(points) >= 1:
            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_np,
                labels=labels_np,
            )
            ax.clear()
            img = Image.open(video_dir[ann_frame_idx])
            ax.imshow(img)
            show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])
            show_points(points_np, labels_np, ax)
            plt.draw()
# # Load and display the image
fig, ax = plt.subplots(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
ax.imshow(Image.open(video_dir[ann_frame_idx]))

# Connect the click event to the function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Show the plot and wait for user interaction
plt.show()

# Convert the points and labels to numpy arrays
points = np.array(points, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)


_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(video_dir[ann_frame_idx]))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])


# With this 2nd refinement click, now we get a segmentation mask of the entire child on frame 0.

# #### Step 3: Propagate the prompts to get the masklet across the video

# To get the masklet throughout the entire video, we propagate the prompts using the `propagate_in_video` API.

# In[36]:


# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Iterate over all frames
for out_frame_idx in range(0, len(video_dir), 1):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(video_dir[out_frame_idx]))
    
    # Show and merge masks for each object in the frame
    masks = {}
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        masks[out_obj_id] = out_mask
    
    save_path = os.path.join(mask_path, video_dir[out_frame_idx].split('/')[-2])
    # Save the merged mask
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_image_path = os.path.join(save_path, video_dir[out_frame_idx].split('/')[-1])
    print(save_image_path)
    merge_and_save_masks(masks, save_image_path)


# render the segmentation results every few frames
vis_frame_stride = 15
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(video_dir[out_frame_idx]))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
