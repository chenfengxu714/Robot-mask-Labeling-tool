import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as mask_util
import cv2


def post_process_mask(mask, mode):

    mask = mask.squeeze()
    if mode == 'auto':
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.erode(mask, kernel, iterations=3)
    return mask


def plot_instructions(mode):
    if mode == 'auto':
        ax.set_title(f'Auto Mode: Left click to set instance points, right click to set background points, c to clear all points, space to save and exit')
    elif mode == 'manual':
        ax.set_title(f'Manual Mode: Click to record polygon vertex and press z to generate mask, c to clear all points, space to save and exit')

def auto_generate_mask(predictor, img, instance_points, background_points, mode):
    instance_points = np.array(instance_points)
    background_points = np.array(background_points)
    if background_points.shape[0] == 0:
        points = instance_points.astype(np.int32)
        label = np.ones((instance_points.shape[0])).astype(np.int32)
    else:
        points = np.concatenate((instance_points, background_points), axis=0).astype(np.int32)
        label = np.concatenate((np.ones((instance_points.shape[0])), np.zeros((background_points.shape[0]))),
                               axis=0).astype(np.int32)
    predictor.set_image(img)
    mask, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=label,
        multimask_output=False,
    )
    mask_bin = mask.astype(np.uint8).transpose(1, 2, 0)
    # mask_bin = post_process_mask(mask_bin, mode)
    return mask_bin


def manual_generate_mask(points, dimensions, mode):
    mask = np.zeros((dimensions[0], dimensions[1]), dtype=np.uint8)
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    mask = mask // 255
    mask_bin = mask[np.newaxis, :, :]
    # mask_bin = post_process_mask(mask_bin, mode)
    return mask_bin

def decode_mask(mask_coco, img):
    if len(mask_coco) == 1:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = mask_util.decode(mask_coco)
    return mask


def pre_process_img(img):
    # check row sum and column sum to remove black border
    # sum over all channels to find 0 for black pixels
    img_1d = np.sum(img, axis=-1)
    row_sum = np.sum(img_1d, axis=1)
    col_sum = np.sum(img_1d, axis=0)
    # find the 4 corners of the image
    row_idx = np.where(row_sum > 0)[0]
    col_idx = np.where(col_sum > 0)[0]
    if len(row_idx) == img.shape[0] and len(col_idx) == img.shape[1]:
        return img, None
    else:
        img = img[row_idx[0]:row_idx[-1]+1, col_idx[0]:col_idx[-1]+1]
        crop_info = [int(row_idx[0]), int(col_idx[0]), int(row_idx[-1]+1), int(col_idx[-1]+1)]
        return img, crop_info


def filter_points(points):
    points = [x for x in points if x[0] and x[1] is not None]
    return points

def onclick(event):
    global instance_coords, bkgd_coords, mask, predictor, mode
    ix, iy = event.xdata, event.ydata

    if event.button == 1:  # Left click
        instance_coords.append([ix, iy])
        instance_coords = filter_points(instance_coords)
    elif event.button == 3:  # Right click
        bkgd_coords.append([ix, iy])
        bkgd_coords = filter_points(bkgd_coords)

    if mode == 'auto':
        if len(instance_coords) > 0:
            mask = auto_generate_mask(predictor, img, instance_coords, bkgd_coords, mode)

    ax.clear()
    # ax.plot(ix, iy, color)
    for x, y in instance_coords:
        ax.plot(x, y, 'bo')
    if len(bkgd_coords) > 0:
        for x, y in bkgd_coords:
            ax.plot(x, y, 'ro')
    if mode == 'auto':
        ax.imshow(img, alpha=0.5)  # Show the original image slightly transparent
        ax.imshow(mask, cmap='jet', alpha=0.5)  # Overlay the mask with transparency
        plot_instructions(mode)
    if mode == 'manual':
        ax.imshow(img)
        plot_instructions(mode)
    fig.canvas.draw()


def onkey(event):
    global instance_coords, bkgd_coords, mask, mode
    if event.key == 'c':
        instance_coords.clear()
        bkgd_coords.clear()
        ax.cla()
        plot_instructions(mode)
        ax.imshow(img)
        fig.canvas.draw()
    elif event.key == ' ':  # Check for space key
        plt.close(fig)  # Close the figure window
    elif event.key == 'z':
        if len(instance_coords) >= 3:
            mask = manual_generate_mask(instance_coords, img.shape[:2], mode)
            ax.imshow(img, alpha=0.5)  # Show the original image slightly transparent
            ax.imshow(mask, cmap='jet', alpha=0.5)  # Overlay the mask with transparency
            fig.canvas.draw()
    elif event.key == 'x':
        if mode == 'auto':
            mode = 'manual'
            instance_coords.clear()
            bkgd_coords.clear()
            ax.cla()
            plot_instructions(mode)
            ax.imshow(img)
            fig.canvas.draw()
        elif mode == 'manual':
            mode = 'auto'
            instance_coords.clear()
            bkgd_coords.clear()
            ax.cla()
            plot_instructions(mode)
            ax.imshow(img)
            fig.canvas.draw()



def interactive_mask(image, sam_predictor, init_mode='auto'):
    global instance_coords, bkgd_coords, ax, fig, img, mask, predictor, mode
    coords, mask = [], np.zeros(image.shape[:2], dtype=bool)
    img = image
    mode = init_mode
    predictor = sam_predictor
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img)
    plot_instructions(mode)
    instance_coords = []
    bkgd_coords = []
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    return mask, mode


def plot_mask(img, masks):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_anns(masks, plot_bbox=False)
    plt.axis('off')
    plt.show()


def save_mask(mask_bin, img_path, mask_dir):
    file_name = img_path
    mask_file = os.path.join(mask_dir, file_name)
    mask = mask_bin * 255
    cv2.imwrite(mask_file, mask)


if __name__ == '__main__':
    root_dir = os.getcwd()
    init_mode = 'auto'
    obj_name = 'diamond'
    mode = ''
    ref_views = 3
    data_dir = f'../demo_data/robot_insertion/{obj_name}/ref_views_{ref_views}{mode}/rgb'
    mask_dir = f'../demo_data/robot_insertion/{obj_name}/ref_views_{ref_views}{mode}/mask'
    # data_dir = f'../demo_data/robot_insertion/{obj_name}/query_views/rgb'
    # mask_dir = f'../demo_data/robot_insertion/{obj_name}/query_views/mask'

    # sam_checkpoint = os.path.join(root_dir, 'checkpoint')
    sam_checkpoint = os.path.join(os.path.join(root_dir, 'checkpoint'), 'sam_vit_h_4b8939.pth')
    model_type = "vit_h"
    device = torch.device('cuda:1')
    mode = 'binary_mask'
    # mode = 'coco_rle'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for i, img_path in enumerate(sorted(os.listdir(data_dir))):
        mask_name = img_path
        # if mask_name in os.listdir(mask_dir):
        #     continue
        rgb_file = os.path.join(data_dir, img_path)
        img = cv2.imread(rgb_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img, crop_info = pre_process_img(img)

        mask, init_mode = interactive_mask(img, predictor, init_mode=init_mode)
        save_mask(mask, img_path, mask_dir)

        # print('mask progress: {}/{}'.format(cur_length, total_length))



