import numpy as np 
import SimpleITK as sitk
import nibabel as nib
import os 
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.measurements import center_of_mass, label, find_objects
from skimage import measure
import argparse

def write_image(image, path):
    image_array = sitk.GetImageFromArray(image)
    sitk.WriteImage(image_array, path)


def gen_bbox(seg_label):
    label_region, label_num = label(seg_label)
    object_regions = find_objects(label_region)
    boxes = []
    for object_region in object_regions:
        box = []
        max_length = 0
        for i in range(3):
            min_coord = object_region[i].start 
            max_coord = object_region[i].stop 
            center = int(0.5 * (min_coord + max_coord))
            box.append(center)
            length = max_coord - min_coord
            if length > max_length:
                max_length = length
            if i == 2:
                box.append(max_length)
        boxes.append(box)

    return boxes 


def save_result(image, label, image_name, save_root):
    box = gen_bbox(label)
    write_image(image, save_root + "{}.nii.gz".format(image_name))
    write_image(label, save_root + "{}_label.nii.gz".format(image_name))
    np.save(save_root + "{}.npy".format(image_name), box)


def find_shift(shape, shape_s, center, top):
    to_min_z  = max(shape[0] - top, 0)
    from_min_z = max(top - shape[0], 0)
    to_max_z = shape[0]
    from_max_z = top
    to_min_y = max(int(shape[1] / 2) - center[1], 0)
    from_min_y = max(int(center[1] - shape[1] / 2), 0)
    to_max_y = min(shape_s[1] - center[1], (shape[1] / 2 - 1)) + int(shape[1]/2)
    from_max_y = min(shape_s[1] - center[1], (shape[1] / 2 - 1)) + center[1]
    
    to_min_x = max(int(shape[2] / 2) - center[2], 0)
    from_min_x = max(int(center[2] - shape[2] / 2), 0)
    to_max_x = min(shape_s[2] - center[2], (shape[2] / 2 - 1)) + int(shape[2] / 2)
    from_max_x = min(shape_s[2] - center[2], (shape[2] / 2 - 1)) + center[2]

    coord_s = [from_min_z, from_max_z, from_min_y, from_max_y, from_min_x, from_max_x]
    coord_t = [to_min_z, to_max_z, to_min_y, to_max_y, to_min_x, to_max_x]

    coord_s = np.array(coord_s).astype(np.int16)
    coord_t = np.array(coord_t).astype(np.int16)

    return coord_s, coord_t


def crop(image, label, shape=(512, 512, 512)):
    np.clip(label, 0, 1, out = label)
    mask = image > 0
    center = tuple(map(int, center_of_mass(mask)))
    max_region = find_objects(mask)[0]
    top_slice = max_region[0].stop
    source_shape = image.shape

    coord_s, coord_t = find_shift(shape, source_shape, center, top_slice)
    image_crop = np.ones(shape, dtype = np.int16) * image.min()
    label_crop = np.zeros(shape, dtype = np.uint8)
    
    image_crop[coord_t[0]:coord_t[1], coord_t[2]:coord_t[3], coord_t[4]:coord_t[5]] = \
        image[coord_s[0]:coord_s[1], coord_s[2]:coord_s[3], coord_s[4]:coord_s[5]]
    
    label_crop[coord_t[0]:coord_t[1], coord_t[2]:coord_t[3], coord_t[4]:coord_t[5]]= \
        label[coord_s[0]:coord_s[1], coord_s[2]:coord_s[3], coord_s[4]:coord_s[5]]

    return image_crop, label_crop


def rescale(image, label, input_space, output_space = (0.39, 0.39, 0.39)):
    assert image.shape == label.shape, "image shape:{} != label shape{}".format(image.shape, label.shape)
    zoom_factor = tuple([input_space[i] / output_space[i] for i in range(3)])
    # image cubic interpolation
    image_rescale = zoom(image, zoom_factor, order = 3)
    # label nearest interpolation
    label_rescale = zoom(label, zoom_factor, order = 0)
    return image_rescale, label_rescale


def read_label(paths):
    # x y z -> z, y, z
    labels = [nib.load(path).get_fdata().astype(np.int8).transpose(2, 1, 0).clip(0, 1) for path in paths]
    label = np.bitwise_or.reduce(labels, axis = 0)
    return label


def compute_space(reader, image):
    slice_number = [int(reader.GetMetaData(i, "0020|0013")) for i in range(image.GetDepth())]
    first_index, last_index = int(np.argmin(slice_number)), int(np.argmax(slice_number))
    first_position = float(reader.GetMetaData(first_index, "0020|0032").split("\\")[-1])
    last_position = float(reader.GetMetaData(last_index, "0020|0032").split("\\")[-1])
    space_z = abs(last_position - first_position) / image.GetDepth()
    # x, y, z
    space_itk = image.GetSpacing()
    space_zyx = (space_z, space_itk[1], space_itk[0])
    return space_zyx 


def read_image(path):
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(names)

    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    image = reader.Execute()
    # z, y, x
    image_array = sitk.GetArrayFromImage(image)
    space_zyx = compute_space(reader, image)
    return image_array, space_zyx


def preprocess(data_list, save_root):
    image_name, image_path, label_path = data_list
    # load
    image, space_zyx = read_image(image_path)
    label = read_label(label_path)

    # process
    image_rescale, label_rescale = rescale(image, label, space_zyx)
    image_crop, label_crop = crop(image_rescale, label_rescale)

    # save
    save_result(image_crop, label_crop, image_name, save_root)


def gen_path(data_dir):   
    image_list = []
    len_splits = []
    for root, dirs, files in os.walk(data_dir):
        len_split = len(root.split("/"))
        len_splits.append(len_split)
    len_thresh = max(len_splits)

    for root, dirs, files in os.walk(data_dir):
        len_split = len(root.split("/"))
        if len_split == len_thresh:
            image_name = root.split("/")[-3]
            label_paths = []
            label_sac = data_dir + "{}-label.nii.gz".format(image_name)
            label_fus = data_dir + "{}-label_0.nii.gz".format(image_name)

            if os.path.exists(label_sac):
                label_paths.append(label_sac)
            if os.path.exists(label_fus):
                label_paths.append(label_fus)

            image_list.append([image_name, root, label_paths])    
    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ca detection')
    parser.add_argument('--input', default='', type=str, metavar='SAVE',
                        help='directory to save dicom data (default: none)')
    parser.add_argument('--output', default='', type=str, metavar='SAVE',
                        help='directory to save nii.gz data (default: none)')
    
    global args
    args = parser.parse_args()
    save_root = args.output
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    raw_data_dir = args.input
    data_lists = gen_path(raw_data_dir)

    for data_list in data_lists:
        preprocess(data_list, save_root)


    