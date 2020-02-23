import numpy as np 
import nibabel as nib
import os
import random

def oversample(config, patient_labels):
    aneurysm_labels = []
    size_lowest = config["sizelim"] / config["reso"]
    size_middle = config["sizelim2"] / config["reso"]
    size_highest = config["sizelim3"] / config["reso"]

    for i, l in enumerate(patient_labels):
        if len(l) > 0:
            for box in l:
                size = box[3]
                if size > size_lowest:
                    aneurysm_labels.append([np.concatenate([[i], box])])
                if size >= size_highest:
                    aneurysm_labels += [[np.concatenate([[i], box])]] * 2
                if size >= size_middle and size < size_highest:
                    aneurysm_labels += [[np.concatenate([[i], box])]] * 6
                if size < size_middle:
                    aneurysm_labels += [[np.concatenate([[i], box])]] * 4
    
    aneurysm_labels = np.concatenate(aneurysm_labels, axis=0)
    
    return aneurysm_labels


def load_label(root, name_list):
    patient_labels = []

    for idx in name_list:
        # z h w
        l = np.load(root + "{}.npy".format(idx))
        if np.all(l == 0):
            l = np.array([])

        patient_labels.append(l)
    return patient_labels
    

def load_image(image_path):
    # w, h, z
    image = nib.load(image_path).get_fdata()
    # z, h, w
    image = image.transpose(2, 1, 0)
    image = np.expand_dims(image, axis = 0)
    return image


def gen_start(neg_sample_flag, aneurysm_label, bound_size, crop_size, image_shape):
    start_coords = []
    for i in range(3):
        if not neg_sample_flag:
            radius = aneurysm_label[3] / 2
            label_low = np.floor(aneurysm_label[i] - radius)
            label_high = np.ceil(aneurysm_label[i] + radius)
            start_low = label_high + bound_size - crop_size[i]# + 1
            start_high = label_low - bound_size# + 1
        else:
            start_low = 0
            start_high = image_shape[i]
            aneurysm_label = np.array([np.nan, np.nan, np.nan, np.nan])

        if start_high > start_low:
            start_coords.append(np.random.randint(start_low, start_high))
        else:
            start_coords.append(int(aneurysm_label[i]) - int(crop_size[i] / 2) + np.random.randint(int(-bound_size / 2), int(bound_size / 2)))
    return start_coords, aneurysm_label


def gen_coords(start, image_shape, crop_size, stride = 4):
    start = np.array(start).astype("float32")
    crop_size = np.array(crop_size).astype("float32")
    image_shape = np.array(image_shape)

    norm_start = start / image_shape - 0.5
    norm_size = crop_size / image_shape
    
    linspaces = []
    for i in range(3):
        linspace = np.linspace(norm_start[i], norm_start[i] + norm_size[i], crop_size[i] / stride)
        linspaces.append(linspace)

    coord_z, coord_y, coord_x = np.meshgrid(linspaces[0], linspaces[1], linspaces[2], indexing = "ij")
    coord_z = coord_z[np.newaxis, ...]
    coord_y = coord_y[np.newaxis, ...]
    coord_x = coord_x[np.newaxis, ...]
    # coord (3, 32, 32, 32)
    coord = np.concatenate([coord_z, coord_y, coord_x], 0)
    coord = coord.astype("float32")

    return coord


def gen_crop(image, crop_size, start, pad_value, image_shape):
    pad = []
    pad.append([0, 0])
    for i in range (3):
        left_pad = max(0, -start[i])
        right_pad = max(0, start[i] + crop_size[i] - image_shape[i])
        pad.append([left_pad, right_pad])

    left_coords, right_coords = [], []
    for i in range (3):
        left_coord = max(start[i], 0)
        left_coords.append(left_coord)

        right_coord = min(start[i] + crop_size[i], image_shape[i])
        right_coords.append(right_coord) 

    crop = image[:, left_coords[0]:right_coords[0], left_coords[1]:right_coords[1], left_coords[2]:right_coords[2]]
    crop = np.pad(crop, pad, "constant", constant_values = pad_value)

    return crop


def convert_neg2pos(crop_shape, aneurysm_label, patient_label, bound_size):
    if np.isnan(aneurysm_label[0]):
        for box in patient_label:
            neg_box = np.array([crop_shape[1] / 2, crop_shape[2] / 2, crop_shape[3] / 2, min(crop_shape[1:4]) - bound_size])
            crop_iou = gen_iou(neg_box, box)

            if crop_iou > 0.2:
                print("this would be positive box")
                aneurysm_label = box
                print("aneurysm label is:", aneurysm_label)
    
    return aneurysm_label


def gen_iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


def crop_patch(image, aneurysm_label, patient_label, neg_sample_flag, config):

    crop_size = config["crop_size"]
    bound_size = config["bound_size"]
    stride = config["stride"]
    pad_value = config["pad_value"]

    aneurysm_label = np.copy(aneurysm_label)
    patient_label = np.copy(patient_label)
    image_shape = image.shape[1:]

    start, aneurysm_label = gen_start(neg_sample_flag, aneurysm_label, bound_size, crop_size, image_shape)

    coord = gen_coords(start, image_shape, crop_size)
    crop = gen_crop(image, crop_size, start, pad_value, image_shape)

    for i in range(3):
        aneurysm_label[i] = aneurysm_label[i] - start[i]

    for i in range(len(patient_label)):
        for j in range(3):
            patient_label[i][j] = patient_label[i][j] - start[j]

    crop_shape = crop.shape
    aneurysm_label = convert_neg2pos(crop_shape, aneurysm_label, patient_label, bound_size)

    crop_dict = {"image_patch" : crop, "aneurysm_label" : aneurysm_label, "patient_label" : patient_label, "coord" : coord}

    return crop_dict


def gen_anchor_coords(stride, output_size = [32, 32, 32]):
    offset = ((stride.astype("float")) - 1) / 2
    anchor_coords = []

    for i in range(3):
        start_coord = offset
        end_coord = offset + stride * (output_size[i] - 1) + 1
        anchor_coord = np.arange(start_coord, end_coord, stride)
        anchor_coords.append(anchor_coord)

    return anchor_coords, offset


def gen_neg_label(label, anchors, patient_label, aneurysm_label, neg_th, anchor_coords, num_neg, convert_th = 0.1):
    for box in patient_label:
        for i, anchor in enumerate(anchors):
            idx_z, idx_h, idx_w = select_samples(box, anchor, neg_th, anchor_coords)
            label[idx_z, idx_h, idx_w, i, 0] = 0

            # convert_z, convert_h, convert_w = select_samples(box, anchor, convert_th, anchor_coords)
            # if len(convert_h) != 0 and np.isnan(aneurysm_label[3]):
            #     aneurysm_label = box

    neg_z, neg_h, neg_w, neg_a = np.where(label[..., 0] == -1)
    select_neg_idxs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
    select_neg_z, select_neg_h, select_neg_w, select_neg_a = neg_z[select_neg_idxs], neg_h[select_neg_idxs], neg_w[select_neg_idxs], neg_a[select_neg_idxs]
    label[..., 0] = 0  
    label[select_neg_z, select_neg_h, select_neg_w, select_neg_a, 0] = -1

    return label, aneurysm_label


def gen_pos_label(label, anchors, aneurysm_label, pos_th, anchor_coords, offset, stride, output_size = [32, 32, 32]):
    idxs_z, idxs_h, idxs_w, idxs_a = [], [], [], []
    for i, anchor in enumerate(anchors):
        idx_z, idx_h, idx_w = select_samples(aneurysm_label, anchor, pos_th, anchor_coords)
        idxs_z.append(idx_z)
        idxs_h.append(idx_h)
        idxs_w.append(idx_w)
        idxs_a.append(i * np.ones((len(idx_h),), np.int64))

    idxs_z = np.concatenate(idxs_z, 0)
    idxs_h = np.concatenate(idxs_h, 0)
    idxs_w = np.concatenate(idxs_w, 0)
    idxs_a = np.concatenate(idxs_a, 0)

    if len(idxs_h) == 0:
        pos_anchor = []
        for i in range(3):
            pos_anchor_coord = min(output_size[i] - 1, max(0, int(np.round((aneurysm_label[i] - offset) / stride))))
            pos_anchor.append(pos_anchor_coord)
        pos_anchor_idx = np.argmin(np.abs(np.log(aneurysm_label[3] / (anchors + 1e-6))))
        pos_anchor.append(int(pos_anchor_idx))
    else:
        pos_anchor_idx = random.sample(range(len(idxs_h)), 1)[0]
        pos_anchor = [idxs_z[pos_anchor_idx], idxs_h[pos_anchor_idx], idxs_w[pos_anchor_idx], idxs_a[pos_anchor_idx]]
    
    label_z, label_h, label_w = aneurysm_label[0], aneurysm_label[1], aneurysm_label[2]
    anchor_z, anchor_h, anchor_w = anchor_coords[0], anchor_coords[1], anchor_coords[2]
    anchor_pos_z, anchor_pos_h, anchor_pos_w = anchor_z[pos_anchor[0]], anchor_h[pos_anchor[1]], anchor_w[pos_anchor[2]]
    anchor_size = anchors[pos_anchor[3]]
    label_size = aneurysm_label[3]
    dz = (label_z - anchor_pos_z) / anchor_size
    dh = (label_h - anchor_pos_h) / anchor_size 
    dw = (label_w - anchor_pos_w) / anchor_size

    dd = np.log(label_size / (anchor_size + 1e-6))

    label[pos_anchor[0], pos_anchor[1], pos_anchor[2], pos_anchor[3], :] = [1, dz, dh, dw, dd]
    return label 


def gen_range(anchor_coord, margin, label):
    start = label - margin
    end = label + margin 
    range_compute_iou = np.logical_and(anchor_coord >= start, anchor_coord <= end)
    idx_compute_iou = np.where(range_compute_iou)[0]
    return idx_compute_iou


def gen_inter(centers, anchor, label):
    r_anchor = anchor / 2
    s_anchor = centers - r_anchor
    e_anchor = centers + r_anchor

    r_label = label[3] / 2
    s_label = label[:3] - r_label 
    s_label = s_label.reshape((1, -1))
    e_label = label[:3] + r_label 
    e_label = e_label.reshape((1, -1)) 

    overlap = np.maximum(0, np.minimum(e_anchor, e_label) - np.maximum(s_anchor, s_label)) 
    inter = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]

    return inter


def gen_centers(range_z, range_h, range_w, anchor_coords):
    len_z, len_h, len_w = len(range_z), len(range_h), len(range_w)
    # h (A, ) -> (A, 1, 1) w (B, ) -> (1, B, 1) z (C, ) -> (1, 1, C)
    range_z = range_z.reshape((-1, 1, 1))
    range_h = range_h.reshape((1, -1, 1))
    range_w = range_w.reshape((1, 1, -1))
    
    # h (A, 1, 1) -> (A, B, C ) -> (A * B * C, )
    range_z = np.tile(range_z, (1, len_h, len_w)).reshape((-1))
    range_h = np.tile(range_h, (len_z, 1, len_w)).reshape((-1))
    range_w = np.tile(range_w, (len_z, len_h, 1)).reshape((-1))

    anchor_z, anchor_h, anchor_w = anchor_coords[0], anchor_coords[1], anchor_coords[2]

    # range_h (A * B * C, )
    # anchor[range_h] (A * B * C, ) -> (A * B * C, 1)
    # centers: from (A * B * C, 1) concate axis = 1 -> (A * B * C, 3)
    centers = np.concatenate([anchor_z[range_z].reshape((-1, 1)), anchor_h[range_h].reshape((-1, 1)), anchor_w[range_w].reshape((-1, 1))], axis = 1)

    return centers, range_z, range_h, range_w


def select_samples(label, anchor, th, anchor_coords):
    z, h, w, d = label 
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / (max_overlap + 1e-6) / (max_overlap + 1e-6)
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0, ), np.int64), np.zeros((0, ), np.int64)
    else:
        margin = 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        
        range_z = gen_range(anchor_coords[0], margin, z)
        range_h = gen_range(anchor_coords[1], margin, h)
        range_w = gen_range(anchor_coords[2], margin, w)

        if len(range_z) == 0 and len(range_h) == 0 and len(range_w) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        centers, range_z, range_h, range_w = gen_centers(range_z, range_h, range_w, anchor_coords)
    
        inter = gen_inter(centers, anchor, label)
        union = anchor * anchor * anchor + d * d * d - inter
        iou = inter / union 

        mask = iou >= th 

        select_z = range_z[mask]
        select_h = range_h[mask]
        select_w = range_w[mask]

        return select_z, select_h, select_w
   

def map_label(config, aneurysm_label, patient_label):
    anchors = np.asarray(config["anchors"])
    stride = np.array(config["stride"])
    neg_th = config["th_neg"]
    pos_th = config["th_pos_train"]
    num_neg = int(config["num_neg"])   
    output_size = [32, 32, 32]
    outfea_size = output_size + [len(anchors), 5]
    label_init = -1 * np.ones(outfea_size, np.float32)

    anchor_coords, offset = gen_anchor_coords(stride)
    #label, anchors, patient_label, aneurysm_label, neg_th, anchor_coords
    label, aneurysm_label = gen_neg_label(label_init, anchors, patient_label, aneurysm_label, neg_th, anchor_coords, num_neg)

    if np.isnan(aneurysm_label[0]):
        return label
    #label, anchors, anchors_coords, patient_label, aneurysm_label, offset, 
    label = gen_pos_label(label, anchors, aneurysm_label, pos_th, anchor_coords, offset, stride)

    return label

    
def augment(sample, aneurysm_label, patient_label, coord):
    flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
    sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
    coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
    for ax in range(3):
        if flipid[ax] == -1:
            aneurysm_label[ax] = np.array(sample.shape[ax + 1]) - aneurysm_label[ax]
            patient_label[:, ax] = np.array(sample.shape[ax + 1]) - patient_label[:, ax]

    return sample, aneurysm_label, patient_label, coord


def pad_image(image, stride, pad_value):
    raw_z, raw_h, raw_w = image.shape[1:]
    pad_z = int(np.ceil(float(raw_z) / stride)) * stride
    pad_h = int(np.ceil(float(raw_h) / stride)) * stride
    pad_w = int(np.ceil(float(raw_w) / stride)) * stride

    pad = [[0, 0], [0, pad_z - raw_z], [0, pad_h - rah_w], [0, pad_w - raw_w]]

    image_pad = np.pad(image, pad, "constant", constant_values = pad_value)

    return image_pad

    
def gen_test_coords(image_shape, stride):
    linspaces = []
    for i in range(3):
        linspace = np.linspace(-0.5, 0.5, image_shape[i] / stride)
        linspaces.append(linspace)

    coord_z, coord_y, coord_x = np.meshgrid(linspaces[0], linspaces[1], linspaces[2], indexing = "ij")
    coord_z = coord_z[np.newaxis, ...]
    coord_y = coord_y[np.newaxis, ...]
    coord_x = coord_x[np.newaxis, ...]
    # coord (3, 32, 32, 32)
    coord = np.concatenate([coord_z, coord_y, coord_x], 0)
    coord = coord.astype("float32")

    return coord
