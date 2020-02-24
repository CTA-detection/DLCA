import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
MAX_16_BIT = pow(2, 15) - 1

class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value

    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin

        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))

        nzhw = [nz,nh,nw]
        self.nzhw = nzhw

        pad = [ [0, 0],
                [margin, nz * side_len - z + margin],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits,nzhw

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):

        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        # if nzhw==None:
        #     nz = self.nz
        #     nh = self.nh
        #     nw = self.nw
        # else:
        nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride
        side_len = int(side_len)
        margin = int(margin)

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones((
            int(nz * side_len),
            int(nh * side_len),
            int(nw * side_len),
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output


def iou(box0, box1):
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


def nms(output, nms_th, num_bbox):
    y_max, x_max, z_max, y_min, x_min, z_min = 350, 380, 355, 123, 135, 16
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        if bbox[1] > y_max or bbox[1] < y_min or bbox[2] > x_max or bbox[2] < x_min or bbox[3] > z_max or bbox[3] < z_min:
            continue
        else:
            flag = 1
            for j in range(len(bboxes)):
                if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                    flag = -1
                    break
            if flag == 1:
                bboxes.append(bbox)
    if len(bboxes) > num_bbox:
        bboxes = bboxes[0:num_bbox]
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes
    

def postprocess(pred_raw, conf_th = -1, nms_th = 0.02, topk = 25, num_bbox = 50):
    pred_raw = pred_raw[pred_raw[:, 0] >= conf_th]
    pred_nms = nms(pred_raw, nms_th, num_bbox)
    sorted(pred_nms,key=lambda x:x[0])
    pred_nms = pred_nms[:topk]

    return pred_nms


def read_nib(path):
    img=nib.load(path)
    img_arr=img.get_data().transpose(2,1,0)
    return img_arr    


def write(image3D_pix,filename):
    image3D = sitk.GetImageFromArray(image3D_pix)
    sitk.WriteImage(image3D,filename)
    

def plot_box(image_name, boxes):
    image_path = "./test_image/{}".format(image_name)
    image = read_nib(image_path)
    D,H,W = image.shape
    print(image.shape)
    
    for box in boxes:
        _,y,x,z,d = box
        y_min, y_max = np.clip(int(y - 0.5 * d), 0, H - 1), np.clip(int(y + 0.5 * d), 0, H - 1)
        x_min, x_max = np.clip(int(x - 0.5 * d), 0, W - 1), np.clip(int(x + 0.5 * d), 0, W - 1)
        z_min, z_max = np.clip(int(z - 0.5 * d), 0, D - 1), np.clip(int(z + 0.5 * d), 0, D - 1)
        y_min, y_max = int(y_min), int(y_max)
        x_min, x_max = int(x_min), int(x_max)
        z_min, z_max = int(z_min), int(z_max)
        image[z_min, y_min : y_max, x_min : x_max] = MAX_16_BIT
        image[z_max, y_min : y_max, x_min : x_max] = MAX_16_BIT
        image[z_min : z_max, y_min, x_min : x_max] = MAX_16_BIT
        image[z_min : z_max, y_max, x_min : x_max] = MAX_16_BIT
        image[z_min : z_max, y_min : y_max, x_min] = MAX_16_BIT
        image[z_min : z_max, y_min : y_max, x_max] = MAX_16_BIT  
        
    write(image, "./prediction/{}.nii.gz".format(image_name))
    