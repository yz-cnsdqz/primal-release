import os
import json
import subprocess
import numpy as np


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)
    file_path = os.path.join(outdir, url.split('/')[-1])
    return file_path

def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    from matplotlib import cm as mpl_cm, colors as mpl_colors
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors

def load_smpl_part_vertex_ids(body_model='smplx'):
    main_url = 'https://meshcapade.wiki/assets/SMPL_body_segmentation/'
    if body_model == 'smpl':
        part_segm_url = os.path.join(main_url, 'smpl/smpl_vert_segmentation.json')
    elif body_model == 'smplx':
        part_segm_url = os.path.join(main_url, 'smplx/smplx_vert_segmentation.json')
    elif body_model == 'smplh':
        part_segm_url = os.path.join(main_url, 'smpl/smpl_vert_segmentation.json')

    part_segm_filepath = os.path.basename(part_segm_url)
    if not os.path.exists(part_segm_filepath):
        part_segm_filepath = download_url(part_segm_url, '.')
    part_segm = json.load(open(part_segm_filepath))

    #vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])
    return part_segm

def get_verts_colors_with_only_hand_painted(body_model='smplx'):
    part_segm = load_smpl_part_vertex_ids(body_model=body_model)
    lhand_vert_ids = part_segm['leftHand']
    rhand_vert_ids = part_segm['rightHand']

    n_vertices = {'smpl': 6890, 'smplx': 10475 }[body_model] #
    lhand_vertex_colors = np.zeros((n_vertices, 4), dtype=np.float32)
    lhand_vertex_colors[lhand_vert_ids] = 1
    rhand_vertex_colors = np.zeros((n_vertices, 4), dtype=np.float32)
    rhand_vertex_colors[rhand_vert_ids] = 1
    #lhand_vertex_colors = part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0)
    return lhand_vertex_colors.astype(np.float32), rhand_vertex_colors.astype(np.float32)


part_names = ['rightHand', 'rightUpLeg', 'leftArm', 'head', 'leftEye', 'rightEye', 'leftLeg', 'leftToeBase', 'leftFoot', 'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'rightArm', 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck', 'rightToeBase', 'spine', 'leftUpLeg', 'eyeballs', 'leftHand', 'hips']
marker_names = ['C7', 'CLAV', 'LANK', 'LFWT', 'LBAK', 'LBCEP', 'LBSH', 'LBUM', 'LBUST', 'LCHEECK', 'LELB', 'LELBIN', 'LFIN', 'LFRM2', 'LFTHI', 'LFTHIIN', 'LHEE', 'LIWR', 'LKNE', 'LKNI', 'LMT1', 'LMT5', 'LNWST', 'LOWR', 'LBWT', 'LRSTBEEF', 'LSHO', 'LTHI', 'LTHMB', 'LTIB', 'LTOE', 'MBLLY', 'RANK', 'RFWT', 'RBAK', 'RBCEP', 'RBSH', 'RBUM', 'RBUSTLO', 'RCHEECK', 'RELB', 'RELBIN', 'RFIN', 'RFRM2', 'RFRM2IN', 'RFTHI', 'RFTHIIN', 'RHEE', 'RKNE', 'RKNI', 'RMT1', 'RMT5', 'RNWST', 'ROWR', 'RBWT', 'RRSTBEEF', 'RSHO', 'RTHI', 'RTHMB', 'RTIB', 'RTOE', 'STRN', 'T8', 'LFHD', 'LBHD', 'RFHD', 'RBHD'] 
def vertex2part_level_contact(vert_ids, body_model='smplx', part_contact_thresh=30):
    part_segm = load_smpl_part_vertex_ids(body_model=body_model)

    part_vertex_count = {part: 0 for part in part_segm}
    for part_name, part_ids in part_segm.items():
        contact_inds = list(set(vert_ids) & set(part_ids))
        part_vertex_count[part_name] += len(contact_inds)

    part_verts_ids = []
    contact_parts = np.zeros(len(part_names))
    contact_markers = np.zeros(len(marker_names))
    for pid, part_name in enumerate(part_names):
        if part_vertex_count[part_name] > part_contact_thresh:
            part_verts_ids+=part_segm[part_name]
            contact_parts[pid] = 1
            for mid, marker_name in enumerate(marker_names):
                if marker67_indices[marker_name] in part_segm[part_name]:
                    contact_markers[mid] = 1
    return part_verts_ids, contact_parts, contact_markers

marker67_indices = {
    "C7": 3832,
    "CLAV": 5533,
    "LANK": 5882,
    "LFWT": 3486,
    "LBAK": 3336,
    "LBCEP": 4029,
    "LBSH": 4137,
    "LBUM": 5694,
    "LBUST": 3228,
    "LCHEECK": 2081,
    "LELB": 4302,
    "LELBIN": 4363,
    "LFIN": 4788,
    "LFRM2": 4379,
    "LFTHI": 3504,
    "LFTHIIN": 3998,
    "LHEE": 8846,
    "LIWR": 4726,
    "LKNE": 3682,
    "LKNI": 3688,
    "LMT1": 5890,
    "LMT5": 5901,
    "LNWST": 3260,
    "LOWR": 4722,
    "LBWT": 5697,
    "LRSTBEEF": 5838,
    "LSHO": 4481,
    "LTHI": 4088,
    "LTHMB": 4839,
    "LTIB": 3745,
    "LTOE": 5787,
    "MBLLY": 5942,
    "RANK": 8576,
    "RFWT": 6248,
    "RBAK": 6127,
    "RBCEP": 6776,
    "RBSH": 7192,
    "RBUM": 8388,
    "RBUSTLO": 8157,
    "RCHEECK": 8786,
    "RELB": 7040,
    "RELBIN": 7099,
    "RFIN": 7524,
    "RFRM2": 7115,
    "RFRM2IN": 7303,
    "RFTHI": 6265,
    "RFTHIIN": 6746,
    "RHEE": 8634,
    "RKNE": 6443,
    "RKNI": 6449,
    "RMT1": 8584,
    "RMT5": 8595,
    "RNWST": 6023,
    "ROWR": 7458,
    "RBWT": 8391,
    "RRSTBEEF": 8532,
    "RSHO": 6627,
    "RTHI": 6832,
    "RTHMB": 7575,
    "RTIB": 6503,
    "RTOE": 8481,
    "STRN": 5531,
    "T8": 5487,
    "LFHD": 707,
    "LBHD": 2026,
    "RFHD": 2198,
    "RBHD": 3066
}

