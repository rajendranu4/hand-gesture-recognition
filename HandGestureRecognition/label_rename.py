import global_constants as gc


def convert_label_to_idx(labels_list):
    return [gc.LABEL_TO_IDX.get(item) for item in labels_list]


def convert_idx_to_label(idx):
    return gc.IDX_TO_LABEL.get(idx)
