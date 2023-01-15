import os
import urllib

import torch
from mmengine.utils import scandir
from prettytable import PrettyTable

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def get_file_list(source_root: str):
    """Get file list.

    Args:
        source_root (str): image or video source path
    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type


def show_data_classes(data_classes):
    """When printing an error, all class names of the dataset."""
    print('\n\nThe name of the class contained in the dataset:')
    data_classes_info = PrettyTable()
    data_classes_info.title = 'Information of dataset class'
    # List Print Settings
    # If the quantity is too large, 25 rows will be displayed in each column
    if len(data_classes) < 25:
        data_classes_info.add_column('Class name', data_classes)
    elif len(data_classes) % 25 != 0 and len(data_classes) > 25:
        col_num = int(len(data_classes) / 25) + 1
        data_name_list = list(data_classes)
        for i in range(0, (col_num * 25) - len(data_classes)):
            data_name_list.append('')
        for i in range(0, len(data_name_list), 25):
            data_classes_info.add_column('Class name',
                                         data_name_list[i:i + 25])

    # Align display data to the left
    data_classes_info.align['Class name'] = 'l'
    print(data_classes_info)
