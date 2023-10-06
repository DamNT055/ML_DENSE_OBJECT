import os 
from tqdm import tqdm
from six import raise_from
from vision_utils.get_image_size import get_image_size

def _parse(value, function, fmt):
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)
def _open_for_csv(path):
    return open(path, 'r', newline='')
def get_image_metadata(file_path):
    size = os.path.getsize(file_path)
def _read_classes(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1
        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(
            class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError(
                'line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
        return result
def _read_images(base_dir):
    result = {}
    #dirs = [os.path.join(base_dir, o) for o in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, o))]
    dirs = [os.path.join(base_dir, 'images')]
    if len(dirs) == 0:
        dirs = ['']
    for project in dirs:
        project_imgs = os.listdir(os.path.join(base_dir, project))
        i = 0
        print("Loading images...")
        for image in tqdm(project_imgs):
            try:
                img_file = os.path.join(base_dir, project, image)
                exists = os.path.isfile(img_file)

                if not exists:
                    ####################### continue
                    print("Warning: Image file {} is not existing".format(img_file))
                    continue
                # Image shape
                height, width = get_image_size(img_file)
                result[img_file] = {"width": width, "height": height}
                i += 1
            except Exception as e:
                print("Error: {} in image: {}".format(str(e), img_file))
                continue
    return result
def _read_annotations(csv_reader, classes, base_dir, image_existence):
    result = {}
    base_dir = os.path.join(base_dir, 'images')
    for line, row in enumerate(csv_reader):
        line += 1
        try:
            img_file, x1, y1, x2, y2, class_name, width, height = row[:]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            width = int(width)
            height = int(height)
            if x1 >= width:
                x1 = width - 1
            if x2 >= width:
                x2 = width - 1

            if y1 >= height:
                y1 = height - 1
            if y2 >= height:
                y2 = height - 1
            # x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0
            if x1<0 | y1<0 or x2<=0 or y2<=0:
                print("Warning: Image file {} has some bad boxes annotations".format(img_file))
                continue
            # Append root path
            img_file = os.path.join(base_dir, img_file)
            # Check images exists
            if img_file not in image_existence:
                ####################### continue
                print("Warning: Image file {} is not existing".format(img_file))
                continue
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                None)
        if img_file not in result:
            result[img_file] = []
        # If a row contains only an image path, it's an image without annotations.
        if (x1,x2,y1,y2,class_name) == ('', '', '', '', ''):
            continue
        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result

