# utilities for the face mask detection project.

from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def parse_annotation(xml_path):
    """Parse one PASCAL VOC XML annotation file.

    Returns a dict: {"filename": str, "boxes": [(label, xmin, ymin, xmax, ymax), ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = int(bnd.find("xmin").text)
        ymin = int(bnd.find("ymin").text)
        xmax = int(bnd.find("xmax").text)
        ymax = int(bnd.find("ymax").text)
        boxes.append((label, xmin, ymin, xmax, ymax))

    return {"filename": filename, "boxes": boxes}


def list_annotation_files(annotation_dir):
    """Return a sorted list of all .xml annotation files in a directory."""
    return sorted(Path(annotation_dir).glob("*.xml"))
