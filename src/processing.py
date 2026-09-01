#Build a face-crop classification dataset from the raw images + XML annotations.


from pathlib import Path
import cv2
from utils import parse_annotation, list_annotation_files, CLASSES

MIN_BOX_SIZE = 10  # px, skip boxes smaller than this in either dimension


def crop_faces(image_dir, annotation_dir, output_dir):
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    for label in CLASSES:
        (output_dir / label).mkdir(parents=True, exist_ok=True)

    xml_files = list_annotation_files(annotation_dir)

    n_saved = 0
    n_skipped = 0

    for xml_path in xml_files:
        record = parse_annotation(xml_path)
        image_path = image_dir / record["filename"]

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image, skipping: {image_path}")
            continue

        for i, (label, xmin, ymin, xmax, ymax) in enumerate(record["boxes"]):
            width = xmax - xmin
            height = ymax - ymin

            if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
                n_skipped += 1
                continue

            crop = img[ymin:ymax, xmin:xmax]

            stem = Path(record["filename"]).stem
            out_path = output_dir / label / f"{stem}_{i}.png"
            cv2.imwrite(str(out_path), crop)
            n_saved += 1

    print(f"Saved {n_saved} face crops, skipped {n_skipped} tiny/degenerate boxes")


if __name__ == "__main__":
    crop_faces(
        image_dir="../data/images",
        annotation_dir="../data/annotations",
        output_dir="../data/crops",
    )
