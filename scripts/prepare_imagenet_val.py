from pathlib import Path
import argparse
import shutil
import tarfile
import tempfile

from scipy.io import loadmat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devkit",
        type=str,
        required=True,
        help="ILSVRC2012_devkit_t12.tar.gz 的路径",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="已经解压出来的 ImageNet val 图片目录",
    )
    return parser.parse_args()

def load_id_to_wnid(devkit_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with tarfile.open(devkit_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        meta_path = tmpdir / "ILSVRC2012_devkit_t12" / "data" / "meta.mat"
        meta = loadmat(meta_path, squeeze_me=True, struct_as_record=False)

        id_to_wnid = {}

        for synset in meta["synsets"]:
            class_id = int(synset.ILSVRC2012_ID)
            wnid = str(synset.WNID)

            if 1 <= class_id <= 1000:
                id_to_wnid[class_id] = wnid

    return id_to_wnid

def load_val_labels(devkit_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with tarfile.open(devkit_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        label_path = (
            tmpdir
            / "ILSVRC2012_devkit_t12"
            / "data"
            / "ILSVRC2012_validation_ground_truth.txt"
        )

        labels = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                labels.append(int(line.strip()))

    return labels

def main():
    args = parse_args()

    devkit_path = Path(args.devkit)
    val_dir = Path(args.val_dir)

    id_to_wnid = load_id_to_wnid(devkit_path)
    labels = load_val_labels(devkit_path)

    print("num classes:", len(id_to_wnid))
    print("num val labels:", len(labels))

    if len(id_to_wnid) != 1000:
        raise ValueError("class mapping should contain 1000 ImageNet classes")

    if len(labels) != 50000:
        raise ValueError("ImageNet validation set should contain 50000 labels")

    moved = 0

    for image_index, class_id in enumerate(labels, start=1):
        image_name = f"ILSVRC2012_val_{image_index:08d}.JPEG"
        src_path = val_dir / image_name

        wnid = id_to_wnid[class_id]
        dst_dir = val_dir / wnid
        dst_path = dst_dir / image_name

        dst_dir.mkdir(parents=True, exist_ok=True)

        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            moved += 1
        elif dst_path.exists():
            continue
        else:
            raise FileNotFoundError(f"missing validation image: {src_path}")


    print("moved images:", moved)
    print("done:", val_dir)

if __name__ == "__main__":
    main()