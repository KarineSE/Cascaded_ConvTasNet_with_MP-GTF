import os
import shutil
import argparse
from glob import glob

# Command line arguments
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--librimix_dir", type=str, default=none, help="Path to librispeech root directory',
# )

parser.add_argument(
    "--librimix_dir", type=str, default=r'C:\Users\eliashivk\Documents\DSP_final_project\asteroid-master\egs\librimix\ConvTasNet\local\LibriMix\storage\Libri2Mix', help='Path to librispeech root directory',
)


def main(args):
    librimix_dir = args.librimix_dir
    create_local_metadata(librimix_dir)


def create_local_metadata(librimix_dir):

    md_dirs = [f for f in glob(os.path.join(librimix_dir, "*/*/*")) if f.endswith("metadata")]
    for md_dir in md_dirs:
        md_files = [f for f in os.listdir(md_dir) if f.startswith("mix")]
        for md_file in md_files:
            subset = md_file.split("_")[1]
            local_path = os.path.join(
                "data", os.path.relpath(md_dir, librimix_dir), subset
            ).replace("/metadata", "")

            local_path = os.path.join(
                r'C:\Users\eliashivk\Documents\DSP_final_project\asteroid-master\egs\librimix\ConvTasNet\local',
                local_path)  # Karine and shira

            os.makedirs(local_path, exist_ok=True)
            shutil.copy(os.path.join(md_dir, md_file), local_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
