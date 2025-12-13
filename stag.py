#!/usr/bin/env python3

#############################################
## STAG                                     #
## Stephan's Automatic Image Tagger         #
#############################################

import argparse
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

# Version information
VERSION = "1.0.3"

import rawpy
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from pillow_heif import register_heif_opener
from ram import get_transform, inference_ram as inference
from ram.models import ram_plus

from xmphandler import XMPHandler

raw_extensions = [
    ".3fr",  # Hasselblad RAW
    ".ari",  # ARRIFLEX Raw
    ".arw",  # Sony Alpha Raw
    ".bay",  # Casio RAW
    ".cr2",  # Canon RAW 2
    ".cr3",  # Canon RAW 3
    ".cap",  # Phase One RAW
    ".data",  # RED Digital Camera RAW
    ".dcr",  # Kodak RAW
    ".dng",  # Adobe Digital Negative
    ".drf",  # Kodak RAW
    ".eip",  # Phase One Enhanced Image Package
    ".erf",  # Epson RAW
    ".fff",  # Imacon/Hasselblad RAW
    ".gpr",  # GoPro RAW
    ".iiq",  # Phase One RAW
    ".k25",  # Kodak RAW
    ".kdc",  # Kodak Digital Camera RAW
    ".mdc",  # Minolta RAW
    ".mef",  # Mamiya RAW
    ".mos",  # Leaf RAW
    ".mrw",  # Minolta RAW
    ".nef",  # Nikon Electronic Format
    ".nrw",  # Nikon RAW (Coolpix)
    ".orf",  # Olympus RAW
    ".pef",  # Pentax RAW
    ".ptx",  # Pentax RAW
    ".pxn",  # Logitech RAW
    ".r3d",  # RED Digital Cinema
    ".raf",  # Fujifilm RAW
    ".raw",  # Generic RAW
    ".rwl",  # Leica RAW
    ".rw2",  # Panasonic RAW
    ".rwz",  # Rawzor compressed RAW
    ".sr2",  # Sony RAW
    ".srf",  # Sony RAW
    ".srw",  # Samsung RAW
    ".x3f",  # Sigma RAW (Foveon X3 sensor)
]


class SKTagger:
    """Main class for STAG (Stephan's Automatic Image Tagger)"""

    def __init__(
        self,
        model_path: str,
        image_size: int,
        force_tagging: bool,
        test_mode: bool,
        prefer_exact_filenames: bool,
        tag_prefix: str,
        exclusions: Optional[List[str]] = None,
    ):
        """
        Initialize the tagger with given parameters

        Args:
            model_path: Path to the pretrained model
            image_size: Image size for the model
            force_tagging: Force tagging even if images are already tagged
            test_mode: Don't actually write or modify XMP files
            prefer_exact_filenames: Use exact filenames for XMP sidecars
            tag_prefix: Prefix for tags (empty string for no prefix)
        """
        register_heif_opener()
        self.transform = get_transform(image_size=image_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"STAG using device: {self.device}")

        # Load and prepare model
        self.model = ram_plus(
            pretrained=model_path, image_size=image_size, vit="swin_l"
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        # Store configuration
        self.force_tagging = force_tagging
        self.test_mode = test_mode
        self.prefer_exact_filenames = prefer_exact_filenames
        self.tag_prefix = tag_prefix
        self.exclusions = exclusions if exclusions is not None else []

    def get_tags_for_image(self, pil_image: Image.Image) -> str:
        """
        Generate tags for a given PIL image

        Args:
            pil_image: PIL Image object

        Returns:
            String containing tags separated by '|'
        """
        try:
            torch_image = self.transform(pil_image).unsqueeze(0).to(self.device)
            res = inference(torch_image, self.model)
            return res[0]
        except Exception as e:
            print(f"Tagging failed: {e}")
            return ""

    def get_tags_for_image_at_path(self, path: str) -> str:
        """
        Open an image file and generate tags for it

        Args:
            path: Path to the image file

        Returns:
            String containing tags separated by '|'
        """
        pillow_image = Image.open(path)
        return self.get_tags_for_image(pillow_image)

    def load_image(self, image_path: str) -> Tuple[Optional[Image.Image], str]:
        """
        Load an image using the appropriate method

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (PIL Image or None, loader method used)
        """
        filename, file_extension = os.path.splitext(image_path)
        file_extension = file_extension.lower()

        # Skip XMP files - they're not images
        if file_extension == ".xmp":
            return None, "none"

        image = None
        loader = "none"

        # First try Pillow for common formats
        if file_extension not in raw_extensions:
            try:
                image = Image.open(image_path)
                loader = "pillow"
            except Exception as e:
                print(f"Pillow can't read image {image_path}: {e}")

        # If Pillow failed or it's a raw file, try rawpy
        if image is None:
            try:
                with rawpy.imread(image_path) as raw:
                    rgb = raw.postprocess()
                    image = Image.fromarray(rgb)
                    loader = "rawpy"
            except Exception as e:
                print(f"Rawpy could not read {image_path}: {e}")

        return image, loader

    def is_already_tagged(self, sidecar_files: List[str]) -> bool:
        """
        Check if an image is already tagged

        Args:
            sidecar_files: List of XMP sidecar files

        Returns:
            True if already tagged, False otherwise
        """
        if self.force_tagging:
            return False

        for current_file in sidecar_files:
            handler = XMPHandler(current_file)
            if self.tag_prefix:
                if handler.has_subject_prefix(self.tag_prefix):
                    return True
            else:
                # For empty prefix, check if any tags exist
                if len(handler.get_all_subjects()) > 0:
                    return True
        return False

    def save_tags(
        self, image_file: str, sidecar_files: List[str], tags: List[str]
    ) -> None:
        """
        Save tags to XMP sidecar files

        Args:
            image_file: Path to the image file
            sidecar_files: List of existing XMP sidecar files
            tags: List of tags to save
        """
        if not tags:
            return

        # Create sidecar file if none exists
        if len(sidecar_files) == 0:
            if not self.test_mode:
                sidecar_files = [
                    XMPHandler.create_xmp_sidecar(
                        image_file, self.prefer_exact_filenames
                    )
                ]
            else:
                print("Skipping XMP file creation, not writing tags")
                return

        # Write tags to all sidecar files
        for current_file in sidecar_files:
            handler = XMPHandler(current_file)
            for tag in tags:
                if self.tag_prefix:
                    handler.add_hierarchical_subject(f"{self.tag_prefix}|{tag}")
                else:
                    handler.add_hierarchical_subject(tag)

            if not self.test_mode:
                handler.save()

    def enter_dir(self, img_dir: str, stop_event: threading.Event) -> None:
        """
        Process all images in a directory and its subdirectories

        Args:
            img_dir: Path to the image directory
            stop_event: Threading event to stop processing
        """
        print(f"Entering {img_dir}")

        for current_dir, _, file_list in os.walk(img_dir):
            for fname in sorted(file_list):
                # Check if processing should be stopped
                if stop_event.is_set():
                    print("Tagging cancelled.")
                    return

                # Skip hidden files
                if fname.startswith("."):
                    continue

                # Process the image
                image_file = os.path.join(current_dir, fname)

                # Check exclusions
                if any(exclusion in image_file for exclusion in self.exclusions):
                    print(f"Excluding file {image_file} due to exclusion patterns.")
                    continue

                sidecar_files = XMPHandler.get_xmp_sidecars_for_image(image_file)

                # Skip if already tagged
                if self.is_already_tagged(sidecar_files):
                    print(f"File {fname} already tagged.")
                    continue

                # Load the image
                image, loader = self.load_image(image_file)

                if image is not None:
                    print(f"Looking at {image_file} loaded with {loader}:")

                    # Generate and process tags
                    tag_string = self.get_tags_for_image(image)
                    tags = [item.strip() for item in tag_string.split("|")]
                    print(f"Tags found: {tags}")

                    # Save tags to XMP
                    self.save_tags(image_file, sidecar_files, tags)


def main():
    """Main entry point for the STAG command-line tool"""
    parser = argparse.ArgumentParser(description="STAG image tagger")

    parser.add_argument("imagedir", metavar="DIR", help="path to dataset")

    parser.add_argument(
        "--prefix",
        metavar="STR",
        help='top category for tags (default="st", use empty string for no prefix)',
        default="st",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="force tagging, even if images are already tagged",
    )

    parser.add_argument(
        "--test", action="store_true", help="don't actually write or modify XMP files"
    )

    parser.add_argument(
        "--prefer-exact-filenames",
        action="store_true",
        help="write <originial_file_name>.<original_file_extension>.xmp instead of <original_file_name>.xmp",
    )

    parser.add_argument(
        "--exclusions",
        nargs="+",
        metavar="PATTERN",
        help="list of substrings; files with matching path/name will be excluded",
        default=[],
    )

    args = parser.parse_args()

    # Download the model
    pretrained = hf_hub_download(
        repo_id="xinyu1205/recognize-anything-plus-model",
        filename="ram_plus_swin_large_14m.pth",
    )

    # Create and run the tagger
    tagger = SKTagger(
        model_path=pretrained,
        image_size=384,
        force_tagging=args.force,
        test_mode=args.test,
        prefer_exact_filenames=args.prefer_exact_filenames,
        tag_prefix=args.prefix,
        exclusions=args.exclusions,
    )

    stop_event = threading.Event()
    stop_event.clear()
    tagger.enter_dir(args.imagedir, stop_event)


if __name__ == "__main__":
    main()
