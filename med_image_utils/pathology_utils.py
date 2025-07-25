import os
import cv2
import staintools
from tqdm import tqdm
from natsort import natsorted


def wsi_stain_normalization(reference_image_path: str,
                            source_images_dir: str,
                            normalized_images_dir: str,
                            method: str = 'macenko'):
    """
    Batch stain-normalization for whole-slide images (WSI).

    Parameters
    ----------
    reference_image_path : str
        Path to the reference patch/slide used to fit the normalizer.
    source_images_dir : str
        Directory containing images to be normalized.
    normalized_images_dir : str
        Output directory for normalized images (will be created if absent).
    method : str, optional
        Stain-normalization algorithm: 'macenko', 'vahadane', or 'reinhard'
        (default 'macenko').

    Returns
    -------
    failed_files : list[tuple[str, str]]
        List of (image_path, error_message) for images that failed to process;
        empty list if all succeed.
    """

    # ---------- helpers ----------
    def _load_rgb(path: str):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _save_rgb(img_rgb, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    def _build_normalizer(ref_rgb, alg: str):
        alg = alg.lower()
        if alg == 'macenko':
            norm = staintools.StainNormalizer(method='macenko')
        elif alg == 'vahadane':
            norm = staintools.StainNormalizer(method='vahadane')
        elif alg == 'reinhard':
            norm = staintools.ReinhardColorNormalizer()
        else:
            raise ValueError(f"Unsupported method: {alg}")
        norm.fit(ref_rgb)
        return norm

    # ---------- pipeline ----------
    ref_rgb = _load_rgb(reference_image_path)
    normalizer = _build_normalizer(ref_rgb, method)

    os.makedirs(normalized_images_dir, exist_ok=True)
    src_paths = [os.path.join(source_images_dir, f)
                 for f in natsorted(os.listdir(source_images_dir))]

    failed = []
    for img_path in tqdm(src_paths,
                         desc=f"Normalizing images with {method}"):
        try:
            base = os.path.basename(img_path)
            out_path = os.path.join(
                normalized_images_dir,
                os.path.splitext(base)[0] + "_normalized.png")

            if os.path.exists(out_path):  # skip if done
                continue

            img_rgb = _load_rgb(img_path)
            norm_rgb = normalizer.transform(img_rgb)
            _save_rgb(norm_rgb, out_path)
        except Exception as e:
            failed.append((img_path, str(e)))

    print("\nStain normalization finished.")
    print(f"Failed files: {len(failed)}" if failed else "All images processed successfully.")
    return failed
