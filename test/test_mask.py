
import cv2

from mgen3d.mask import MaskExtractor


def test_mask_extraction():
    # load image with alpha channel
    # TODO: generate dummy image that is tmp instead of referencing a file
    img = cv2.imread('./data/test/teddy_bear.png', cv2.IMREAD_UNCHANGED)
    
    mask_extractor = MaskExtractor()
    mask_extractor.extract(img)