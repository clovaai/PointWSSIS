import torch
import torchvision.transforms.functional as F
import random
import numbers
import numpy as np
from PIL import Image
import collections

try:
    interpolation_bilinear = F.InterpolationMode.BILINEAR
    interpolation_nearest = F.InterpolationMode.NEAREST
except:
    interpolation_bilinear = Image.BILINEAR
    interpolation_nearest = Image.NEAREST

#
#  Extended Transforms for Semantic Segmentation
#

class ExtCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, lbls):
        for t in self.transforms:
            imgs, lbls = t(imgs, lbls)
        return imgs, lbls

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ExtCenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs, lbls):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if lbls is None:
            return [F.center_crop(img, self.size) for img in imgs], lbls
        
        return [F.center_crop(img, self.size) for img in imgs], [F.center_crop(lbl, self.size) for lbl in lbls]

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class ExtRandomScaleResize(object):
    def __init__(self, size, scale_range, interpolation="nearest"):
        self.size = size
        self.scale_range = scale_range
        self.interpolation = (
            interpolation_nearest
            if interpolation == "nearest"
            else interpolation_bilinear
        )

    def __call__(self, imgs, lbls):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        #assert imgs[0].size == lbls[0].size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        w, h = imgs[0].size

        if w >= h:
            ratio = w / h
            new_h = self.size
            new_w = int(ratio * new_h)
        else:
            ratio = h / w
            new_w = self.size
            new_h = int(ratio * new_w)

        target_size = (int(new_h * scale), int(new_w * scale))
        
        if lbls is None:
            return [F.resize(img, target_size, interpolation_bilinear) for img in imgs], lbls
        
        return [F.resize(img, target_size, interpolation_bilinear) for img in imgs], [F.resize(lbl, target_size, self.interpolation) for lbl in lbls]

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


    

class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, lbls):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            if lbls is None:
                return [F.hflip(img) for img in imgs], lbls
            
            return [F.hflip(img) for img in imgs], [F.hflip(lbl) for lbl in lbls]
        return imgs, lbls

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class ExtPad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser

    def __call__(self, imgs, lbls):
        h, w = imgs[0].size
        ph = (h // 32 + 1) * 32 - h if h % 32 != 0 else 0
        pw = (w // 32 + 1) * 32 - w if w % 32 != 0 else 0
        imgs = [F.pad(img, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)) for img in imgs]
        
        if lbls is not None:
            lbls = [F.pad(lbl, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)) for lbl in lbls]

        return imgs, lbls


class ExtToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, normalize=True, target_type="float32"):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, imgs, lbls):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor.
        Returns:
            Tensor: Converted image and label
        """
        if self.normalize:
            if lbls is None:
                return [F.to_tensor(img) for img in imgs], lbls
            
            return [F.to_tensor(img) for img in imgs], \
                    [torch.from_numpy(np.array(lbl, dtype=self.target_type)) for lbl in lbls]
        else:
            if lbls is None:
                return [torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1)) for img in imgs], lbls
            
            return [torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1)) for img in imgs], \
                    [torch.from_numpy(np.array(lbl, dtype=self.target_type)) for lbl in lbls]

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ExtNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs, lbls):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return [F.normalize(img, self.mean, self.std) for img in imgs], lbls

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, imgs, lbls):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        # assert (
        #     imgs[0].size == lbls[0].size
        # ), "size of img and lbl should be the same. %s, %s" % (img.size, lbl.size)
        if self.padding > 0:
            imgs = [F.pad(imgs, self.padding) for img in imgs]
            if lbls is not None:
                lbls = [F.pad(lbl, self.padding) for lbl in lbls]

        # pad the width if needed
        if self.pad_if_needed and imgs[0].size[0] < self.size[1]:
            imgs = [F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))  for img in imgs]
            if lbls is not None:
                lbls = [F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2)) for lbl in lbls]

        # pad the height if needed
        if self.pad_if_needed and imgs[0].size[1] < self.size[0]:
            imgs = [F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2)) for img in imgs]
            if lbls is not None:
                lbls = [F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2)) for lbl in lbls]

        i, j, h, w = self.get_params(imgs[0], self.size)

        if lbls is None:
            return [F.crop(img, i, j, h, w) for img in imgs], lbls
        
        return [F.crop(img, i, j, h, w) for img in imgs], [F.crop(lbl, i, j, h, w) for lbl in lbls]

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding
        )


class ExtResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation="nearest"):
        assert isinstance(size, int) or (
            isinstance(size, collections.Iterable) and len(size) == 2
        )
        self.size = size
        self.interpolation = (
            interpolation_nearest
            if interpolation == "nearest"
            else interpolation_bilinear
        )

    def __call__(self, imgs, lbls):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        if lbls is None:
            return [F.resize(img, self.size, interpolation_bilinear) for img in imgs], lbls
        
        return [F.resize(img, self.size, interpolation_bilinear) for img in imgs], \
                [F.resize(lbl, self.size, self.interpolation) for lbl in lbls]

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)

    


class ExtColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, imgs, lbls):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        if torch.rand(1) < self.p:
            transform = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
            imgs = [transform(img) for img in imgs]
            
        return imgs, lbls

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
