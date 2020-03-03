import math
# from urllib.request import urlretrieve
import requests
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)
class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


    def __str__(self):
        return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            # filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
            data = requests.get(url)
            with open(destination, 'wb') as f:
                f.write(data.content)
    else:
        data = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(data.content)
        # filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False, n_class=157):
        super(AveragePrecisionMeter, self).__init__()
        self.n_class = n_class
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.sample_num = 0.
        self.ap = 0
        self.Nc = np.zeros(self.n_class)
        self.Np = np.zeros(self.n_class)
        self.Ng = np.zeros(self.n_class)
        
        self.ap_topk = 0
        self.Nc_topk = np.zeros(self.n_class)
        self.Np_topk = np.zeros(self.n_class)
        self.Ng_topk = np.zeros(self.n_class)

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if output.numel() > 0:
            assert target.size(1) == output.size(1), \
                'dimensions for output should match target.'

        for k in range(output.size(0)):
            # sort scores
            outputk = output[k, :]
            targetk = target[k, :]
            # compute average precision
            self.ap += AveragePrecisionMeter.average_precision(outputk, targetk, self.difficult_examples)
            self.sample_num += 1

        Nc = np.zeros(self.n_class)
        Np = np.zeros(self.n_class)
        Ng = np.zeros(self.n_class)
        for i in range(self.n_class):
            outputk = output[:, i]
            targetk = target[:, i]
            Ng[i] = sum(targetk == 1)
            Np[i] = sum(outputk >= 0)
            Nc[i] = sum(targetk * (outputk >= 0).type(torch.FloatTensor))
        
        self.Nc += Nc
        self.Np += Np
        self.Ng += Ng

        n = output.shape[0]
        output_topk = torch.zeros((n, self.n_class)) - 1
        index = output.topk(3, 1, True, True)[1].cpu().numpy()
        tmp = output.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                output_topk[i, ind] = 1 if tmp[i, ind] >= 0 else - 1
        for k_topk in range(output_topk.shape[0]):
            # sort scores
            outputk_topk = output_topk[k_topk, :]
            targetk_topk = target[k_topk, :]
            # compute average precision
            self.ap_topk += AveragePrecisionMeter.average_precision(outputk_topk, targetk_topk, self.difficult_examples)

        Nc_topk = np.zeros(self.n_class)
        Np_topk = np.zeros(self.n_class)
        Ng_topk = np.zeros(self.n_class)
        for i in range(self.n_class):
            outputk_topk = output_topk[:, i]
            targetk_topk = target[:, i]
            Ng_topk[i] = sum(targetk_topk == 1)
            Np_topk[i] = sum(outputk_topk >= 0)
            Nc_topk[i] = sum(targetk_topk * (outputk_topk >= 0).type(torch.FloatTensor))
        
        self.Nc_topk += Nc_topk
        self.Np_topk += Np_topk
        self.Ng_topk += Ng_topk

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each sample k
        """
        ap = self.ap / self.sample_num
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=False):
        
        # sort examples
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
            
        _, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        precision_at_i_list = torch.zeros(int(target.sum()))
        for i in indices:
            label = target[i]
            total_count += 1
            if label == 1:
                pos_count += 1
                precision_at_i_list[int(pos_count) - 1] = pos_count / total_count

        for i in range(len(precision_at_i_list) - 1, 0, -1):
            if precision_at_i_list[i - 1] < precision_at_i_list[i]:
                precision_at_i_list[i - 1] = precision_at_i_list[i]

        precision_at_i = precision_at_i_list.mean()
        return precision_at_i

    def overall(self):
        if self.sample_num == 0:
            return 0
        
        OP = sum(self.Nc) / max(sum(self.Np), 1e-8)
        OR = sum(self.Nc) / max(sum(self.Ng), 1e-8)
        OF1 = (2 * OP * OR) / max(OP + OR, 1e-8)

        clip_Np = self.Np.copy()
        for i in range(len(self.Np)):
            if self.Np[i] < 1e-8:
                clip_Np[i] = 1e-8

        clip_Ng = self.Ng.copy()
        for i in range(len(self.Ng)):
            if self.Ng[i] < 1e-8:
                clip_Ng[i] = 1e-8

        CP = sum(self.Nc / clip_Np) / self.n_class
        CR = sum(self.Nc / clip_Ng) / self.n_class
        CF1 = (2 * CP * CR) / max(CP + CR, 1e-8)
        
        return OP, OR, OF1, CP, CR, CF1

    def overall_topk(self, k):
        '''
        k is useless
        '''
        if self.sample_num == 0:
            return 0
        
        OP = sum(self.Nc_topk) / max(sum(self.Np_topk), 1e-8)
        OR = sum(self.Nc_topk) / max(sum(self.Ng_topk), 1e-8)
        OF1 = (2 * OP * OR) / max(OP + OR, 1e-8)

        clip_Np_topk = self.Np_topk.copy()
        for i in range(len(self.Np_topk)):
            if self.Np_topk[i] < 1e-8:
                clip_Np_topk[i] = 1e-8

        clip_Ng_topk = self.Ng_topk.copy()
        for i in range(len(self.Ng_topk)):
            if self.Ng_topk[i] < 1e-8:
                clip_Ng_topk[i] = 1e-8

        CP = sum(self.Nc_topk / clip_Np_topk) / self.n_class
        CR = sum(self.Nc_topk / clip_Ng_topk) / self.n_class
        CF1 = (2 * CP * CR) / max(CP + CR, 1e-8)
        
        return OP, OR, OF1, CP, CR, CF1

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
