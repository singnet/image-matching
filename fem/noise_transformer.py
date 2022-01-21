import numpy
import torch

from fem.util import swap_rows, project_points2
from hom import HomographySamplerTransformer
from fem import util
from transform import TransformCompose, ToTensor


def collate(list_of_dicts):
    keys = list_of_dicts[0].keys()
    result = {key: torch.stack([item[key] for item in list_of_dicts]) for key in keys}
    return result


def to_torch(kwargs):
    return {k: (torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v) for (k, v) in kwargs.items()}


class NoisyTransformWithResize(TransformCompose):
    def __init__(self,
            num=1,
            beta=14,
            theta=0.8,
            random_scale_range=(0.8, 1.3),
            perspective=85,
            noisy=None):
        from fem import noise
        self.noisy = noisy
        if noisy is None:
            # keep old noise as default
            # there is code depending on it
            from fem.training import make_noisy_transformers
            self.noisy = make_noisy_transformers()

        self.imgcrop = noise.RandomCropTransform(size=256, beta=0)
        self.resize = noise.Resize((256, 256))
        self.to_tensor = ToTensor()
        self.homography = HomographySamplerTransformer(num=num,
                                                  beta=beta,
                                                  theta=theta,
                                                  random_scale_range=random_scale_range,
                                                  perspective=perspective)
        self.num = num

    def __call__(self, data=None, target=None):
        # crop
        image, pos = self.imgcrop(data, return_pos=True)
        resized = self.resize(image)
        img = self.to_tensor(resized)
        tmp = []
        if self.num == 1:
            return self.sample(img)
        for i in range(self.num):
            tmp.append(to_torch(self.sample(img)))
        result = collate(tmp)
        result['img1'] = result['img1'][0]
        return result

    def sample(self, x):
        return self.sample_with_source_homography(x)

    def sample_with_source_homography(self, x):
        assert x.shape[0] < x.shape[1]
        assert x.shape[0] < x.shape[2]
        # x is batch 1, h, w
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        # apply noise to source image before sample
        to_sample = self.noisy(x.permute(1, 2, 0).cpu().numpy()).permute(2, 0, 1)
        template2, hom2, mask2 = self.homography(to_sample.permute(1, 2, 0))
        # use different noise for training
        source = self.noisy(x.permute(1, 2, 0).cpu().numpy())
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        template1, hom1, mask1 = self.homography(source)
        H1 = hom1[0][0:3]
        H2 = hom2[0][0:3]
        H1_inv = hom1[0][3:6]
        H2_inv = hom2[0][3:6]
        H12 = H1_inv @ H2
        H12_inv = H2_inv @ H1
        # from fem.hom import bilinear_sampling
        # res12 = bilinear_sampling(template1[1], H12,
        #                           h_template=template1[1].shape[0],
        #                           w_template=template1[1].shape[1],
        #                           to_numpy=True, mode='bilinear')
        # res21 = bilinear_sampling(template2[1], H12_inv,
        #                           h_template=template1[1].shape[0],
        #                           w_template=template1[1].shape[1],
        #                           to_numpy=True, mode='bilinear')
        # import cv2
        # cv2.imshow('template1', template1[1] / 256.)
        # cv2.imshow('source', (source / 256.).numpy())
        # cv2.imshow('template2', template2[1] / 256.)
        # cv2.imshow('1 -> 2', res12 / 256.)
        # cv2.imshow('2 -> 1', res21 / 256.)
        # cv2.waitKey(100)
        return dict(img1=template1[1].squeeze(), img2=template2[1].squeeze(),
                    H=H12, H_inv=H12_inv, H1=H1, H2=H2,
                    H1_inv=H1_inv, H2_inv=H2_inv,
                    mask2=mask2, mask1=mask1)

    def sample_no_source_homography(self, x):
        # x is batch 1, h, w
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        # apply noise to source image before sample
        to_sample = self.noisy(x.permute(1, 2, 0).cpu().numpy()).permute(2, 0, 1)
        template, hom, mask2 = self.homography(to_sample.permute(1, 2, 0))
        mask1 = numpy.ones_like(mask2)
        # use different noise for training
        source = self.noisy(x.permute(1, 2, 0).cpu().numpy())
        #import pdb;pdb.set_trace()
        #import cv2
        #cv2.imshow('template[1]', template[1] / 256. * mask[0][..., numpy.newaxis])
        #cv2.imshow('source', (source / 256.).numpy())
        #cv2.waitKey(100)
        return dict(img1=source.squeeze(), img2=torch.from_numpy(template[1].squeeze()),
                    H=torch.from_numpy(hom[0][0:3]), H_inv=hom[0][3:6], mask2=mask2, mask1=mask1)


class NoisySimpleTransformWithResize(TransformCompose):
    def __init__(self, num=1):
        from fem import noise
        from fem.training import make_noisy_transformers
        self.noisy = make_noisy_transformers()

        self.imgcrop = noise.RandomCropTransform(size=256, beta=0)
        self.resize = noise.Resize((256, 256))
        self.to_tensor = ToTensor()
        self.homography = HomographySamplerTransformer(num=1,
                                                  beta=14,
                                                  theta=0.08,
                                                  random_scale_range=(0.8, 1.3),
                                                  perspective=85)
        self.num = num

    def __call__(self, data=None, target=None):
        # crop
        image, pos = self.imgcrop(data, return_pos=True)
        resized = self.resize(image)
        img = self.to_tensor(resized)
        tmp = []
        for i in range(self.num):
            tmp.append({'img1': self.noisy(img.permute(1, 2, 0).numpy())})
        result = collate(tmp)
        return result

class NoisyORB(NoisyTransformWithResize):
    def __init__(self, num=1, num_orb=300):
        super().__init__(num=num)
        from fem.orb import Orb
        self.orb = Orb(nfeatures=num_orb)

    def sample_with_source_homography(self, x):
        """
        Apply homography to image x

        Function will maxpool orb points and find subset of them which are reproduced on original
        and homographically warped image.

        :param x:
        :return: dict
        """

        # x is batch 1, h, w
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        template2, hom2, mask2 = self.homography(x.permute(1, 2, 0))

        orb_points2 = self.orb.compute(template2[1].squeeze())
        template2 = numpy.transpose(self.noisy(template2[1]), (2, 0, 1)) * mask2
        # use different noise for training
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        template1, hom1, mask1 = self.homography(x.permute(1, 2, 0))
        orb_points1 = self.orb.compute(template1[1].squeeze())
        template1 = numpy.transpose(self.noisy(template1[1]), (2, 0, 1)) * mask1
        H1 = hom1[0][0:3]
        H2 = hom2[0][0:3]
        H1_inv = hom1[0][3:6]
        H2_inv = hom2[0][3:6]
        H12 = H1_inv @ H2
        H12_inv = H2_inv @ H1

        #from fem.hom import bilinear_sampling
        #res12 = bilinear_sampling(template1[0].unsqueeze(2), H12,
        #                          h_template=template1[0].shape[0],
        #                          w_template=template1[0].shape[1],
        #                          to_numpy=True, mode='bilinear')
        #res21 = bilinear_sampling(template2[0].unsqueeze(2), H12_inv,
        #                          h_template=template2[0].shape[0],
        #                          w_template=template2[0].shape[1],
        #                          to_numpy=True, mode='bilinear')
        #import cv2
        #cv2.imshow('temp1', template1[0].numpy() / 256.)
        #cv2.imshow('src', (x[0] / 256.).numpy())
        #cv2.imshow('temp2', template2[0].numpy() / 256.)
        #cv2.imshow('1 -> 2', res12 / 256.)
        #cv2.imshow('2 -> 1', res21 / 256.)
        #import drawing
        #drawing.show_points(template1[0] / 255.0, orb_points1.nonzero(), 'img1')
        #drawing.show_points(template2[0] / 255.0, orb_points2.nonzero(), 'img2')

        #cv2.waitKey(300)
        #import pdb;pdb.set_trace()

        return dict(img1=template1.squeeze(), img2=template2.squeeze(),
                    H=H12, H_inv=H12_inv, mask2=mask2, mask1=mask1, points1=orb_points1, points2=orb_points2)


class NoisyORBSingle(NoisyORB):

    def sample_with_source_homography(self, x):
        """
        Apply homography to image x

        Function will maxpool orb points and project them
        from original to homographically warped image

        :param x:
        :return: dict
        """

        # x is batch 1, h, w
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        template2, hom2, mask2 = self.homography(x.permute(1, 2, 0))

        orb_points2 = self.orb.compute(template2[1].squeeze())
        template2 = numpy.transpose(self.noisy(template2[1]), (2, 0, 1)) * mask2
        # use different noise for training
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        template1, hom1, mask1 = self.homography(x.permute(1, 2, 0))
        orb_points1 = self.orb.compute(template1[1].squeeze())
        template1 = numpy.transpose(self.noisy(template1[1]), (2, 0, 1)) * mask1
        H1 = hom1[0][0:3]
        H2 = hom2[0][0:3]
        H1_inv = hom1[0][3:6]
        H2_inv = hom2[0][3:6]
        H12 = H1_inv @ H2
        H12_inv = H2_inv @ H1
        img_h, img_w = x.shape[-2:]

        in_bounds, points1projected = util.project_points(H12_inv, None, orb_points1.nonzero())
       # in_bounds, points1_projected = project_points2(torch.from_numpy(H12),
       #                                                None,
       #                                                orb_points1.nonzero(),
       #                                                img_h,
       #                                                img_w)
        points12 = points1projected[in_bounds]
        from fem.hom import bilinear_sampling
        res12 = bilinear_sampling(template1[0].unsqueeze(2), H12,
                                  h_template=template1[0].shape[0],
                                  w_template=template1[0].shape[1],
                                  to_numpy=True, mode='bilinear')
        res21 = bilinear_sampling(template2[0].unsqueeze(2), H12_inv,
                                  h_template=template2[0].shape[0],
                                  w_template=template2[0].shape[1],
                                  to_numpy=True, mode='bilinear')
        import cv2
        cv2.imshow('temp1', template1[0].numpy() / 256.)
        cv2.imshow('src', (x[0] / 256.).numpy())
        cv2.imshow('temp2', template2[0].numpy() / 256.)
        cv2.imshow('1 -> 2', res12 / 256.)
        cv2.imshow('2 -> 1', res21 / 256.)
        import drawing
        drawing.show_points(template1[0] / 255.0, orb_points1.nonzero(), 'img1')
        drawing.show_points(template2[0] / 255.0, orb_points2.nonzero(), 'img2')
        drawing.show_points(template2[0] / 255.0, points12, 'img2_points1')

        cv2.waitKey()
        import pdb;pdb.set_trace()

        return dict(img1=template1.squeeze(), img2=template2.squeeze(),
                    H=H12, H_inv=H12_inv, mask2=mask2, mask1=mask1, points1=orb_points1, points2=orb_points2)


