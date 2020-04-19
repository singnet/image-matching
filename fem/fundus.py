from fem.noise_transformer import NoisyTransformWithResize, ToTensor, HomographySamplerTransformer


class FundusTransformWithResize(NoisyTransformWithResize):
    def __init__(self, num=1):
        super().__init__(num=num)
        from fem import noise
        self.imgcrop = noise.RandomCropTransform(size=1408, beta=200)
        # self.noisy = lambda x: self.to_tensor(x)
        self.homography = HomographySamplerTransformer(num=1,
                                                  beta=14,
                                                  theta=0.04,
                                                  random_scale_range=(0.8, 1.3),
                                                  perspective=25)


