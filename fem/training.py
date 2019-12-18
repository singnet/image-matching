import sys
import torch

from torchvision.transforms import Compose
from ignite.engine import Engine

from fem import util
from fem.noise import Resize, ResizeKeypoints
from fem.transform import ToTensor, TransformCompose, RandomTransformer


def setup_engine(Engine, train, model, device, optimizer, scheduler,
                 print_every=100,
                 **kwargs):
    engine = Engine(lambda eng, batch: train(batch, model=model, device=device,
                                            optimizer=optimizer, scheduler=scheduler,
                                             **kwargs))
    util.add_moving_average(engine, 'desc_precession',
                            decay=0.8)
    util.add_moving_average(engine, 'det_loss', decay=0.99)
    util.add_moving_average(engine, 'desc_loss', decay=0.99)
    util.add_moving_average(engine, 'loss', decay=0.99)
    util.add_metrics(engine, True, print_every=print_every)
    return engine


def train_loop(engine: Engine, test_engine: Engine,
               train_loader, test_loader,
               epochs, optimizer,
               sp, **kwargs):
    for epoch in range(1, epochs):
        if test_engine is not None:
            print('testing')
            test_engine.run(test_loader)
        print('done testing')
        print('start epoch ' + str(epoch))
        if engine.state is not None:
            engine.state.epoch = epoch
        engine.run(train_loader)
        state = {'superpoint': sp.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, "super{0}.pt".format(epoch))
        print("epoch {} completed".format(epoch))
    if test_engine is not None:
        print('testing')
        test_engine.run(test_loader)
        print('done testing')


def create_transfromers(size=None):
    totensor = ToTensor()
    transformer_x = []
    transformer_y = []
    if size is not None:
        transformer_x.append(Resize(size))
        transformer_y.append(ResizeKeypoints(size))
    transformer_x.append(totensor)
    transformer_y.append(totensor)
    input_transform = Compose(transformer_x)
    label_transform = Compose(transformer_y)
    return label_transform, input_transform


class TransformWithResize(TransformCompose):
    def __init__(self):
        from fem import noise
        self.imgcrop = noise.RandomCropTransform(size=256, beta=20)
        self.resize = noise.Resize((256, 256))
        self.to_tensor = ToTensor()

    def __call__(self, data=None, target=None):

        assert data.shape[0] == target.shape[0]
        # crop
        image, pos = self.imgcrop(data, return_pos=True)
        row = pos[0]
        col = pos[1]
        h = image.shape[0]
        w = image.shape[1]
        assert len(target.shape) == 2
        target_crop = target[row:row + h, col:col + w]
        assert target_crop.shape == (h, w)
        resized = self.resize(image)
        resized_target = self.resize(target_crop)
        return self.to_tensor(resized), self.to_tensor(resized_target)


class TransformWithResizeThreshold(TransformCompose):
    def __init__(self):
        from fem import photometric
        from fem.nonmaximum import PoolingNms
        self.nms = PoolingNms(8)
        self.imgcrop = photometric.RandomCropTransform(size=256, beta=0)
        self.resize = photometric.Resize((256, 256))
        self.target_threshold = photometric.Threshold(0.25)
        self.target_resize = photometric.ResizeKeypoints((256, 256))
        self.to_tensor = ToTensor()

    def __call__(self, data=None, target=None):

        assert data.shape[0] == target.shape[0]
        # crop
        image, pos = self.imgcrop(data, return_pos=True)
        row = pos[0]
        col = pos[1]
        h = image.shape[0]
        w = image.shape[1]
        assert len(target.shape) == 2
        target_crop = target[row:row + h, col:col + w]
        target_nms = self.nms(torch.from_numpy(target_crop).unsqueeze(0).unsqueeze(0)).numpy().squeeze()
        assert target_nms.shape == (h, w)
        target_thres = self.target_threshold(target_nms)
        resized = self.resize(image)
        resized_target = self.target_resize(target_thres)
        return self.to_tensor(resized), self.to_tensor(resized_target)


def make_noisy_transformers():
    from fem.noise import AdditiveGaussian, RandomBrightness, AdditiveShade, MotionBlur, SaltPepper, RandomContrast
    totensor = ToTensor()
    # ColorInversion doesn't seem to be usefull on most datasets
    transformer = [
                   AdditiveGaussian(var=30),
                   RandomBrightness(range=(-50, 50)),
                   AdditiveShade(kernel_size_range=[45, 85],
                                 transparency_range=(-0.25, .45)),
                   SaltPepper(),
                   MotionBlur(max_kernel_size=5),
                   RandomContrast([0.6, 1.05])
                   ]
    return Compose([RandomTransformer(transformer), totensor])

def train_3d_airsim(batch, sp):
    imgs = torch.cat([batch['img1'], batch['img2']]).float()
    height, width = imgs.shape[1], imgs.shape[2]
    assert len(imgs.shape) == 3
    sp = sp.eval()
    heatmaps, desc = sp.forward(imgs.unsqueeze(1) / 255.0)
    sl1 = slice(0, len(imgs) // 2)
    sl2 = slice(len(imgs) // 2, len(imgs))



    desc_orig = desc[sl1]
    desc_offset = desc[sl2]
    K1 = batch['K1']
    K2 = batch['K2']
    depth1 = batch['depth1']
    depth2 = batch['depth2']
    pos1 = batch['pose1']
    pos2 = batch['pose2']
    points_orig = batch['points1'][:,1]
    points_offset = batch['points2'][:,1]
    # print((points_orig > 0.3).nonzero().shape)
    # print((points_offset > 0.3).nonzero().shape)
    det_no_homography = heatmaps[sl1]
    keypoints_homography = heatmaps[sl2]


    points_orig = sp.nms(points_orig.unsqueeze(1)).squeeze()
    points_offset = sp.nms(points_offset.unsqueeze(1)).squeeze()

    kwargs = dict(K1=K1,
                  K2=K2,
                  depth1=depth1,
                  pose1=pos1, pose2=pos2,
                  desc1=desc_orig, desc2=desc_offset,
                  hom_target=points_offset > 0.2,
                  target=points_orig > 0.2,
                  height=height,
                  width=width,
                  img_1=batch['img1'],
                  img_2=batch['img2'])
    desc_loss, prec = interpolated_desc_3d_loss(**kwargs)


    kwargs_det = dict(det_no_homography=det_no_homography,
                      target=batch['points1'],
                      hom_target=batch['points2'],
                      keypoints_homography=keypoints_homography,
                      sd_loss_det=False,
                      only_positive=False,
                      soft=False,
                      valid_mask=1.0, weighted=True)
    det_loss = compute_det_loss(**kwargs_det)
    loss = desc_loss + det_loss
    # loss = desc_loss
    result = form_result(desc_loss, det_loss, batch['points2'], keypoints_homography, loss, prec)
    return result



def train(batch, model:'SuperPoint', device, optimizer, scheduler=None, loss_function=None, **loss_kw):
    model.train()
    data, target = batch
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    model.zero_grad()

    result = loss_function(model, data, target, **loss_kw)

    # result = model.loss(data, target,
    #                     compute_precession=True,
    #                     visualize=True,
    #                     noisy=noisy)

    loss = result['loss']

    loss.backward()
    result['lr'] = scheduler.get_lr()
    optimizer.step()
    # since pytorch 1.1 scheduler should be called after optimizer
    scheduler.step()
    sys.stdout.flush()
    result['loss'] = result['loss'].detach()
    return result


def test(engine, batch, model, device, loss_function, **loss_kw):
    model.eval()
    data, target = batch
    data, target = data.float().to(device), target.to(device)
    model.zero_grad()
    result = loss_function(model, data, target, **loss_kw)
    result['loss'] = result['loss'].cpu().detach().numpy()
    return result


def exponential_lr(decay_rate, global_step, decay_steps, staircase=False):
    if staircase:
        return decay_rate ** (global_step // decay_steps)
    return decay_rate ** (global_step / decay_steps)


def on_iteration_completed(engine, print_every=100):
    iteration = engine.state.iteration
    if iteration % print_every == 0:
        loss = engine.state.output.get('loss', '')
        lr = engine.state.output.get('lr', '')
        print("Iteration: {}, Loss: {}, lr {}".format(iteration, loss, lr))
        print("metrics {0}".format({k: float(v) for
                                    k, v in engine.state.metrics.items()}))


def form_result(desc_loss, det_loss, hom_target, keypoints_homography, loss, prec=None):
    result = dict()
    if prec is not None:
        result['desc_precession'] = prec
    result['loss'] = loss
    result['desc_loss'] = desc_loss.detach().cpu().numpy()
    result['det_loss'] = det_loss.detach().cpu().numpy()
    result['det_target'] = hom_target.cpu().detach().numpy()[:, 1]
    result['det_prob'] = keypoints_homography.cpu().detach().numpy()[:, 1]
    return result
