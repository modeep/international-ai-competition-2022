# Import Library
import cv2
from layers.functions import Detect
from m2det import build_net
from data import BaseTransform
from utils.core import *


class ClassBar:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Classber')
        parser.add_argument('-c', '--config', default='configs/m2det512_vgg.py', type=str)
        parser.add_argument('-m', '--trained_model', default='weights/m2det512_vgg.pth', type=str,
                            help='Trained state_dict file path to open')
        parser.add_argument('--cam', default=0, type=int, help='camera device id')
        parser.add_argument('--show', default=True, action='store_true', help='Whether to display the images')
        args = parser.parse_args()

        print_info(' ----------------------------------------------------------------------\n'
                   '|                              Classber                                |\n'
                   ' ----------------------------------------------------------------------', ['yellow', 'bold'])

        self.cfg = Config.fromfile(args.config)
        anchor_config = anchors(self.cfg)
        print_info('The Anchor info: \n{}'.format(anchor_config))
        prior_box = PriorBox(anchor_config)
        net = build_net('test',
                        size=self.cfg.model.input_size,
                        config=self.cfg.model.m2det_config)
        init_net(net, self.cfg, args.trained_model)
        print_info('===> Finished constructing and loading model', ['yellow', 'bold'])
        net.eval()
        with torch.no_grad():
            priors = prior_box.forward()
            if self.cfg.test_cfg.cuda:
                self.net = net.cuda()
                self.priors = priors.cuda()
                cudnn.benchmark = True
            else:
                self.net = self.net.cpu()
        self._preprocess = BaseTransform(self.cfg.model.input_size, self.cfg.model.rgb_means, (2, 0, 1))
        self.detector = Detect(self.cfg.model.m2det_config.num_classes, self.cfg.loss.bkg_label, anchor_config)

        self.im_path = args.directory
        self.cam = args.cam

    def draw_detection(self, im, bboxes, scores, cls_inds, thr=0.8):
        imgcv = np.copy(im)
        h, w, _ = imgcv.shape
        boxs = []
        boxs_area = []
        boxs_score = []
        for i, box in enumerate(bboxes):
            cls_indx = int(cls_inds[i])
            if cls_indx == 40 or cls_indx == 42:
                if scores[i] < thr:
                    continue
                else:
                    boxs.append([box[0], box[1], box[2], box[3]])
                    boxs_area.append(((box[2] - box[0]) * (box[3] - box[1])))
                    boxs_score.append(scores[i])
            else:
                continue

        thick = int((h + w) / 300)

        if len(boxs_area) == 0:
            return imgcv, None
        else:
            n = boxs_area.index(max(boxs_area))
            cv2.rectangle(imgcv,
                          (int(boxs[n][0]), int(boxs[n][1])), (int(boxs[n][2]), int(boxs[n][3])),
                          (0, 0, 255), thick)
            mess = 'bottle: %.3f' % (boxs_score[n])
            cv2.putText(imgcv, mess, (int(boxs[n][0]), int(boxs[n][1]) - 7),
                        0, 1e-3 * h, (0, 0, 255), thick // 3)
            return imgcv, boxs[n]

    # def run_model(image:np.ndarray) -> Union[str, tuple, bool]:
    def run_model(self, image):
        # cap = cv2.VideoCapture(cam)
        # _, image = cap.read()
        w, h, _ = image.shape

        img = self._preprocess(image).unsqueeze(0)

        if self.cfg.test_cfg.cuda:
            img = img.cuda()

        scale = torch.Tensor([w, h, w, h])
        out = self.net(img)
        boxes, scores = self.detector.forward(out, self.priors)
        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []

        for j in range(1, self.cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:, j] > self.cfg.test_cfg.score_threshold)[0]

            if len(inds) == 0:
                continue

            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            soft_nms = self.cfg.test_cfg.soft_nms
            keep = nms(c_dets, self.cfg.test_cfg.iou, force_cpu=soft_nms)
            keep = keep[:self.cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist() + [j] for _ in c_dets])

        allboxes = np.array(allboxes)
        boxes = allboxes[:, :4]
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]
        cls_list = cls_inds.tolist()
        cls_list = list(map(int, cls_list))

        if 40 in cls_list or 42 in cls_list:
            for i in cls_list:
                if i == 40 or i == 42:
                    im2show, img_location = self.draw_detection(image, boxes, scores, cls_inds)
                    cv2.imshow("classber", im2show)
                    if cv2.waitKey():
                        cv2.destroyAllWindows()
                    return im2show, img_location
                else:
                    continue
        else:
            cv2.imshow("classber", image)
            if cv2.waitKey():
                cv2.destroyAllWindows()
            return image, None
