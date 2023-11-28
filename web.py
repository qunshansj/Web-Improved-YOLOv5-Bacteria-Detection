python


class ObjectDetector:
    def __init__(self, weights='./best.pt', data='./data/coco128.yaml', device='', half=False, dnn=False):
        self.device = select_device(device)
        self.model = self.load_model(weights, data, half, dnn)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def load_model(self, weights, data, half, dnn):
        device = select_device(self.device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

        half &= (pt or jit or onnx or engine) and device.type != 'cpu'
        if pt or jit:
            model.model.half() if half else model.model.float()
        return model

    def detect_objects(self, img, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.05, max_det=1000, classes=None, agnostic_nms=False, augment=False, half=False):
        cal_detect = []

        im = letterbox(img, imgsz, self.model.stride, self.model.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=augment)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{self.names[c]}'
                    cal_detect.append([label, xyxy, float(conf)])
        return cal_detect
