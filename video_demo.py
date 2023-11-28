python

class VideoDemo:
    def __init__(self, video, config, checkpoint, device='cuda:0', score_thr=0.3, out=None, show=False, wait_time=1):
        self.video = video
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.score_thr = score_thr
        self.out = out
        self.show = show
        self.wait_time = wait_time

    def run(self):
        assert self.out or self.show, \
            ('Please specify at least one operation (save/show the '
             'video) with the argument "--out" or "--show"')

        model = init_detector(self.config, self.checkpoint, device=self.device)

        video_reader = mmcv.VideoReader(self.video)
        video_writer = None
        if self.out:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                self.out, fourcc, video_reader.fps,
                (video_reader.width, video_reader.height))

        for frame in mmcv.track_iter_progress(video_reader):
            result = inference_detector(model, frame)
            frame = model.show_result(frame, result, score_thr=self.score_thr)
            if self.show:
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame, 'video', self.wait_time)
            if self.out:
                video_writer.write(frame)

        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


