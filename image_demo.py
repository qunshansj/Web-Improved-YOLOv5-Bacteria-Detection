python

class ObjectDetector:
    def __init__(self, config, checkpoint, device='cuda:0', score_thr=0.3):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.score_thr = score_thr

    def detect(self, img_path):
        model = init_detector(self.config, self.checkpoint, device=self.device)
        result = inference_detector(model, img_path)
        show_result_pyplot(model, img_path, result, score_thr=self.score_thr)

    async def async_detect(self, img_path):
        model = init_detector(self.config, self.checkpoint, device=self.device)
        tasks = asyncio.create_task(async_inference_detector(model, img_path))
        result = await asyncio.gather(tasks)
        show_result_pyplot(model, img_path, result[0], score_thr=self.score_thr)


