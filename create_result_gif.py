python

class GifCreator:
    def __init__(self, image_dir, out='result.gif'):
        self.image_dir = image_dir
        self.out = out

    def _generate_batch_data(self, sampler, batch_size):
        batch = []
        for idx in sampler:
            batch.append(idx)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def create_gif(self, frames, duration=2):
        """Create gif through imageio.

        Args:
            frames (list[ndarray]): Image frames
            duration (int): Display interval (s),
                Default: 2
        """
        if imageio is None:
            raise RuntimeError('imageio is not installed,'
                               'Please use “pip install imageio” to install')
        imageio.mimsave(self.out, frames, 'GIF', duration=duration)

    def create_frame_by_matplotlib(self, nrows=1, fig_size=(300, 300), font_size=15):
        """Create gif frame image through matplotlib.

        Args:
            nrows (int): Number of rows displayed, Default: 1
            fig_size (tuple): Figure size of the pyplot figure.
               Default: (300, 300)
            font_size (int): Font size of texts. Default: 15

        Returns:
            list[ndarray]: image frames
        """

        result_dir_names = os.listdir(self.image_dir)
        assert len(result_dir_names) == 2
        # Longer length has higher priority
        result_dir_names.reverse()

        images_list = []
        for dir_names in result_dir_names:
            images_list.append(mmcv.scandir(osp.join(self.image_dir, dir_names)))

        frames = []
        for paths in self._generate_batch_data(zip(*images_list), nrows):

            fig, axes = plt.subplots(nrows=nrows, ncols=2)
            fig.suptitle('Good/bad case selected according '
                         'to the COCO mAP of the single image')

            det_patch = mpatches.Patch(color='salmon', label='prediction')
            gt_patch = mpatches.Patch(color='royalblue', label='ground truth')
            # bbox_to_anchor may need to be finetuned
            plt.legend(
                handles=[det_patch, gt_patch],
                bbox_to_anchor=(1, -0.18),
                loc='lower right',
                borderaxespad=0.)

            if nrows == 1:
                axes = [axes]

            dpi = fig.get_dpi()
            # set fig size and margin
            fig.set_size_inches(
                (fig_size[0] * 2 + fig_size[0] // 20) / dpi,
                (fig_size[1] * nrows + fig_size[1] // 3) / dpi,
            )

            fig.tight_layout()
            # set subplot margin
            plt.subplots_adjust(
                hspace=.05,
                wspace=0.05,
                left=0.02,
                right=0.98,
                bottom=0.02,
                top=0.98)

            for i, (path_tuple, ax_tuple) in enumerate(zip(paths, axes)):
                image_path_left = osp.join(
                    osp.join(self.image_dir, result_dir_names[0], path_tuple[0]))
                image_path_right = osp.join(
                    osp.join(self.image_dir, result_dir_names[1], path_tuple[1]))
                image_left = mmcv.imread(image_path_left)
                image_left = mmcv.rgb2bgr(image_left)
                image_right = mmcv.imread(image_path_right)
                image_right = mmcv.rgb2bgr(image_right)

                if i == 0:
                    ax_tuple[0].set_title(
                        result_dir_names[0], fontdict={'size': font_size})
                    ax_tuple[1].set_title(
                        result_dir_names[1], fontdict={'size': font_size})
                ax_tuple[0].imshow(
                    image_left, extent=(0, *fig_size, 0), interpolation='bilinear')
                ax_tuple[0].axis('off')
                ax_tuple[1].imshow(
                    image_right,
                    extent=(0, *fig_size, 0),
                    interpolation='bilinear')
                ax_tuple[1].axis('off')

            canvas = fig.canvas
            s, (width, height) = canvas.print_to_buffer()
            buffer = np.frombuffer(s, dtype='uint8')
            img_rgba = buffer.reshape(height, width, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            img = rgb.astype('uint8')

            frames.append(img)

        return frames

    def create_gif_from_images(self):
        frames = self.create_frame_by_matplotlib()
        self.create_gif(frames, self.out)

