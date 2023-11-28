
这个程序文件名为yolov5-NWD.py，主要实现了一个名为wasserstein_loss的函数，用于计算目标检测中的损失函数。该函数的实现参考了论文《Enhancing Geometric Factors into Model Learning and Inference for Object Detection and Instance Segmentation》。

wasserstein_loss函数接受两个参数pred和target，分别表示预测的边界框和真实的边界框。其中，pred和target的形状都为(n, 4)，表示n个边界框，每个边界框由(x_center, y_center, w, h)四个值组成。

函数首先从pred和target中提取出中心点坐标center1和center2，然后计算中心点坐标之间的距离center_distance。接着，计算预测边界框的宽度w1和高度h1，以及真实边界框的宽度w2和高度h2。然后，计算宽度和高度之间的距离wh_distance。

最后，根据中心点距离和宽度高度距离计算wasserstein_2，并返回torch.exp(-torch.sqrt(wasserstein_2) / constant)作为损失值。

在主程序中，首先调用wasserstein_loss函数计算nwd，然后根据nwd和iou_ratio计算lbox的值。最后，根据iou和nwd的值计算iou_loss，并将其加到lbox上。最终得到的lbox即为目标检测的损失值。



#### 5.3 demo\create_result_gif.py
