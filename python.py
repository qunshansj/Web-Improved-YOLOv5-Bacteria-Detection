python
wasserstein_loss = WassersteinLoss()
nwd = wasserstein_loss(pbox, tbox[i]).squeeze()
iou_ratio = 0.5
lbox += (1 - iou_ratio) * (1.0 - nwd).mean() + iou_ratio * (1.0 - iou).mean()  # iou loss

# Objectness
iou = (iou.detach() * iou_ratio + nwd.detach() * (1 - iou_ratio)).clamp(0, 1).type(tobj.dtype)
