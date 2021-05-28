import torch


class RelativeResponseLoss:
    def __init__(self, scale, standard_sample_size, epsilon=1.0e-16):
        self._scale = scale
        self._epsilon = epsilon
        self._standard_sample_size = torch.tensor(standard_sample_size).float().cuda()

    def _loss(self, source_F, target_F, pos_pairs, get_heatmap):
        source_point_num, feature_length = source_F.shape
        target_point_num, _ = target_F.shape

        sampling_size = pos_pairs.shape[0]
        sampled_source_F = source_F[pos_pairs[:, 0].long()]
        # N x N1 x C
        # Assume the source and target features are already L2 normalized
        feature_responses = -0.5 * torch.sum(
            (sampled_source_F.reshape(-1, 1, feature_length) - target_F.reshape(1, -1, feature_length)) ** 2,
            dim=2)
        # N x N1
        feature_responses = torch.exp(self._scale * feature_responses)
        feature_responses = feature_responses / (torch.sum(feature_responses, dim=(1,), keepdim=True) + self._epsilon)

        # Sampling_size x 1
        sampled_feature_responses = torch.gather(feature_responses, dim=1,
                                                 index=pos_pairs[:, 1].long().reshape(sampling_size, 1))
        # Calibrate the rr loss value to make it invariant to the size of the samples
        rr_loss = -torch.log(sampled_feature_responses + self._epsilon) - torch.log(
            target_point_num / self._standard_sample_size)
        rr_loss = torch.sum(rr_loss) / sampling_size
        if get_heatmap:
            return rr_loss, feature_responses[0, :] / torch.max(feature_responses[0, :]), pos_pairs[0]
        else:
            return rr_loss

    def __call__(self, input0, input1, pos_pairs):
        return self._loss(input0, input1, pos_pairs, False)
