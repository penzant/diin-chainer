
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import cuda
import numpy

from utils import softsel, make_mask, \
    flat_linear, get_logits, interaction_layer

xp = cuda.cupy


class SelfAttentionNetwork(chainer.Chain):
    
    def __init__(self, config):
        super(SelfAttentionNetwork, self).__init__()
        
        with self.init_scope():
            # linear logits
            self.logits_linear = L.Linear(None, 1)

            # fuse gate
            fuse_dim = config.enc_dim
            self.fg_lhs_1 = L.Linear(None, fuse_dim)
            self.fg_rhs_1 = L.Linear(None, fuse_dim)
            self.fg_lhs_2 = L.Linear(None, fuse_dim)
            self.fg_rhs_2 = L.Linear(None, fuse_dim) 
            self.fg_lhs_3 = L.Linear(None, fuse_dim)
            self.fg_rhs_3 = L.Linear(None, fuse_dim)

        
    def __call__(self, p, p_mask=None):
        xp.cuda.Device(self._device_id).use()

        p_len = p.shape[1]
        p_aug_i = F.tile(F.expand_dims(p, 2), (1, 1, p_len, 1))
        p_aug_j = F.tile(F.expand_dims(p, 1), (1, p_len, 1, 1))

        if p_mask is None:
            self_mask = None
        else:
            p_mask_aug_i = F.tile(F.expand_dims(p_mask, 2), (1, 1, p_len, 1))
            p_mask_aug_i = xp.any(F.cast(p_mask_aug_i, 'bool').data, axis=3)
            p_mask_aug_j = F.tile(F.expand_dims(p_mask, 1), (1, p_len, 1, 1))
            p_mask_aug_j = xp.any(F.cast(p_mask_aug_j, 'bool').data, axis=3)
            self_mask = p_mask_aug_i & p_mask_aug_j

        h_logits = get_logits(self.logits_linear,
                              [p_aug_i, p_aug_j], self_mask) # ->(N,48,48)
        self_att = softsel(p_aug_j, h_logits) # ->(N,48,448)
        out = self.fuse_gate(p, self_att)

        return out

    def fuse_gate(self, lhs, rhs, drop_rate=0.0):
        lhs_1 = flat_linear(self.fg_lhs_1, F.dropout(lhs, drop_rate))
        rhs_1 = flat_linear(self.fg_rhs_1, F.dropout(rhs, drop_rate))
        z = F.tanh(lhs_1 + rhs_1)

        lhs_2 = flat_linear(self.fg_lhs_2, F.dropout(lhs, drop_rate))
        rhs_2 = flat_linear(self.fg_rhs_2, F.dropout(rhs, drop_rate))
        f = F.sigmoid(lhs_2 + rhs_2)

        lhs_3 = flat_linear(self.fg_lhs_3, F.dropout(lhs, drop_rate))
        rhs_3 = flat_linear(self.fg_rhs_3, F.dropout(rhs, drop_rate))
        f2 = F.sigmoid(lhs_3 + rhs_3)

        out = f * lhs + f2 * z
        return out


class HighwayNetwork(chainer.Chain):
    
    def __init__(self, config):
        super(HighwayNetwork, self).__init__()
        self.seq_length = config.seq_length
        self.enc_dim = config.enc_dim
        self.layer_num = config.highway_n_layer

        with self.init_scope():
            for i in range(self.layer_num):
                setattr(self, 'highway_layer_{0}'.format(i), L.Highway(self.enc_dim))

            if config.gpu[0] >= 0:
                for i in range(self.layer_num):
                    layer = getattr(self, 'highway_layer_{0}'.format(i))
                    layer.to_gpu(config.gpu[0])

    def __call__(self, h):
        h = h.reshape((-1, self.enc_dim))
        for i in range(self.layer_num):
            layer = getattr(self, 'highway_layer_{0}'.format(i))
            h = layer(h)
        h = h.reshape((-1, self.seq_length, self.enc_dim))

        return h


class CharacterConvolution(chainer.Chain):

    def __init__(self, config):
        super(CharacterConvolution, self).__init__()
        with self.init_scope():
            k = (1, config.char_conv_height)
            s = 1
            out_channel = config.char_conv_n_kernel
            self.conv_layer = L.Convolution2D(None, out_channel, k, s)

    def __call__(self, h):
        h = h.transpose(0, 3, 1, 2) # NHWC -> NCHW
        h = F.relu(self.conv_layer(h))
        h = F.max(h.transpose(0, 2, 3, 1), 2) # NCHW -> NHWC
        return h
            

class DenseNetBlock(chainer.Chain):

    def __init__(self, config, in_dim):
        super(DenseNetBlock, self).__init__()
        self.conv_num = config.dense_n_layer
        with self.init_scope():
            dense_ksize = config.dense_kernel_size
            for i in range(self.conv_num):
                setattr(self, 'conv_layer_{0}'.format(i),
                        L.Convolution2D(None, config.dense_growth_rate, ksize=dense_ksize,
                                        stride=1, pad=int(dense_ksize/2)))
    
            feat_dim = in_dim + config.dense_growth_rate * config.dense_n_layer
            feat_dim = int(feat_dim * config.dense_trans_scale_down_rate)
            self.transition_layer = L.Convolution2D(None, feat_dim, ksize=1)        

    def __call__(self, h):
        hs = [h]
        for i in range(self.conv_num):
            conv_layer = getattr(self, 'conv_layer_{0}'.format(i))
            h = F.relu(conv_layer(h))
            hs.append(h)
            h = F.concat(hs, axis=1)

        feat_map = self.transition_layer(h)
        feat_map = F.max_pooling_2d(feat_map, 2, 2)

        return feat_map

    def _to_gpu(self, device):
        self.to_gpu(device)
        for i in range(self.conv_num):
            conv_layer = getattr(self, 'conv_layer_{0}'.format(i))
            conv_layer.to_gpu(device)
        self.transition_layer.to_gpu(device)
        

class DenseNet(chainer.Chain):
    # (N,48,48,448) -> (N,24,24,147=0.5*(448*0.3+20*8))
    # -> (N,12,12,153=0.5*(147+20*8)) -> (N,6,6,156)
    def __init__(self, config):
        super(DenseNet, self).__init__()
        self.block_num = config.dense_n_block
        with self.init_scope():
            out_dim = int(config.enc_dim * config.dense_first_scale_down_rate)
            self.conv = L.Convolution2D(None, out_dim, ksize=1)
            for i in range(self.block_num):
                setattr(self, 'dense_block_{0}'.format(i), DenseNetBlock(config, out_dim))
            
            if config.gpu[0] >= 0:
                for i in range(self.block_num):
                    dblock = getattr(self, 'dense_block_{0}'.format(i))
                    dblock._to_gpu(config.gpu[0])

    def __call__(self, h):
        h = h.transpose(0, 3, 1, 2)
        h = self.conv(h)
        for i in range(self.block_num):
            dblock = getattr(self, 'dense_block_{0}'.format(i))
            h = dblock(h)
        h = h.reshape((-1, h.shape[1] * h.shape[2] * h.shape[3])) # (N,6*6*156)

        return h


class DIIN(chainer.Chain):

    def __init__(self, config):
        super(DIIN, self).__init__()
        with self.init_scope():
            self.word_embed = L.EmbedID(config.word_vocab_size, config.word_emb_dim,
                                        initialW=config.embeddings, ignore_label=-1)
            self.char_embed = L.EmbedID(config.char_vocab_size,
                                        config.char_emb_dim, ignore_label=-1)
            self.char_conv = CharacterConvolution(config)

            self.self_att_layers_p = SelfAttentionNetwork(config)
            self.self_att_layers_h = SelfAttentionNetwork(config)

            self.highway_network = HighwayNetwork(config)

            self.interaction_layer = interaction_layer
            self.dense_net = DenseNet(config)

            self.output_layer = L.Linear(None, config.pred_size)

        self.init_rate = config.keep_rate
        self.drop_rate = 1 - self.init_rate


    def concat_emb(self, x_emb, x_pos, x_char_emb, x_match):
        conv_char = F.dropout(self.char_conv(x_char_emb), self.drop_rate)
        ret_emb = F.concat((x_emb, conv_char, x_pos, x_match), axis=2)
        return ret_emb

    def __call__(self, x_prem, x_hypo, x_prem_pos, x_hypo_pos,
                 x_prem_char, x_hypo_char, x_prem_match, x_hypo_match, y):
        pred  = self._forward(x_prem, x_hypo, x_prem_pos, x_hypo_pos,
                              x_prem_char, x_hypo_char, x_prem_match, x_hypo_match)

        loss = F.softmax_cross_entropy(pred, y)
        acc = F.accuracy(pred, y)

        return loss, acc

    def _forward(self, x_prem, x_hypo, x_prem_pos, x_hypo_pos,
                 x_prem_char, x_hypo_char, x_prem_match, x_hypo_match):

        x_prem = self.word_embed(x_prem)
        x_hypo = self.word_embed(x_hypo)
        x_prem_char = self.char_embed(x_prem_char)
        x_hypo_char = self.char_embed(x_hypo_char)

        self.prem_mask = make_mask(x_prem.data)
        self.hypo_mask = make_mask(x_hypo.data)

        # embedding
        h_prem = self.concat_emb(x_prem, x_prem_pos, x_prem_char, x_prem_match)
        h_hypo = self.concat_emb(x_hypo, x_hypo_pos, x_hypo_char, x_hypo_match)

        # encoding
        h_prem = self.highway_network(h_prem)
        h_hypo = self.highway_network(h_hypo)

        h_prem = self.self_att_layers_p(h_prem, self.prem_mask)
        h_hypo = self.self_att_layers_h(h_hypo, self.hypo_mask)
        
        # interaction
        h_interaction = F.dropout(self.interaction_layer(h_prem, h_hypo), self.drop_rate)

        # feature extraction
        h_out = self.dense_net(h_interaction)

        # output
        pred = self.output_layer(F.dropout(h_out, self.drop_rate))

        return pred

    def predict(self, x):
        pass
