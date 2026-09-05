import math
import torch
from torch import nn
from torch.nn import functional as F, Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from stft import TorchSTFT
import commons
from commons import init_weights, get_padding
import modules as modules


class Multistream_iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=0):
        super(Multistream_iSTFT_Generator, self).__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)), k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []
 
        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands*(self.post_n_fft + 2), 7, 1, padding=3))
        self.subband_conv_post.apply(init_weights)
        
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()

        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0

        self.register_buffer("updown_filter", updown_filter)
        self.multistream_conv_post = weight_norm(Conv1d(4, 1, kernel_size=63, bias=False, padding=get_padding(63, 1)))
        self.multistream_conv_post.apply(init_weights)
        
    def forward(self, x, g=None):
      stft = TorchSTFT(filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size, win_length=self.gen_istft_n_fft).to(x.device)
      x = self.conv_pre(x) # [B, ch, length]
        
      for i in range(self.num_upsamples):
          x = F.leaky_relu(x, modules.LRELU_SLOPE)
          x = self.ups[i](x)
          xs = None

          for j in range(self.num_kernels):
              if xs is None:
                  xs = self.resblocks[i*self.num_kernels+j](x)
              else:
                  xs += self.resblocks[i*self.num_kernels+j](x)

          x = xs / self.num_kernels
          
      x = F.leaky_relu(x)
      x = self.reflection_pad(x)
      x = self.subband_conv_post(x)
      x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1]//self.subbands, x.shape[-1]))

      spec = torch.exp(x[:,:,:self.post_n_fft // 2 + 1, :])
      phase = math.pi*torch.sin(x[:,:, self.post_n_fft // 2 + 1:, :])

      y_mb_hat = stft.inverse(torch.reshape(spec, (spec.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1])), torch.reshape(phase, (phase.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1])))
      y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1]))
      y_mb_hat = y_mb_hat.squeeze(-2)

      y_mb_hat = F.conv_transpose1d(y_mb_hat, self.updown_filter.to(x.device) * self.subbands, stride=self.subbands)
      y_g_hat = self.multistream_conv_post(y_mb_hat)
      return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
      print('Removing weight norm...')
      for l in self.ups:
          remove_weight_norm(l)
      for l in self.resblocks:
          l.remove_weight_norm()


class EFTS2(nn.Module):
  """
  Synthesizer for Training
  """
  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    spec_encoder_layers,
    prior_nn1_layers,
    prior_nn2_layers,
    vap_layers,
    n_speakers=0,
    gin_channels=0,
    ms_istft_vits=True,
    **kwargs
  ):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.enc_p = modules.TextEncoder(
        n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    )
    self.dec = modules.Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

    self.enc_q = modules.SpectrogramEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, spec_encoder_layers, gin_channels=gin_channels)
    self.attn = modules.HybridAttention(inter_channels)
    self.prior_nn1 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn1_layers, gin_channels=gin_channels)
    self.prior_nn2 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn2_layers, gin_channels=gin_channels)
    self.VAP = modules.VarationalAlignmentPredictor(hidden_channels, 3, n_layers=vap_layers, gin_channels=gin_channels)
    self.attn_op = modules.AttentionOperator(hidden_channels)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  def forward(self, x, x_lengths, y, y_lengths, sid=None, bi=False):
    x_h, x_mask = self.enc_p(x, x_lengths)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    y_h, z1, z2, m1, logs1, m2, logs2, y_mask = self.enc_q(y, y_lengths, g=g)
    x_mask_b, y_mask_b = x_mask.squeeze(1).bool(),  y_mask.squeeze(1).bool()
    
    e, a, b = self.attn_op(x_h, y_h, x_mask_b, y_mask_b, sigma=0.5)
    x_align, attns, real_sigma = self.attn(e, a, b, x_h, x_mask_b, y_mask_b)

    z_slice, ids_slice = commons.rand_slice_segments(z2, y_lengths, self.segment_size)

    o = self.dec(z_slice, g=g)

    z1_r, m_q1, logs_q1, y_mask = self.prior_nn1(x_align, y_mask, g=g)
    _, m_q2, logs_q2, y_mask = self.prior_nn2(z1, y_mask, g=g)
    
    e_a = e - a
    b_a = b - a
    loss_a, loss_a_kl = self.VAP(x_h, x_mask, e_a, b_a, g=g)
    
    if bi is True:
      _, m_q2_r, logs_q2_r, y_mask= self.prior_nn2(z1_r, y_mask, g=g)
      return o, (loss_a, loss_a_kl), attns, ids_slice, x_mask, y_mask, (m1, logs1, m_q1, logs_q1), (m2, logs2, m_q2, logs_q2), (m2, logs2, m_q2_r, logs_q2_r), real_sigma
    else:
      return o, (loss_a, loss_a_kl), attns, ids_slice, x_mask, y_mask, (m1, logs1, m_q1, logs_q1), (m2, logs2, m_q2, logs_q2), None, real_sigma

  def infer(self, x, x_lengths, sid=None, t1=0.7, t2=0.7, length_scale=1.0, ta=0.7, max_len=2000):
    x_h, x_mask = self.enc_p(x=x, x_lengths=x_lengths)

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    vap_outs = self.VAP.infer(x_h, x_mask, t_a=0.7, g=g)
    b = torch.cumsum(vap_outs[:,1,:], dim=1) * length_scale
    a = torch.cat([torch.zeros(b.size(0), 1).type_as(b), b[:,:-1]], -1)
    e = a + vap_outs[:,0,:] * length_scale
    x_align, attns, _ = self.attn(e, a, b, x_h, text_mask=None, mel_mask=None, max_length=max_len)
    y_mask = torch.ones(x_align.size(0), 1, x_align.size(-1)).type_as(x_mask)
    z_1, _, _, _= self.prior_nn1(x_align, y_mask, g=g, t=t1)
    z_2, _, _, _= self.prior_nn2(z_1, y_mask, g=g, t=t2)
    o = self.dec(z_2, g=g)
    return o, attns, y_mask


class EFTS2VC(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    gen_istft_n_fft,
    gen_istft_hop_size,
    spec_encoder_layers,
    prior_nn1_layers,
    prior_nn2_layers,
    subbands,
    n_speakers=0,
    gin_channels=0,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.enc_p = modules.TextEncoder(
        n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    )
    # self.dec = modules.Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.dec = Multistream_iSTFT_Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=gin_channels)
    self.enc_q = modules.SpectrogramEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, spec_encoder_layers, gin_channels=gin_channels)
    self.attn = modules.AttentionPI(inter_channels, 4)
    self.prior_nn1 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn1_layers, gin_channels=gin_channels)
    self.prior_nn2 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn2_layers, gin_channels=gin_channels)
    self.attn_op = modules.AttentionOperator(hidden_channels)

  def forward(self, x, x_lengths, y, y_lengths, sp_embed):
    x_h, x_mask = self.enc_p(x=x, x_lengths=x_lengths)
    g = sp_embed.unsqueeze(-1)

    y_h, z1, z2, m1, logs1, m2, logs2, y_mask = self.enc_q(y, y_lengths, g=g)
    x_mask_b, y_mask_b = x_mask.squeeze(1).bool(),  y_mask.squeeze(1).bool()
    
    pi, p = self.attn_op.compute_PI(x_h, y_h, x_mask_b, y_mask_b)
    x_align, attns, real_sigma = self.attn(pi, p, x_h, x_mask_b, y_mask_b)

    z_slice, ids_slice = commons.rand_slice_segments(z2, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)

    _, m_q1, logs_q1, y_mask= self.prior_nn1(x_align, y_mask, g=None)
    _, m_q2, logs_q2, y_mask= self.prior_nn2(z1, y_mask, g=g)

    return o, attns, ids_slice, x_mask, y_mask, (m1, logs1, m_q1, logs_q1), (m2, logs2, m_q2, logs_q2), real_sigma

  def infer(self, y, y_lengths, y_emb, tgt_emb, t1=0.7, t2=0.7):
    y_emb, tgt_emb = y_emb.unsqueeze(-1), tgt_emb.unsqueeze(-1)
    _, z1, z2, _, _, _, _, y_mask = self.enc_q(y, y_lengths, g=y_emb, t1=t1)
    z2, _, _, y_mask= self.prior_nn2(z1, y_mask, g=tgt_emb, t=t2)
    o = self.dec(z2, g=tgt_emb)
    return o
