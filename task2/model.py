import torch
import torch.nn as nn

from utils import HIST, QUEUE_RATIO, SEQ


class QueueNet(nn.Module):
    def __init__(self, c_in=19, h=128, no_ode=False, pe_freqs=0, no_nb=False, phys=False):
        super().__init__()
        self.no_ode = no_ode
        self.pe_freqs = pe_freqs
        self.no_nb = no_nb
        self.phys = phys
        if phys:
            self.alpha = nn.Parameter(torch.tensor(1.4))
        self.proj = nn.Sequential(nn.Linear(c_in, 64), nn.SiLU())
        self.g1 = nn.GRUCell(64, h)
        self.g2 = nn.GRUCell(h, h)
        self.head_q = nn.Linear(h, 1)
        self.head_s = nn.Linear(h, 1)
        self.h = h

    def forward(self, b, teacher_prob=0.0):
        sph, obs, flh, tf, odd, wy, stat, st = b['sph'], b['obs'], b['flh'], b['tf'], b['odd'], b['wy'], b['stat'], b['st']
        cnd = b['cnd'] if 'cnd' in b else None
        B, _, L = sph.shape
        dev = sph.device
        h1 = torch.zeros(B * L, self.h, device=dev)
        h2 = torch.zeros(B * L, self.h, device=dev)
        pe = None
        if self.pe_freqs:
            freqs = 2.0 ** torch.arange(self.pe_freqs, device=dev, dtype=torch.float32)
            ang = torch.pi * stat[:, 3:4] * freqs[None, :]
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], -1).unsqueeze(0).expand(B, L, 2 * self.pe_freqs)
        sp_prev = None
        logits, spreds = [], []
        for i in range(SEQ):
            if i < HIST:
                sp = sph[:, i]
                ob = obs[:, i]
                fl = flh[:, i]
            else:
                if teacher_prob > 0:
                    coin = (torch.rand(B, 1, device=dev) < teacher_prob).float()
                    sp = coin * st[:, i - 1] + (1 - coin) * sp_prev
                else:
                    sp = sp_prev
                ob = torch.zeros_like(sp)
                fl = flh[:, HIST - 1]
            qn = (sp <= QUEUE_RATIO).float()
            if self.no_nb:
                xdyn = torch.stack([sp, ob, fl, qn], -1)
            else:
                nb_a = torch.cat([sp[:, :1], sp[:, :-1]], 1)
                nb_b = torch.cat([sp[:, 1:], sp[:, -1:]], 1)
                xdyn = torch.stack([sp, ob, fl, qn, nb_a, nb_b], -1)
            xt = tf[:, i].unsqueeze(1).expand(B, L, 4)
            xo = odd[:, i].unsqueeze(-1)
            xs = stat.unsqueeze(0).expand(B, L, stat.shape[-1])
            parts = [xdyn, xt, xs] if self.no_ode else [xdyn, xt, xo, wy, xs]
            if pe is not None:
                parts.append(pe)
            if cnd is not None:
                parts.append(cnd.unsqueeze(1).expand(B, L, cnd.shape[-1]))
            xx = torch.cat(parts, -1).reshape(B * L, -1)
            z1 = self.proj(xx)
            h1 = self.g1(z1, h1)
            h2 = self.g2(h1, h2)
            logits.append(self.head_q(h2).reshape(B, L))
            sp_hat = torch.clamp(self.head_s(h2), 0.0, 1.5).reshape(B, L)
            spreds.append(sp_hat)
            sp_prev = sp_hat
        out = torch.stack(logits, 1)
        prp = b.get('prp') if self.phys else None
        if prp is not None:
            pm = torch.sigmoid(out[:, HIST - 1:SEQ])
            pc = (1 - (1 - pm) * (1 - torch.sigmoid(self.alpha) * prp)).clamp(1e-6, 1 - 1e-6)
            out = torch.cat([out[:, :HIST - 1], torch.log(pc) - torch.log1p(-pc)], 1)
        return out, torch.stack(spreds, 1)
