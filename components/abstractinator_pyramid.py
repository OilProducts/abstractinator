from dataclasses import dataclass

from typing import Optional, List, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import AbstractinatorConfig
from .abstractinator import Abstractinator, insert_eop_fixed_len
from .vector_quantizer import ComposedIndexCodec

@dataclass
class PyramidConfig:
    levels: List[AbstractinatorConfig]
    w_vq: float = 0.1
    w_byte_lm: float = 0.0
    use_top_code_lm: bool = False
    # If you keep a top LM, expose a tiny interface:
    # predict_one_next(top_context: (B,S,D), kpm: (B,S)?) -> (B,D)

class AbstractinatorPyramid(nn.Module):
    def __init__(self,
                 cfg: PyramidConfig,
                 top_lm: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg

        levels = []
        prev_vq = None

        for i, c in enumerate(cfg.levels):
            lvl = Abstractinator(c, lo_vq=prev_vq)   # bottom: None; upper: child VQ
            levels.append(lvl)
            prev_vq = lvl.compressor.vq              # becomes child VQ for next level

        self.levels = nn.ModuleList(levels)
        self.top_lm = top_lm

    # @torch.no_grad()
    def compress_all(
        self, tokens: torch.Tensor, kpm: Optional[torch.Tensor]
    ) -> List[Any]:
        """Return list of CompressorOutput from bottom(0)→top(L-1)."""
        outs = []
        cur_ids, cur_kpm = tokens, kpm
        for L, level in enumerate(self.levels):
            co = level.compress(cur_ids, key_padding_mask=cur_kpm)  # CompressorOutput
            outs.append(co)
            # next level inputs
            cur_ids = co.vq_indices
            if co.valid_mask is not None:
                # num_queries per segment (assume 1 unless configured otherwise)
                q = level.compressor.num_queries_per_segment
                cur_kpm = ~co.valid_mask.repeat_interleave(q, dim=1)
            else:
                cur_kpm = None
        return outs

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full training step:
          - compress all levels
          - build EOP-aware teacher-forcing targets
          - decode at each level (teacher forced)
          - losses: factorized code CE (+ specials), optional byte-LM CE, VQ loss
          - optional top code LM
        """
        B = tokens.size(0)
        device = tokens.device
        comp_outs = self.compress_all(tokens, key_padding_mask)

        # aggregate vq loss
        vq_total = sum(co.vq_loss for co in comp_outs)

        # Build teacher-forcing targets from bottom→top inputs
        targets: List[torch.Tensor] = []
        tgt_kpms: List[Optional[torch.Tensor]] = []
        for L, co in enumerate(comp_outs):        # bottom input for level L is tokens at that level
            tgt = co.input_sequence               # (B, S_in at level L)
            kpm = co.input_padding_mask
            # EOP insertion for decoder at level L (so it learns to stop per segment)
            if co.patch_end_mask is not None and kpm is not None:
                eop_id = self.levels[L-1].compressor.vq.eop_token_id if L > 0 else self.levels[0].compressor.vq.eop_token_id
                pad_id = self.levels[L-1].compressor.vq.padding_token_id if L > 0 else self.levels[0].compressor.vq.padding_token_id
                tgt, kpm = insert_eop_fixed_len(tgt, co.patch_end_mask, kpm, eop_id=eop_id, pad_id=pad_id)
            targets.append(tgt)
            tgt_kpms.append(kpm)

        # Decode each level with teacher forcing
        total_digit = torch.zeros((), device=device)
        total_special = torch.zeros((), device=device)
        total_byte_lm = torch.zeros((), device=device)


        for L, level in enumerate(self.levels):
            # memory is *continuous* embeddings output at level L (hi)
            memory = comp_outs[L].vq_embeddings        # (B, S_hi, D)
            src_kpm = ~comp_outs[L].valid_mask if comp_outs[L].valid_mask is not None else None

            # low side target is what the *next lower* compressor took as input,
            # or bytes for bottom
            codes_lo = targets[L]                      # (B, L_len) composed ids in that space
            tgt_kpm = tgt_kpms[L]
            seg_ids = comp_outs[L].seg_id[:, :codes_lo.size(1)]

            logits = level.decode_logits(
                memory=memory, codes_lo=codes_lo,
                src_key_padding_mask=src_kpm,
                tgt_key_padding_mask=tgt_kpm,
                seg_ids=seg_ids,
            )

            ce = factorized_code_ce(
                stage_logits=logits["stage_logits"],
                special_logits=logits.get("special_logits"),
                targets=codes_lo,
                codec=level.lo_codec,
                tgt_kpm=tgt_kpm,
            )
            total_digit += ce["digit_ce"]
            total_special += ce["special_ce"]

            # Optional byte-LM CE—uses compressor LM branch at this level
            if self.cfg.w_byte_lm > 0 and comp_outs[L].entropy_model_logits.size(1) >= 2:
                lm_logits = comp_outs[L].entropy_model_logits[:, :-1, :]
                lm_target = comp_outs[L].input_sequence[:, 1:]
                lm_kpm = comp_outs[L].input_padding_mask[:, 1:] if comp_outs[L].input_padding_mask is not None else None
                per_tok = F.cross_entropy(lm_logits.transpose(1, 2), lm_target, reduction="none")
                if lm_kpm is not None:
                    mask = ~lm_kpm
                    total_byte_lm += (per_tok * mask).sum() / mask.sum().clamp(min=1)
                else:
                    total_byte_lm += per_tok.mean()

        total_loss = (
            total_digit + total_special
            + self.cfg.w_vq * vq_total
            + self.cfg.w_byte_lm * total_byte_lm
        )

        return {
            "loss_total": total_loss,
            "loss_digit_ce": total_digit,
            "loss_special_ce": total_special,
            "loss_vq": vq_total,
            "loss_byte_lm_ce": total_byte_lm,
        }

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt: torch.Tensor,               # (1, S0) bytes
        prompt_kpm: Optional[torch.Tensor] = None,
        max_top_steps: int = 256,
        max_child_len: int = 64,
        top_sample_fn: Optional[Any] = None,  # if you keep a top LM
    ) -> torch.Tensor:
        """
        Segment-wise generation from the top level down.
        If no top LM, we just use the last available top memory row as 'parent' to expand bottom.
        """
        self.eval()
        assert prompt.size(0) == 1
        buf = prompt.clone()
        kpm = prompt_kpm

        for _ in range(max_top_steps):
            # 1) compress bytes to top
            comp_all = self.compress_all(buf, kpm)
            top_co = comp_all[-1]
            top_mem = top_co.vq_embeddings
            top_kpm = ~top_co.valid_mask if top_co.valid_mask is not None else None

            if self.top_lm and top_sample_fn:
                # predict one new top vector (continuous)
                next_top = top_sample_fn(self.top_lm, top_mem, top_kpm)  # (1,D)
                hi_seq = torch.cat([top_mem, next_top.unsqueeze(1)], dim=1)
            else:
                # no top LM: just expand under the newest top segment
                hi_seq = top_mem

            # 2) expand down the stack (L-1..0), generating codes with EOP/EOS stop
            # codes_seed = torch.full((1,1), self.levels[0].eos_id, dtype=torch.long, device=buf.device)
            seg_ids = torch.full((1,1), hi_seq.size(1)-1, dtype=torch.long, device=buf.device)
            cur_hi = hi_seq
            for L in reversed(range(len(self.levels))):
                # seed is the EOS of the low space for the *current* level
                codes_seed = torch.full(
                    (1, 1), int(self.levels[L].expander.eos_id),
                    dtype=torch.long, device=buf.device
                )
                codes = self.levels[L].generate_codes(
                    hi_memory=cur_hi, seed_lo=codes_seed,
                    src_key_padding_mask=None, seg_ids=seg_ids,
                    max_new_tokens=max_child_len,
                )
                # For L>0, the generated 'codes' become hi_memory for the level below;
                # map to embeddings if you want continuous memory at the next step.
                if L > 0:
                    # discrete hi for next level
                    cur_hi = codes
                    codes_seed = torch.full_like(codes_seed, self.levels[L-1].eos_id)
                    seg_ids = torch.full_like(seg_ids, cur_hi.size(1)-1)
                else:
                    # bottom: bytes
                    new_bytes = codes[:, 1:]  # drop seed
                    buf = torch.cat([buf, new_bytes], dim=1)

            # 3) stop on EOS/EOP at bottom
            last = int(buf[0, -1].item())
            if last in (self.levels[0].eos_id, self.levels[0].eop_id):
                break

            kpm = torch.zeros_like(buf, dtype=torch.bool)

        return buf


def factorized_code_ce(stage_logits, special_logits, targets, codec, tgt_kpm):
    # stage_logits: list of (B,L,K)
    lg = torch.stack(stage_logits, dim=0)            # (D,B,L,K)
    digits, is_special = codec.decompose(targets)    # list[(B,L)], (B,L)
    tg = torch.stack(digits, dim=0)                  # (D,B,L)

    valid = ~is_special if tgt_kpm is None else (~is_special & ~tgt_kpm)
    D,B,L,K = lg.shape
    lg2 = lg.permute(0,1,2,3).reshape(D*B*L, K)
    tg2 = tg.reshape(D*B*L)
    mask2 = valid.unsqueeze(0).expand(D,-1,-1).reshape(D*B*L)

    if mask2.any():
        ce_all = F.cross_entropy(lg2[mask2], tg2[mask2], reduction="mean")
    else:
        ce_all = lg2.sum()*0  # zero on empty

    # specials unchanged
    special_ce = torch.zeros_like(ce_all)
    if special_logits is not None:
        sp_target = codec.special_local_index(targets)
        sp_mask = valid & codec.is_special(targets)
        if sp_mask.any():
            ce_sp = F.cross_entropy(
                special_logits.transpose(1,2)[sp_mask], sp_target[sp_mask], reduction="mean")
            special_ce = ce_sp
    return {"digit_ce": ce_all, "special_ce": special_ce}
