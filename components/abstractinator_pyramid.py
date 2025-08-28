from dataclasses import dataclass

from typing import Optional, List, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import AbstractinatorConfig
from .abstractinator import Abstractinator
from .segment_compressor import GaussianEntropyModel


@dataclass
class PyramidConfig:
    levels: List[AbstractinatorConfig]
    w_vq: float = 1.0
    w_byte_lm: float = 0.0
    use_top_code_lm: bool = False


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
            lvl = Abstractinator(c, lo_vq=prev_vq)  # bottom: None; upper: child VQ
            levels.append(lvl)
            # prev_vq = lvl.compressor.vq  # becomes child VQ for next level

        self.levels = nn.ModuleList(levels)
        self.top_lm = top_lm

    def compress_all(
            self,
            tokens: torch.Tensor,
            kpm: Optional[torch.Tensor],
            *,
            comp_only: bool = False,
    ) -> List[Any]:
        """Return list of CompressorOutput from bottom(0)→top(L-1)."""
        outs = []
        cur_ids, cur_kpm = tokens, kpm
        for L, level in enumerate(self.levels):
            if comp_only:
                co = level.compress(cur_ids, key_padding_mask=kpm)
                outs.append({"comp_out": co})                             # keep shape compat
            else:
                # co = level.compress(cur_ids, key_padding_mask=cur_kpm)  # CompressorOutput
                co = level(cur_ids, key_padding_mask=cur_kpm)  # CompressorOutput
                outs.append(co)
            # next level inputs
            cur_ids = outs[-1]["comp_out"].vq_indices
            if outs[-1]["comp_out"].valid_mask is not None:
                q = level.compressor.num_queries_per_segment
                cur_kpm = ~outs[-1]["comp_out"].valid_mask.repeat_interleave(q, dim=1)
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
        # vq_total = sum(co['loss_vq'] for co in comp_outs)

        total_loss = sum(co['loss_total'] for co in comp_outs)

        # Optional Top-LM training (predict next top embedding)
        top_lm_avg_loss = None
        top_lm_details: dict[str, torch.Tensor] | None = None
        if self.top_lm is not None and getattr(self.cfg, "use_top_code_lm", False):
            top_co = comp_outs[-1]['comp_out']
            top_mem = top_co.pre_vq_embeddings  # (B, S_hi, D)
            top_kpm = (~top_co.valid_mask) if top_co.valid_mask is not None else None  # (B, S_hi)

            # teacher-forced next-step prediction
            if top_mem.size(1) >= 2:  # need at least 2 steps for next-step supervision
                inp = top_mem[:, :-1, :]
                tgt = top_mem[:, 1:, :]
                kpm_in = top_kpm[:, :-1] if top_kpm is not None else None

                lm_out = self.top_lm(inp, key_padding_mask=kpm_in)
                pred = lm_out.get("predictions_pre_vq", lm_out.get("predictions"))  # (B,S-1,D)
                # Masked MSE over non-padded positions
                if kpm_in is not None:
                    valid = (~kpm_in).float()
                    mse = ((pred - tgt) ** 2).mean(dim=-1)
                    mse = (mse * valid).sum() / valid.sum().clamp(min=1)
                else:
                    mse = ((pred - tgt) ** 2).mean()

                top_vq_loss = lm_out.get("vq_loss", torch.zeros((), device=pred.device))

                top_lm_avg_loss = mse
                top_lm_details = {
                    "top_code_mse": mse,
                    "top_code_vq_loss": top_vq_loss,
                }

        # Collect useful diagnostics for logging
        # all_codebook_perplexities = [co['comp_out'].vq_perplexity for co in comp_outs]

        out = {
            "comp_outs": comp_outs,
            "loss_total": total_loss,
            # "loss_digit_ce": total_digit,
            # "loss_special_ce": total_special,
            # "loss_vq": vq_total,
            # "loss_byte_lm_ce": total_byte_lm,
            # "all_codebook_perplexities": all_codebook_perplexities,
        }

        if top_lm_avg_loss is not None and top_lm_details is not None:
            out["avg_top_code_lm_loss"] = top_lm_avg_loss
            out["top_code_lm_loss_details"] = top_lm_details

        return out

    @torch.no_grad()
    def generate_bytes(
            self,
            prompt: torch.Tensor,  # (1, S0) bytes
            prompt_kpm: torch.Tensor | None = None,
            *,
            max_top_steps: int = 256,
            max_child_len: int = 128,
            lo_window: int = 128,  # low-side AR context window (like training)
            top_sample_fn=None,  # optional CodeSequenceTransformer
            honor_eos: bool = True,
    ) -> torch.Tensor:
        """
        Each full iteration within the generate_bytes loop should execute a precise, three-phase procedure to generate
        the next segment of "bytes" (tokens).

        Phase 1: Upward Compression (compress_all): The process begins with the current sequence of bytes, which
        includes the initial prompt plus all bytes generated in previous steps. This sequence is fed into the bottom
        level (Level 0) of the pyramid. The Abstractinator at this level processes the byte stream and compresses it
        into a shorter, more abstract sequence of quantized vectors, or "codes." This sequence of codes is then passed
        as input to the next level (Level 1), which performs its own compression. This continues up the stack until the
        top-most level produces the most abstract, compact representation of the entire input sequence.

        Phase 2: Top-Level Prediction (top_lm): At the apex of the pyramid, the top_lm (a CodeSequenceTransformer) takes
        the final, high-level sequence of codes as its context. Its task is to function as a standard autoregressive
        language model, predicting the next abstract code vector in the sequence. This prediction is the generative seed
        for the entire step; it represents the model's decision about the next high-level concept to introduce into the
        sequence.

        Phase 3: Downward Expansion (generate_codes): This phase is the inverse of compression. It starts with the newly
        predicted top-level code. This code is passed to the top-level Abstractinator's expander module, which treats it
        as conditioning context and autoregressively generates a sequence of codes for the level below it. This newly
        generated sequence of codes then becomes the conditioning context for the next level down. This recursive
        expansion continues down the pyramid, with each level generating a longer, less abstract sequence based on the
        context provided by the level above. The process terminates when the bottom-level Abstractinator (Level 0)
        generates a new sequence of raw bytes.

        Returns only the newly generated bytes (continuation).
        """
        self.eval()
        self._set_all_mla_fusions(enabled=True, include_expanders=False)

        def logits_from_embedding_distance(z, E, tau=0.5):
            # z: (B,D), E: (V,D)
            # logits_i ∝ -||z - E_i||^2 / tau
            # = (2 z·E_i - ||E_i||^2)/tau  + const  (drop ||z||^2)
            Ez = z @ E.T  # (B,V)
            E2 = (E * E).sum(dim=1)  # (V,)
            logits = (2.0 * Ez - E2) / tau
            return logits



        assert prompt.size(0) == 1
        device = prompt.device
        if prompt_kpm is None:
            prompt_kpm = torch.zeros_like(prompt, dtype=torch.bool, device=device)

        all_new_bytes = []

        def _valid_len(kpm: torch.Tensor | None, S: int) -> int:
            if kpm is None:
                return S
            real = (~kpm[0]).nonzero(as_tuple=False)
            return int(real[-1].item()) + 1 if real.numel() else 0


        E = self.levels[-1].embedding  # (V, D)
        prompt_ent_gen = prompt.clone()
        kpm_ent_gen = prompt_kpm.clone()
        emb = self.levels[-1].embedding(prompt_ent_gen)  # (1, S, D)
        cache, _ = self.levels[-1].compressor.stream_prefill(emb, kpm_ent_gen)
        B = prompt.shape[0]
        compressor = self.levels[-1].compressor
        embed = self.levels[-1].embedding
        generated = []

        for x in range(128):
            if isinstance(compressor.entropy_model, GaussianEntropyModel):
                # next embedding from N(mu, sigma^2) at last position
                mu = cache.entropy.last_mu  # (B, D)
                logvar = cache.entropy.last_logvar  # (B, D)
                std = torch.exp(0.5 * logvar)
                next_emb = mu + std * torch.randn_like(std)  # temperature=1.0
                x_new = next_emb.unsqueeze(1)  # (B, 1, D)
            else:
                # logits branch: sample next id, then embed to get the new x
                logits = cache.entropy.last_logits  # (B, V)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
                x_new = embed(next_id).unsqueeze(1)  # (B, 1, D)

            kpm_new = torch.zeros(B, 1, dtype=torch.bool, device=device)  # not padded
            cache, _ = compressor.stream_step(cache, x_new, kpm_new)  # pos auto-advances
            generated.append(x_new)


        from .tokenizer import ByteLevelTokenizer
        tokenizer = ByteLevelTokenizer()
        print(f"{tokenizer.decode(prompt_ent_gen.squeeze())}{tokenizer.decode(torch.argmax(torch.cat(generated, dim=1) @ E.weight.T, dim=-1).squeeze())}")
        for _ in range(max_top_steps):
            # ── Phase 1: Upward compression on current bytes ─────────────────────
            comp_all = self.compress_all(prompt, prompt_kpm, comp_only=True)
            top_comp = comp_all[-1]['comp_out']
            top_mem = top_comp.pre_vq_embeddings  # (1, S_hi, D)
            top_kpm = (~top_comp.valid_mask) if top_comp.valid_mask is not None else None

            # ── Phase 2: Top-level "append a new row" (optional) ────────────────
            # If you have a top LM, predict one new row into the first padded slot.
            if (self.top_lm is not None) and (top_sample_fn is not None) and (top_kpm is not None):
                valid = int((~top_kpm).sum().item())
                assert valid < top_mem.size(1), "No free top rows to write a new segment."
                lm_out = top_sample_fn(top_mem[:, :valid, :].contiguous(),
                                       key_padding_mask=top_kpm[:, :valid].contiguous())
                next_top = lm_out.get("predictions_pre_vq", lm_out.get("predictions"))[:, -1, :]
                # materialize the new row
                top_mem = top_mem.clone()
                top_kpm = top_kpm.clone()
                top_mem[:, valid, :] = next_top
                top_kpm[:, valid] = False
                hi_memory = top_mem[:, :valid + 1, :]
                src_kpm = top_kpm[:, :valid + 1]
                target_row = valid
            else:
                # Continue under last existing row
                if top_kpm is not None:
                    valid = int((~top_kpm).sum().item())
                    target_row = valid - 1
                    hi_memory = top_mem[:, :valid, :]
                    src_kpm = top_kpm[:, :valid]
                else:
                    target_row = top_mem.size(1) - 1
                    hi_memory = top_mem
                    src_kpm = None

            # ── Phase 3: Downward expansion with entropy-based stop ─────────────
            cur_hi = hi_memory
            cur_src = src_kpm
            bottom_new = None

            for L in reversed(range(len(self.levels))):
                lvl = self.levels[L]
                compL = comp_all[L]['comp_out']

                # Build low-side seed = tokens from START OF LAST SEGMENT to end (truncate to lo_window-1)
                # This preserves the compressor's "since last break" state.
                seqL = compL.input_sequence  # (1, S_L)
                kpmL = compL.input_padding_mask  # (1, S_L) or None
                segL = compL.seg_id  # (1, S_L)
                real_lenL = _valid_len(kpmL, seqL.size(1))
                if real_lenL == 0:
                    # Fallback: 1-token EOS seed mapped to target_row
                    seed_lo = torch.full((1, 1), int(lvl.expander.eos_id), dtype=seqL.dtype, device=device)
                    seed_seg = torch.full((1, 1), int(cur_hi.size(1) - 1), dtype=segL.dtype, device=device)
                else:
                    last_idx = real_lenL - 1
                    # start index of last segment in *current prompt* at this level
                    start_last = int(compL.first_byte_idx[0, last_idx].item())
                    # trailing window but not before start_last
                    seed_start = max(start_last, real_lenL - (lo_window - 1))
                    seed_lo = seqL[:, seed_start:real_lenL]
                    seed_seg = segL[:, seed_start:real_lenL].clone()
                    # force the last seed token to map to the row we expand under
                    seed_seg[:, -1] = int(cur_hi.size(1) - 1)

                # Multi-query: map to query row 0, like training
                Lq = lvl.compressor.num_queries_per_segment
                if Lq > 1:
                    seed_seg = seed_seg * Lq

                # Incrementally generate until *this level’s* compressor LM says "boundary now".
                run_lo = seed_lo
                run_seg = seed_seg

                # --- incremental generation at level L, entropy-stop outside this block ---
                for _t in range(max_child_len):
                    # 0) Make sure we leave headroom if the decoder has a hard max_len
                    max_len = getattr(lvl.expander, "max_len", None)
                    if max_len is not None and run_lo.size(1) >= max_len:
                        # keep last (max_len-1) tokens so the model can append one
                        keep = max_len - 1
                        run_lo = run_lo[:, -keep:]
                        run_seg = run_seg[:, -keep:]

                    # 1) Extend seg_ids for the *future* step(s) we request
                    steps = 1
                    seg_ext = torch.cat([run_seg, run_seg[:, -1:].expand(1, steps)], dim=1)
                    assert seg_ext.size(1) == run_lo.size(1) + steps

                    # 2) Generate
                    gen = lvl.generate_codes(
                        hi_memory=cur_hi,
                        seed_lo=run_lo,
                        src_key_padding_mask=cur_src if L == len(self.levels) - 1 else None,
                        seg_ids=seg_ext,
                        max_new_tokens=steps,
                    )

                    # 3) Normalize return convention
                    if gen.size(1) == run_lo.size(1) + steps:
                        # expander returns prefix+new
                        new_tok = gen[:, -steps:]  # (1, 1)
                        run_lo = gen  # carry full sequence forward
                        run_seg = seg_ext  # keep aligned seg ids
                    elif gen.size(1) == steps:
                        # expander returns only the new tokens
                        new_tok = gen  # (1, 1)
                        run_lo = torch.cat([run_lo, new_tok], dim=1)
                        run_seg = torch.cat([run_seg, run_seg[:, -1:].expand(1, steps)], dim=1)
                    else:
                        # Unexpected: treat as "no token" and bail with a hint
                        # (you can raise instead if you prefer)
                        print("Warn: unexpected generate() shape", gen.shape)
                        break

                    # 4) Optional EOS policy
                    if honor_eos and (new_tok == lvl.expander.eos_id).all():
                        break

                    # 5) Your entropy-based boundary check (unchanged):
                    _, patch_end = lvl.segment_only(run_lo, key_padding_mask=None)
                    if bool(patch_end[0, run_lo.size(1) - 1].item()):
                        break

                # "New tokens" for this level (exclude the seed)
                new_lo = run_lo[:, seed_lo.size(1):]

                if L == 0:
                    bottom_new = new_lo
                else:
                    # For the next lower level, the entire sequence (seed+new) acts as discrete hi memory.
                    cur_hi = run_lo
                    cur_src = None  # discrete memory has no src pad mask

            # If nothing new at bottom, stop the outer loop
            if bottom_new is None or bottom_new.size(1) == 0:
                print(f'Warning: no new bottom bytes generated, stopping.')
                break

            # Append to bytes and continue
            all_new_bytes.append(bottom_new)
            prompt = torch.cat([prompt, bottom_new], dim=1)
            prompt_kpm = torch.cat([prompt_kpm, torch.zeros_like(bottom_new, dtype=torch.bool)], dim=1)

            # Optional overall EOS/EOP policy at byte level (only EOS remains now)
            if honor_eos and (bottom_new[:, -1:] == self.levels[0].expander.eos_id).all():
                break

        self._set_all_mla_fusions(enabled=False, include_expanders=False)


        return torch.cat(all_new_bytes, dim=1) if all_new_bytes else prompt.new_zeros((1, 0), dtype=prompt.dtype)

    @torch.no_grad()
    def _set_all_mla_fusions(self, enabled: bool, include_expanders: bool = False) -> None:
        """
        Toggle MLA fused query/output across all compressor blocks (and optionally decoders).
        Safe to call repeatedly; fusions rebuild lazily with correct dtype/device.
        """
        for lvl in self.levels:
            # Compressors: shared / lm / compression stacks
            for blk in list(lvl.compressor.shared_layers) + list(lvl.compressor.entropy_model.layers) + list(lvl.compressor.compression_layers):
                if hasattr(blk, "attn") and hasattr(blk.attn, "mla"):
                    mla = blk.attn.mla
                    if enabled and hasattr(mla, "enable_inference_fusions"):
                        mla.enable_inference_fusions()
                    elif not enabled and hasattr(mla, "disable_fusions"):
                        mla.disable_fusions()

            if include_expanders:
                # Decoder self-attn uses CausalMLA(...).mla internally
                if hasattr(lvl.expander, "decoder"):
                    for dblk in getattr(lvl.expander.decoder, "layers", []):
                        if hasattr(dblk, "self_attn") and hasattr(dblk.self_attn, "mla"):
                            mla = dblk.self_attn.mla
                            if enabled and hasattr(mla, "enable_inference_fusions"):
                                mla.enable_inference_fusions()
                            elif not enabled and hasattr(mla, "disable_fusions"):
                                mla.disable_fusions()



def factorized_code_ce(stage_logits, special_logits, targets, codec, tgt_kpm):
    # stage_logits: list of (B,L,K)
    lg = torch.stack(stage_logits, dim=0)  # (D,B,L,K)
    digits, is_special = codec.decompose(targets)  # list[(B,L)], (B,L)
    tg = torch.stack(digits, dim=0)  # (D,B,L)

    valid = ~is_special if tgt_kpm is None else (~is_special & ~tgt_kpm)
    D, B, L, K = lg.shape
    lg2 = lg.permute(0, 1, 2, 3).reshape(D * B * L, K)
    tg2 = tg.reshape(D * B * L)
    mask2 = valid.unsqueeze(0).expand(D, -1, -1).reshape(D * B * L)

    if mask2.any():
        ce_all = F.cross_entropy(lg2[mask2], tg2[mask2], reduction="mean")
    else:
        ce_all = lg2.sum() * 0  # zero on empty

    # specials unchanged
    special_ce = torch.zeros_like(ce_all)
    if special_logits is not None:
        sp_target = codec.special_local_index(targets)
        sp_mask = valid & codec.is_special(targets)
        if sp_mask.any():
            ce_sp = F.cross_entropy(
                special_logits.transpose(1, 2)[sp_mask], sp_target[sp_mask], reduction="mean")
            special_ce = ce_sp
    return {"digit_ce": ce_all, "special_ce": special_ce}
