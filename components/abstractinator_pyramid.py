"""
Adapter responsibilities

This module exposes training (forward), compression (compress_all), and
generation (generate_bytes). For convenience, future additions may include:

- from_checkpoint(path, config=None, device=None): helper to instantiate
  AbstractinatorPyramid and load weights, preferring config embedded in the
  checkpoint when available.
- generate(prompt_tokens, max_top_steps, max_child_len, top_sample_fn): thin
  wrapper around generate_bytes that accepts byte/token inputs and returns
  decoded text or tokens using the project tokenizer.
- loglikelihood(prefix_tokens, continuation_tokens): teacher-forced scoring for
  LM-style evaluation harnesses.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.config_types import PyramidConfig

from .abstractinator import Abstractinator


class AbstractinatorPyramid(nn.Module):
    def __init__(self, cfg: PyramidConfig, top_lm: Optional[nn.Module] = None, device: Optional[torch.device] = None):
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
                # Use the progressively updated padding mask for each level
                co = level.compress(cur_ids, key_padding_mask=cur_kpm)
                outs.append({"comp_out": co})  # keep shape compat
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

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Full training step:
          - compress all levels
          - build teacher-forcing targets
          - decode at each level (teacher forced)
          - losses: factorized code CE (+ specials), optional entropy-model CE, VQ loss
          - optional top code LM
        """
        tokens.size(0)
        comp_outs = self.compress_all(tokens, key_padding_mask)
        # vq_total = sum(co['loss_vq'] for co in comp_outs)

        total_loss = sum(co['loss_total'] for co in comp_outs)

        # Aggregate VQ loss across levels (0 if a level has no quantizer)
        vq_total = torch.zeros((), device=total_loss.device)
        for co in comp_outs:
            comp = co.get("comp_out") if isinstance(co, dict) else None
            if comp is not None and getattr(comp, "vq_loss", None) is not None:
                vq_total = vq_total + comp.vq_loss

        # Optional Top-LM training (predict next top embedding)
        top_lm_avg_loss = None
        top_lm_details: dict[str, torch.Tensor] | None = None
        if self.top_lm is not None and getattr(self.cfg, "use_top_code_lm", False):
            top_co = comp_outs[-1]['comp_out']
            # Inputs to the top LM: use pre‑VQ embeddings for a smooth predictor state.
            top_mem = top_co.vq_embeddings  # (B, S_hi, D)
            top_kpm = (~top_co.valid_mask) if top_co.valid_mask is not None else None  # (B, S_hi)

            # teacher-forced next-step prediction
            if top_mem.size(1) >= 2:  # need at least 2 steps for next-step supervision
                inp = top_mem[:, :-1, :]
                # Targets: use codebook centers (quantized top rows) to pull z_pre toward the correct centroid
                tgt = top_co.vq_embeddings[:, 1:, :]
                kpm_in = top_kpm[:, :-1] if top_kpm is not None else None

                lm_out = self.top_lm(inp, key_padding_mask=kpm_in)
                # Use pre‑VQ predictions for loss to avoid STE plateaus
                pred = lm_out.get("predictions_pre_vq")  # (B,S-1,D)
                # Masked MSE over non-padded positions
                if kpm_in is not None:
                    valid = (~kpm_in).float()
                    mse = ((pred - tgt) ** 2).mean(dim=-1)
                    mse = (mse * valid).sum() / valid.sum().clamp(min=1)
                else:
                    mse = ((pred - tgt) ** 2).mean()

                # Keep VQ loss for logging only; train.py does not include it in total loss.
                top_vq_loss = lm_out.get("vq_loss", torch.zeros((), device=pred.device))

                # Top-level index accuracy (if indices are available)
                top_acc = None
                pred_idx = lm_out.get("indices", None)  # (B, S-1)
                tgt_idx = top_co.vq_indices[:, 1:] if getattr(top_co, "vq_indices", None) is not None else None
                if pred_idx is not None and tgt_idx is not None:
                    if kpm_in is not None:
                        valid_mask = ~kpm_in  # (B, S-1)
                        denom = valid_mask.sum().clamp(min=1)
                        correct = (pred_idx == tgt_idx) & valid_mask
                        top_acc = correct.sum().to(torch.float32) / denom.to(torch.float32)
                    else:
                        correct = (pred_idx == tgt_idx)
                        top_acc = correct.to(torch.float32).mean()

                top_lm_avg_loss = mse
                top_lm_details = {
                    "top_code_mse": mse,
                    "top_code_vq_loss": top_vq_loss,
                }
                if top_acc is not None:
                    top_lm_details["top_code_acc"] = top_acc

                # --- Byte-level CE through predicted top rows (grad flows to top LM) ---
                pred_q = lm_out.get("predictions", None)  # (B, S-1, D) quantized via STE
                if pred_q is not None and getattr(top_co, "vq_embeddings", None) is not None:
                    # Build predicted top memory: keep row 0 from ground truth, rows 1.. from top LM
                    gt_mem_q = top_co.vq_embeddings  # (B, S, D)
                    hi_mem_pred = gt_mem_q.clone()
                    hi_mem_pred[:, 1 : 1 + pred_q.size(1), :] = pred_q

                    # Decode bytes at level 0 using predicted memory (teacher forcing)
                    comp0 = comp_outs[0]["comp_out"]
                    lvl0 = self.levels[0]

                    # src padding over memory rows
                    if comp0.valid_mask is not None:
                        if lvl0.compressor.num_queries_per_segment == 1:
                            src_kpm0 = ~comp0.valid_mask
                        else:
                            Lq = lvl0.compressor.num_queries_per_segment
                            src_kpm0 = ~comp0.valid_mask.repeat_interleave(Lq, dim=1)
                    else:
                        src_kpm0 = None

                    # targets and masks for low stream
                    tgt_kpm0 = comp0.input_padding_mask
                    seg_ids0 = comp0.seg_id
                    if lvl0.compressor.num_queries_per_segment > 1:
                        seg_ids0 = seg_ids0 * lvl0.compressor.num_queries_per_segment
                    if hi_mem_pred.dim() == 3:
                        seg_ids0 = seg_ids0.clamp(min=0, max=hi_mem_pred.size(1) - 1)

                    # Decoder inputs mirror training: compressor hidden states
                    codes_lo0 = comp0.hidden_states

                    logits0 = lvl0.decode_logits(
                        memory=hi_mem_pred,
                        codes_lo=codes_lo0,
                        src_key_padding_mask=src_kpm0,
                        tgt_key_padding_mask=tgt_kpm0,
                        seg_ids=seg_ids0,
                    )

                    lg0 = logits0["stage_logits"][0]  # (B, L, K)
                    tgt_full0 = comp0.input_sequence  # (B, L)
                    if tgt_full0.numel() == 0 or lg0.size(1) <= 1:
                        ce_top_bytes = torch.zeros((), device=lg0.device)
                    else:
                        logits_used = lg0[:, :-1, :]
                        targets_used = tgt_full0[:, 1:]
                        if tgt_kpm0 is None:
                            ce_top_bytes = torch.nn.functional.cross_entropy(
                                logits_used.transpose(1, 2), targets_used, reduction="mean"
                            )
                        else:
                            valid = ~tgt_kpm0[:, 1:]
                            denom = valid.sum().clamp(min=1)
                            ce_all = torch.nn.functional.cross_entropy(
                                logits_used.transpose(1, 2), targets_used, reduction="none"
                            )
                            ce_top_bytes = (ce_all * valid).sum() / denom

                    top_lm_details["top_code_byte_ce"] = ce_top_bytes

                # --- Optional diagnostics (logging only) ---
                with torch.no_grad():
                    vq = getattr(self.top_lm, "vq", None)
                    if vq is not None and tgt_idx is not None and pred_idx is not None:
                        B_all = pred.size(0)
                        T_all = pred.size(1)
                        # Compute last valid position per sample
                        if kpm_in is not None:
                            valid_len = (~kpm_in).sum(dim=1)  # (B,)
                        else:
                            valid_len = torch.full((B_all,), T_all, dtype=torch.long, device=pred.device)
                        last_idx = (valid_len - 1).clamp(min=0)
                        idx_b = torch.arange(B_all, device=pred.device)

                        # Branch on quantizer type: MS-RVQ vs standard VQ
                        if hasattr(vq, "down") and hasattr(vq, "stages") and hasattr(vq, "codec"):
                            # ---- MS-RVQ path ----
                            r_dc = vq.down(pred)  # (B, S-1, d_c)
                            W0 = vq.stages[0].codebook  # (K, d_c)
                            r_last = r_dc[idx_b, last_idx, :]  # (B, d_c)
                            dot = F.linear(r_last, W0)  # (B, K)
                            e2 = (W0 * W0).sum(dim=1).view(1, -1)  # (1, K)
                            logits0 = 2.0 * dot - e2  # (B, K)
                            digits_tgt, _ = vq.codec.decompose(tgt_idx)
                            t0 = digits_tgt[0][idx_b, last_idx]  # (B,)
                            for k in (5, 10):
                                topk_idx = logits0.topk(k, dim=-1).indices  # (B, k)
                                in_topk = topk_idx.eq(t0.unsqueeze(-1)).any(dim=-1).to(torch.float32)  # (B,)
                                acc_topk = in_topk.mean()
                                top_lm_details[f"top_code_acc_top{k}_stage0_last"] = acc_topk

                            digits_tgt_all, _ = vq.codec.decompose(tgt_idx)
                            d0_tgt = digits_tgt_all[0]
                            digits_pred_all, _ = vq.codec.decompose(pred_idx)
                            d0_pred = digits_pred_all[0]
                            if kpm_in is not None:
                                vm = ~kpm_in
                                denom = vm.sum().clamp(min=1)
                                agree0 = ((d0_pred == d0_tgt) & vm).sum().to(torch.float32) / denom.to(torch.float32)
                            else:
                                agree0 = (d0_pred == d0_tgt).to(torch.float32).mean()
                            top_lm_details["top_code_center_agree_stage0"] = agree0

                            # Top-k over all valid positions (stage 0 digits)
                            # Compute logits for all positions with constant-shape matmul
                            r_dc_2d = r_dc.reshape(-1, r_dc.size(-1))  # (B*T, d_c)
                            dot_all = F.linear(r_dc_2d, W0).view(r_dc.size(0), r_dc.size(1), -1)  # (B,T,K)
                            logits0_all = 2.0 * dot_all - e2.view(1, 1, -1)  # (B,T,K)
                            t0_all = digits_tgt_all[0]  # (B,T)
                            for k in (5, 10):
                                topk_idx_all = logits0_all.topk(k, dim=-1).indices  # (B,T,k)
                                in_topk_all = topk_idx_all.eq(t0_all.unsqueeze(-1)).any(dim=-1).to(torch.float32)  # (B,T)
                                if kpm_in is not None:
                                    vm2 = ~kpm_in
                                    denom2 = vm2.sum().clamp(min=1)
                                    acc_topk_all = (in_topk_all * vm2).sum() / denom2
                                else:
                                    acc_topk_all = in_topk_all.mean()
                                top_lm_details[f"top_code_acc_top{k}_stage0"] = acc_topk_all

                            def _embed_indices_D_rvq(indices: torch.Tensor) -> torch.Tensor:
                                digits, _ = vq.codec.decompose(indices)
                                y_dc = torch.zeros(indices.size(0), indices.size(1), vq.d_c, device=pred.device, dtype=pred.dtype)
                                for s in range(vq.depth):
                                    W = vq.stages[s].codebook  # (K, d_c)
                                    y_dc = y_dc + F.embedding(digits[s], W)
                                return vq.up(y_dc)  # (B, T, D)

                            z = pred
                            center_tgt_D = _embed_indices_D_rvq(tgt_idx)
                            center_pred_D = _embed_indices_D_rvq(pred_idx)
                            d2_tgt = ((z - center_tgt_D) ** 2).sum(dim=-1)
                            d2_pred = ((z - center_pred_D) ** 2).sum(dim=-1)
                            margin = d2_tgt - d2_pred
                            if kpm_in is not None:
                                vm = ~kpm_in
                                denom = vm.sum().clamp(min=1)
                                margin_avg = (margin * vm).sum() / denom
                                mse_to_center = (((z - center_tgt_D) ** 2).mean(dim=-1) * vm).sum() / denom
                            else:
                                margin_avg = margin.mean()
                                mse_to_center = ((z - center_tgt_D) ** 2).mean()
                            top_lm_details["top_code_margin_D"] = margin_avg
                            top_lm_details["top_code_mse_to_target_center_D"] = mse_to_center
                        elif hasattr(vq, "codebook"):
                            # ---- Standard VQ path ----
                            E = vq.codebook  # (K, D)
                            z_last = pred[idx_b, last_idx, :]  # (B, D)
                            r2 = (z_last * z_last).sum(dim=-1, keepdim=True)  # (B,1)
                            e2 = (E * E).sum(dim=-1).view(1, -1)  # (1,K)
                            dot = F.linear(z_last, E)  # (B,K)
                            logits = -(r2 - 2 * dot + e2)  # (B,K)
                            for k in (5, 10):
                                topk_idx = logits.topk(k, dim=-1).indices  # (B,k)
                                in_topk = topk_idx.eq(tgt_idx[idx_b, last_idx].unsqueeze(-1)).any(dim=-1).to(torch.float32)
                                top_lm_details[f"top_code_acc_top{k}_stage0_last"] = in_topk.mean()

                            # Center agreement: top-1 across valid positions
                            if kpm_in is not None:
                                vm = ~kpm_in
                                denom = vm.sum().clamp(min=1)
                                agree = ((pred_idx == tgt_idx) & vm).sum().to(torch.float32) / denom.to(torch.float32)
                            else:
                                agree = (pred_idx == tgt_idx).to(torch.float32).mean()
                            top_lm_details["top_code_center_agree_stage0"] = agree

                            # Margin/MSE2C using D-space codebook
                            z = pred  # (B,T,D)
                            center_tgt_D = F.embedding(tgt_idx, E)  # (B,T,D)
                            center_pred_D = F.embedding(pred_idx, E)  # (B,T,D)
                            d2_tgt = ((z - center_tgt_D) ** 2).sum(dim=-1)
                            d2_pred = ((z - center_pred_D) ** 2).sum(dim=-1)
                            margin = d2_tgt - d2_pred
                            if kpm_in is not None:
                                vm = ~kpm_in
                                denom = vm.sum().clamp(min=1)
                                margin_avg = (margin * vm).sum() / denom
                                mse_to_center = (((z - center_tgt_D) ** 2).mean(dim=-1) * vm).sum() / denom
                            else:
                                margin_avg = margin.mean()
                                mse_to_center = ((z - center_tgt_D) ** 2).mean()
                            top_lm_details["top_code_margin_D"] = margin_avg
                            top_lm_details["top_code_mse_to_target_center_D"] = mse_to_center

                            # Top-k over all valid positions using D-space codebook logits
                            z2d = z.reshape(-1, z.size(-1))  # (B*T,D)
                            dot_all = F.linear(z2d, E).view(z.size(0), z.size(1), -1)  # (B,T,K)
                            # Omit r2 term (constant per position) for ranking; reuse e2
                            logits_all = 2.0 * dot_all - e2.view(1, 1, -1)
                            for k in (5, 10):
                                topk_idx_all = logits_all.topk(k, dim=-1).indices  # (B,T,k)
                                in_topk_all = topk_idx_all.eq(tgt_idx.unsqueeze(-1)).any(dim=-1).to(torch.float32)  # (B,T)
                                if kpm_in is not None:
                                    vm2 = ~kpm_in
                                    denom2 = vm2.sum().clamp(min=1)
                                    acc_topk_all = (in_topk_all * vm2).sum() / denom2
                                else:
                                    acc_topk_all = in_topk_all.mean()
                                top_lm_details[f"top_code_acc_top{k}_stage0"] = acc_topk_all

        # Collect useful diagnostics for logging
        # Per-level reconstruction details (sum of digit + special per level)
        reco_details: dict[str, torch.Tensor] = {}
        for lvl, co in enumerate(comp_outs):
            if isinstance(co, dict):
                dev = total_loss.device
                ce_d = co.get("loss_digit_ce", torch.zeros((), device=dev))
                ce_s = co.get("loss_special_ce", torch.zeros((), device=dev))
                reco = ce_d + ce_s
                # Short, stable key for console + MLflow
                reco_details[f"Reco_L{lvl}"] = reco

        all_codebook_perplexities = []
        for co in comp_outs:
            comp = co.get("comp_out") if isinstance(co, dict) else None
            ppl = getattr(comp, "vq_perplexity", None) if comp is not None else None
            # Use 0.0 if missing to keep list length consistent with levels
            if ppl is None:
                zero = torch.tensor(0.0, device=total_loss.device)
                all_codebook_perplexities.append(zero)
            else:
                all_codebook_perplexities.append(ppl)

        out = {
            "comp_outs": comp_outs,
            "loss_total": total_loss,
            "loss_vq": vq_total,
            "reconstruction_loss_details": reco_details,
            # "loss_digit_ce": total_digit,
            # "loss_special_ce": total_special,
            # "loss_vq": vq_total,
            # "loss_entropy_ce": total_entropy,
            "all_codebook_perplexities": all_codebook_perplexities,
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
        # self._set_all_mla_fusions(enabled=True, include_expanders=False)

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

        prompt_ent_gen = prompt.clone()
        kpm_ent_gen = prompt_kpm.clone()
        emb = self.levels[-1].compressor.embedding(prompt_ent_gen)  # (1, S, D)
        cache, _ = self.levels[-1].compressor.stream_prefill(emb, kpm_ent_gen)
        B = prompt.shape[0]
        compressor = self.levels[-1].compressor
        embed = self.levels[-1].compressor.embedding

        # for x in range(128):
        #     # Prefer Gaussian if cache exposes stats; else fall back to logits
        #     if hasattr(cache, "entropy") and hasattr(cache.entropy, "last_mu") and cache.entropy.last_mu is not None:
        #         mu = cache.entropy.last_mu  # (B, D)
        #         logvar = cache.entropy.last_logvar  # (B, D)
        #         std = torch.exp(0.5 * logvar)
        #         next_emb = mu + std * torch.randn_like(std)
        #         x_new = next_emb.unsqueeze(1)
        #     elif hasattr(cache, "entropy") and hasattr(cache.entropy, "last_logits") and cache.entropy.last_logits is not None:
        #         logits = cache.entropy.last_logits
        #         probs = torch.softmax(logits, dim=-1)
        #         next_id = torch.multinomial(probs, 1).squeeze(-1)
        #         x_new = embed(next_id).unsqueeze(1)
        #     else:
        #         # Fallback: query entropy model directly
        #         next_emb_or_id = compressor.sample_next(emb, kpm_ent_gen, embed_fn=embed)
        #         if next_emb_or_id.dim() == 2 and next_emb_or_id.size(-1) == emb.size(-1):
        #             x_new = next_emb_or_id.unsqueeze(1)
        #         else:
        #             x_new = embed(next_emb_or_id).unsqueeze(1)
        #
        #     kpm_new = torch.zeros(B, 1, dtype=torch.bool, device=device)  # not padded
        #     cache, _ = compressor.stream_step(cache, x_new, kpm_new)  # pos auto-advances
        for _ in range(max_top_steps):
            # ── Phase 1: Upward compression on current bytes ─────────────────────
            comp_all = self.compress_all(prompt, prompt_kpm, comp_only=True)
            top_comp = comp_all[-1]['comp_out']
            # Track both memories: pre‑VQ for top‑LM context; quantized for expanders
            top_mem_pre = top_comp.pre_vq_embeddings  # (1, S_hi, D)
            top_mem_quant = (
                top_comp.vq_embeddings
                if getattr(top_comp, "vq_embeddings", None) is not None
                else top_comp.pre_vq_embeddings
            )  # (1, S_hi, D)
            top_kpm = (~top_comp.valid_mask) if top_comp.valid_mask is not None else None

            # ── Phase 2: Top-level "append a new row" (optional) ────────────────
            # Match training: feed padded inputs of fixed length; pick prediction at valid-1.
            if (self.top_lm is not None) and (top_sample_fn is not None) and (top_kpm is not None):
                valid = int((~top_kpm).sum().item())
                assert valid < top_mem_pre.size(1), "No free top rows to write a new segment."
                assert valid >= 1, "Top memory has no real rows to condition on."

                # Training uses pre‑VQ embeddings; match here
                inp = top_mem_pre[:, :-1, :].contiguous()
                kpm_in = top_kpm[:, :-1].contiguous()
                lm_out = top_sample_fn(inp, key_padding_mask=kpm_in)
                preds_pre = lm_out.get("predictions_pre_vq")  # (B, S-1, D)
                preds_quant = lm_out.get("predictions")  # (B, S-1, D) already quantized by shared VQ
                # Select next row for current last real position
                next_top_pre = preds_pre[:, valid - 1, :]
                next_top_quant = preds_quant[:, valid - 1, :]

                # materialize the new row in both memories
                top_mem_pre = top_mem_pre.clone()
                top_mem_quant = top_mem_quant.clone()
                top_kpm = top_kpm.clone()
                top_mem_pre[:, valid, :] = next_top_pre
                top_mem_quant[:, valid, :] = next_top_quant
                top_kpm[:, valid] = False
                # For expansion, use the quantized memory
                hi_memory = top_mem_quant[:, : valid + 1, :]
                src_kpm = top_kpm[:, : valid + 1]
            else:
                # Continue under last existing row
                if top_kpm is not None:
                    valid = int((~top_kpm).sum().item())
                    hi_memory = top_mem_quant[:, :valid, :]
                    src_kpm = top_kpm[:, :valid]
                else:
                    hi_memory = top_mem_quant
                    src_kpm = None

            #  Phase 3: Downward expansion with entropy-based stop
            cur_hi = hi_memory
            cur_src = src_kpm
            bottom_new = None

            for L in reversed(range(len(self.levels))):
                lvl = self.levels[L]
                compL = comp_all[L]['comp_out']

                # Use full prefix as left context (training-consistent)
                seqL = compL.input_sequence  # (1, S_L)
                kpmL = compL.input_padding_mask  # (1, S_L) or None
                segL = compL.seg_id  # (1, S_L)
                real_lenL = _valid_len(kpmL, seqL.size(1))

                # There should always be a prompt; guard minimally
                if real_lenL == 0:
                    # Degenerate guard; shouldn't happen with a real prompt
                    prefix_lo = torch.full((1, 1), int(lvl.expander.eos_id), dtype=seqL.dtype, device=device)
                    prefix_seg = torch.full((1, 1), int(cur_hi.size(1) - 1), dtype=segL.dtype, device=device)
                else:
                    prefix_lo = seqL[:, :real_lenL]
                    prefix_seg = segL[:, :real_lenL].clone()
                    # Force the last prefix token to map to the target high row
                    prefix_seg[:, -1] = int(cur_hi.size(1) - 1)

                # Multi-query: map to query row 0, like training
                Lq = lvl.compressor.num_queries_per_segment
                if Lq > 1:
                    prefix_seg = prefix_seg * Lq

                # Incrementally generate the ENTIRE next segment at this level.
                run_lo = prefix_lo
                run_seg = prefix_seg

                # Track the segment id of the new segment (as defined by the entropy model)
                current_segment_id: int | None = None
                for _t in range(max_child_len):
                    # 0) Make sure we leave headroom if the decoder has a hard max_len
                    max_len = getattr(lvl.expander, "max_len", None)
                    if max_len is not None and run_lo.size(1) >= max_len:
                        keep = max_len - 1
                        run_lo = run_lo[:, -keep:]
                        run_seg = run_seg[:, -keep:]

                    # 1) Extend seg_ids for the single new step we request
                    steps = 1
                    seg_ext = torch.cat([run_seg, run_seg[:, -1:].expand(1, steps)], dim=1)
                    assert seg_ext.size(1) == run_lo.size(1) + steps

                    # 2) Generate one new token
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
                        new_tok = gen[:, -steps:]
                        run_lo = gen
                        run_seg = seg_ext
                    elif gen.size(1) == steps:
                        # expander returns only the new tokens
                        new_tok = gen
                        run_lo = torch.cat([run_lo, new_tok], dim=1)
                        run_seg = torch.cat([run_seg, run_seg[:, -1:].expand(1, steps)], dim=1)
                    else:
                        # Unexpected: stop
                        break

                    # 4) Optional EOS policy
                    if honor_eos and (new_tok == lvl.expander.eos_id).all():
                        break

                    # 5) Entropy-based segmentation: keep generating while the
                    # last token remains in the same (new) segment. When
                    # segmentation advances, drop that last token and stop.
                    seg_idsL, _ = lvl.segment_only(run_lo, key_padding_mask=None)
                    last_seg = int(seg_idsL[0, run_lo.size(1) - 1].item())
                    if current_segment_id is None:
                        current_segment_id = last_seg
                        continue
                    if last_seg != current_segment_id:
                        # Remove the token that started the next segment
                        run_lo = run_lo[:, :-1]
                        run_seg = run_seg[:, :-1]
                        break

                # "New tokens" for this level (exclude the prefix)
                new_lo = run_lo[:, prefix_lo.size(1) :]


                if L == 0:
                    bottom_new = new_lo
                else:
                    # For the next lower level, the entire sequence (seed+new) acts as discrete hi memory.
                    cur_hi = run_lo
                    cur_src = None  # discrete memory has no src pad mask

            # If nothing new at bottom, stop the outer loop
            if bottom_new is None or bottom_new.size(1) == 0:
                print('Warning: no new bottom bytes generated, stopping.')
                break

            # Append to bytes and continue
            all_new_bytes.append(bottom_new)
            prompt = torch.cat([prompt, bottom_new], dim=1)
            prompt_kpm = torch.cat([prompt_kpm, torch.zeros_like(bottom_new, dtype=torch.bool)], dim=1)

            # Optional overall EOS policy at byte level
            if honor_eos and (bottom_new[:, -1:] == self.levels[0].expander.eos_id).all():
                break

        # self._set_all_mla_fusions(enabled=False, include_expanders=False)

        return torch.cat(all_new_bytes, dim=1) if all_new_bytes else prompt.new_zeros((1, 0), dtype=prompt.dtype)

    # @torch.no_grad()
    # def _set_all_mla_fusions(self, enabled: bool, include_expanders: bool = False) -> None:
    #     """
    #     Toggle MLA fused query/output across all compressor blocks (and optionally decoders).
    #     Safe to call repeatedly; fusions rebuild lazily with correct dtype/device.
    #     """
    #     for lvl in self.levels:
    #         # Compressors: shared / lm / compression stacks
    #         for blk in (
    #             list(lvl.compressor.shared_layers)
    #             + list(lvl.compressor.entropy_model.layers)
    #             + list(lvl.compressor.compression_layers)
    #         ):
    #             if hasattr(blk, "attn") and hasattr(blk.attn, "mla"):
    #                 mla = blk.attn.mla
    #                 if enabled and hasattr(mla, "enable_inference_fusions"):
    #                     mla.enable_inference_fusions()
    #                 elif not enabled and hasattr(mla, "disable_fusions"):
    #                     mla.disable_fusions()
    #
    #         if include_expanders:
    #             # Decoder self-attn uses CausalMLA(...).mla internally
    #             if hasattr(lvl.expander, "decoder"):
    #                 for dblk in getattr(lvl.expander.decoder, "layers", []):
    #                     if hasattr(dblk, "self_attn") and hasattr(dblk.self_attn, "mla"):
    #                         mla = dblk.self_attn.mla
    #                         if enabled and hasattr(mla, "enable_inference_fusions"):
    #                             mla.enable_inference_fusions()
    #                         elif not enabled and hasattr(mla, "disable_fusions"):
    #                             mla.disable_fusions()


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
            ce_sp = F.cross_entropy(special_logits.transpose(1, 2)[sp_mask], sp_target[sp_mask], reduction="mean")
            special_ce = ce_sp
    return {"digit_ce": ce_all, "special_ce": special_ce}
