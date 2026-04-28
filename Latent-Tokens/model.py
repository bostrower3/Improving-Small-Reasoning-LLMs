from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from config import LatentTokenConfig


class LatentTokenModel(nn.Module):
    """
    Frozen base CausalLM + trainable latent token embeddings.

    Notes:
    - Base model params are frozen.
    - Latent embeddings are the only trainable params.
    - Latent tokens are inserted in the input stream according to config.
    - Position IDs can be "frozen" so latent tokens share the following verbal token's position.
    """

    def __init__(self, config: LatentTokenConfig):
        super().__init__()
        self.config = config

        # Forward optional dtype/device options into HuggingFace model loader.
        model_kwargs: Dict[str, object] = {}
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map
        if config.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, config.torch_dtype)

        # Load pretrained decoder-only LM (e.g., Llama) from HF.
        self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        # Freeze all pretrained model weights/disable dropout etc.
        self.base_model.requires_grad_(False)
        self.base_model.eval()

        # Latent embeddings are the ONLY trainable parameters.
        hidden_size = self.base_model.config.hidden_size
        self.latent_embeddings = nn.Embedding(config.num_latent_tokens, hidden_size)
        # Match device/dtype of the backbone's token embedding table. With device_map="auto"
        # (or any HF load that places weights on GPU), a fresh nn.Embedding stays on CPU by
        # default and would break _build_inputs_embeds() when writing into verbal embeds.
        _emb_w = self.base_model.get_input_embeddings().weight
        self.latent_embeddings = self.latent_embeddings.to(device=_emb_w.device, dtype=_emb_w.dtype)

        self.ignore_index = config.ignore_index
        self.vocab_size = self.base_model.config.vocab_size

    def trainable_parameters(self):
        # Convenience helper for optimizer: optimizer(model.trainable_parameters(), ...)
        return self.latent_embeddings.parameters()

    #????TODO: may need to change this if we use batched inputs, cuz each example may have different length end padding
    def _insertion_positions(self, verbal_ids: Sequence[int], comma_token_id: Optional[int]) -> List[int]:
        """
        Returns token indices where latent groups are inserted BEFORE(default) that token.
        In append mode, insertion happens after that token and uses the same index set.
        """
        strategy = self.config.insertion_strategy
        n = len(verbal_ids)

        if n == 0:
            return []
        if strategy == "start":
            return [0]
        if strategy == "end": #may need to change this if we use batched inputs, cuz each example may have different length end padding
            return [n] # (???) shouldn't this be n-1 and not n? No; the _augment_single method already checks 0-(n-1), using n here just marks that it's "end" insertion strategy
                                    #we NEED TO CHECK (maybe contact authors) if by "end" insertion they mean the latent tokens are really just the latent tokens attached to the first "response" token,
                                    #or, if they prepend the "end" tokens to the last of the query tokens(ie right before the last query token), like in figure 3.
        if strategy == "periodic":
            k = self.config.insertion_period
            # Insert before token i whenever i is a multiple of k.
            return [i for i in range(1, n) if i % k == 0]
        if strategy == "comma":
            if comma_token_id is None:
                raise ValueError("comma_token_id is required for insertion_strategy='comma'.")
            return [i for i, tok in enumerate(verbal_ids) if tok == comma_token_id]
        raise ValueError(f"Unsupported insertion strategy: {strategy}")

    def _augment_single(self, verbal_ids: Sequence[int], verbal_labels: Optional[Sequence[int]], verbal_loss_mask: Optional[Sequence[int]], comma_token_id: Optional[int]) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
        """
        Build one augmented (unpadded) sample by inserting latent slots into a verbal sequence.

        Params:
        - verbal_ids:
            One sample's verbal token ids (no padding). Same length defines the sequence before latent insertion.
            At training/validation time: usually query + response tokens in order (for teacher forcing).
            At test time: pass only the prompt first and grow the sequence
            step by step outside this function.
        - verbal_labels:
            Per-verbal-token targets for language modeling (same length as verbal_ids).
            At training/validation time: same sequence as verbal_ids (shifted automatically by HF causal LMs)
            At test time: None
        - verbal_loss_mask:
            Mask per verbal token, same length as verbal_ids: 1 = include this in loss
            position in the loss, 0 = ignore (e.g. mask the query/prompt, supervise only the answer span).
            At training/validation time: use 0 on query tokens and 1 on response tokens.
            At test time: None
        - comma_token_id:
            Vocabulary id for the comma token; required only when insertion_strategy is "comma".
            If needed, used in train, validation, and test time.
            

        Returns (all lists have identical length = augmented sequence length):
        - augmented_ids:
            Token ids(vocab number) after insertion. Verbal tokens keep original ids.
            Latent positions are placeholders with id -1 (replaced later by latent vectors).
        - position_ids:
            Position id for each augmented token.
            With freeze_position_ids=True, latent tokens share an anchor verbal position id. (DEFAULT)
            With freeze_position_ids=False, inserted positions increase naively.
        - labels_out:
            Training targets aligned to augmented_ids.
            Uses ignore_index for latent positions and for any verbal token masked out by verbal_loss_mask (for example, query-side tokens).
        - attention_mask:
            1 for each valid token in this sample (padding is added later in _augment_batch).
        - verbal_mask:
            1 for verbal tokens, 0 for latent tokens; used later to identify verbal positions.
        """
        insert_positions = set(self._insertion_positions(verbal_ids, comma_token_id))

        augmented_ids: List[int] = []
        position_ids: List[int] = []
        labels_out: List[int] = []
        attention_mask: List[int] = []
        verbal_mask: List[int] = []

        def add_latent_group(anchor_pos: int) -> None: 
            # Add m latent placeholders(id = -1) before corresponding anchor verbal token  #MAY NEED TO CHANGE latent placeholder id if -1 is already used
            # later _build_inputs_embeds() swaps these with learned latent vectors.
            for _ in range(self.config.num_latent_tokens):
                augmented_ids.append(-1)
                if self.config.freeze_position_ids:
                    #(default) latent tokens share position with anchor verbal token.
                    position_ids.append(anchor_pos)
                else:
                    # Naive increasing IDs for inserted tokens.
                    position_ids.append(len(position_ids))

                labels_out.append(self.ignore_index)
                attention_mask.append(1)
                verbal_mask.append(0)

        verbal_pos = 0 # verbal_pos is "position among verbal tokens only" (ignores inserted latent count).
        n = len(verbal_ids)
        for i, token_id in enumerate(verbal_ids):
            # Prepend latent tokens before token i.
            if (not self.config.append_mode) and (i in insert_positions):
                anchor = verbal_pos if self.config.freeze_position_ids else len(position_ids)
                add_latent_group(anchor)

            # Add verbal token.
            augmented_ids.append(int(token_id))
            # verbal_pos is "position among verbal tokens only" (ignores inserted latent count).
            position_ids.append(verbal_pos if self.config.freeze_position_ids else len(position_ids))
            attention_mask.append(1)
            verbal_mask.append(1)

            if verbal_labels is None:
                labels_out.append(self.ignore_index)
            else:
                label_val = int(verbal_labels[i])
                if verbal_loss_mask is not None and int(verbal_loss_mask[i]) == 0:
                    # Keep this target out of CE loss (e.g., query-side tokens).
                    labels_out.append(self.ignore_index)
                else:
                    labels_out.append(label_val)

            verbal_pos += 1

            # Append latent tokens after token i.  !!!!!!! TODO: NEED TO DOUBLE CHECK THIS LOGIC
            if self.config.append_mode and (i in insert_positions):
                anchor = verbal_pos - 1 if self.config.freeze_position_ids else len(position_ids)
                add_latent_group(anchor)

        # End insertion when strategy == "end" in prepend mode. !!!!!!! TODO: NEED TO DOUBLE CHECK THIS LOGIC 
                    #we NEED TO CHECK (maybe contact authors) if by "end" insertion they mean the latent tokens are really just the latent tokens attached to the first "response" token,
                    #or, if they prepend the "end" tokens to the last of the query tokens(ie right before the last query token), like in figure 3.
        if (not self.config.append_mode) and (n in insert_positions):
            anchor = verbal_pos if self.config.freeze_position_ids else len(position_ids)
            add_latent_group(anchor)

        # End insertion when strategy == "end" in append mode.
        if self.config.append_mode and (n in insert_positions) and n > 0:
            anchor = verbal_pos - 1 if self.config.freeze_position_ids else len(position_ids)
            add_latent_group(anchor)

        return augmented_ids, position_ids, labels_out, attention_mask, verbal_mask

    def _augment_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        loss_mask: Optional[torch.LongTensor],
        comma_token_id: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Augment each batch row with latent slots, then right-pad to a common length.

        Parameters (all tensors share the same batch dimension B):
        - input_ids: LongTensor [B, T_in]
            Verbal token ids. May be padded; use attention_mask to mark valid tokens.
        - attention_mask: optional LongTensor [B, T_in]
            1 for real tokens, 0 for padding. If omitted, all positions are treated as valid.
            Row b uses the first ``attention_mask[b].sum()`` positions from input_ids.
        - labels: optional LongTensor [B, T_in]
            Verbal-space targets aligned with input_ids (same valid-length slicing as above).
        - loss_mask: optional LongTensor [B, T_in]
            Verbal-space mask: 1 = include position in loss, 0 = ignore (e.g. query tokens).
        - comma_token_id: optional int
            Token id for comma when insertion_strategy is "comma"; unused otherwise.

        Returns a dict of LongTensors, each [B, T_aug] where T_aug is the maximum augmented
        sequence length in this batch (latent insertion lengthens rows; shorter rows are padded):
        - "augmented_ids": verbal ids with -1 at latent positions.
        - "position_ids": position id per augmented token.
        - "labels": augmented targets; ignore_index for latent and masked verbal positions.
        - "attention_mask": 1 for valid augmented tokens, 0 for right-padding.
        - "verbal_mask": 1 for verbal tokens, 0 for latent tokens.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        if attention_mask is None:
            # If caller provides padded input_ids without mask, treat all tokens as valid.
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        per_sample = []
        max_len = 0
        for b in range(batch_size):
            # Convert each sample to python lists for easier insertion logic.
            valid_len = int(attention_mask[b].sum().item())
            verbal_ids = input_ids[b, :valid_len].tolist()
            verbal_labels = labels[b, :valid_len].tolist() if labels is not None else None
            verbal_loss_mask = loss_mask[b, :valid_len].tolist() if loss_mask is not None else None

            out = self._augment_single(verbal_ids, verbal_labels, verbal_loss_mask, comma_token_id)
            per_sample.append(out)
            max_len = max(max_len, len(out[0]))

        aug_ids = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
        pos_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        aug_labels = torch.full(
            (batch_size, max_len), self.ignore_index, dtype=torch.long, device=device
        )
        aug_attn = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        verbal_mask_aug = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for b, (ids, pos, lbs, attn, vmask) in enumerate(per_sample):
            l = len(ids)
            # Right-pad each augmented sample to max_len.
            aug_ids[b, :l] = torch.tensor(ids, dtype=torch.long, device=device)
            pos_ids[b, :l] = torch.tensor(pos, dtype=torch.long, device=device)
            aug_labels[b, :l] = torch.tensor(lbs, dtype=torch.long, device=device)
            aug_attn[b, :l] = torch.tensor(attn, dtype=torch.long, device=device)
            verbal_mask_aug[b, :l] = torch.tensor(vmask, dtype=torch.long, device=device)

        return {
            "augmented_ids": aug_ids,
            "position_ids": pos_ids,
            "labels": aug_labels,
            "attention_mask": aug_attn,
            "verbal_mask": verbal_mask_aug,
        }

    def _build_inputs_embeds(self, augmented_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Replace placeholder -1 entries with latent embedding vectors.
        """
        input_embed_layer = self.base_model.get_input_embeddings()
        emb_w = input_embed_layer.weight
        # If the user moved the module or the HF model updated placement, keep latent rows aligned
        # with the verbal embedding table (same device/dtype as embeds we scatter into).
        if (
            self.latent_embeddings.weight.device != emb_w.device
            or self.latent_embeddings.weight.dtype != emb_w.dtype
        ):
            self.latent_embeddings.to(device=emb_w.device, dtype=emb_w.dtype)
        # Replace -1 with 0 before lookup so embedding() receives legal indices.
        # Those positions are overwritten with latent vectors right after.
        safe_token_ids = augmented_ids.clamp(min=0)
        embeds = input_embed_layer(safe_token_ids)

        latent_positions = augmented_ids.eq(-1)
        if latent_positions.any():
            # Build latent slot IDs [0, 1, ..., m-1] repeated per latent group.
            # Indices must live on the same device as latent_embeddings.weight.
            latent_slot_ids = torch.arange(
                self.config.num_latent_tokens, device=self.latent_embeddings.weight.device
            )
            latent_slot_ids = latent_slot_ids.unsqueeze(0).unsqueeze(0)

            # Repeat latent IDs across all latent groups in order.
            flat_count = int(latent_positions.sum().item())
            tiled = latent_slot_ids.view(-1).repeat((flat_count + self.config.num_latent_tokens - 1) // self.config.num_latent_tokens)
            tiled = tiled[:flat_count]

            latent_vecs = self.latent_embeddings(tiled)
            # Boolean indexing fills only latent placeholder positions.
            embeds[latent_positions] = latent_vecs.to(dtype=embeds.dtype)

        return embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.LongTensor] = None,
        comma_token_id: Optional[int] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: verbal token ids, shape [B, T]
            labels: optional verbal-space labels [B, T]
            loss_mask: optional verbal-space mask [B, T], 1=include in loss
            comma_token_id: required for insertion_strategy='comma'
        """
        self.base_model.eval() #even if external code calls model.train(), this line flips the pretrained model back to eval mode to ensure dropout is off / deterministic behavior
        augmented = self._augment_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            comma_token_id=comma_token_id,
        )
        inputs_embeds = self._build_inputs_embeds(augmented["augmented_ids"])

        # Important HF detail: when inputs_embeds is provided, input_ids is not needed.
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=augmented["attention_mask"],
            position_ids=augmented["position_ids"],
            labels=augmented["labels"] if labels is not None else None,
            use_cache=use_cache,
        )

        result = {
            "logits": outputs.logits,
            "verbal_mask": augmented["verbal_mask"],
            "attention_mask": augmented["attention_mask"],
            "position_ids": augmented["position_ids"],
            "augmented_ids": augmented["augmented_ids"],
        }
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        comma_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.0,
    ) -> torch.LongTensor:
        """
        Greedy (temperature=0) or sampled (temperature>0) generation in verbal token space.
        """
        self.eval()
        device = input_ids.device
        out = input_ids.clone()

        if attention_mask is None:
            attention_mask = torch.ones_like(out, dtype=torch.long, device=device)

        max_steps = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        stop_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        for _ in range(max_steps):
            # Re-run forward on current verbal sequence; forward() inserts latent tokens internally.
            fw = self.forward(
                input_ids=out,
                attention_mask=attention_mask,
                labels=None,
                loss_mask=None,
                comma_token_id=comma_token_id,
                use_cache=False,
            )
            logits = fw["logits"]  # [B, T_aug, V]
            verbal_mask = fw["verbal_mask"]  # [B, T_aug]

            next_tokens = []
            for b in range(logits.size(0)):
                # We pick logits at the LAST verbal position, not the last augmented position.
                # (Augmented positions include latent placeholders.)
                verbal_positions = torch.where(verbal_mask[b] == 1)[0]
                if verbal_positions.numel() == 0:
                    raise RuntimeError("No verbal positions found after augmentation.")
                last_verbal_idx = int(verbal_positions[-1].item())
                next_logit = logits[b, last_verbal_idx, :]

                if temperature > 0.0:
                    # Sampling mode.
                    probs = torch.softmax(next_logit / temperature, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding mode.
                    nxt = torch.argmax(next_logit, dim=-1, keepdim=True)
                next_tokens.append(nxt)

            next_tokens_t = torch.stack(next_tokens, dim=0).to(device).view(-1, 1)
            out = torch.cat([out, next_tokens_t], dim=1)
            # Attention mask tracks valid verbal tokens in out.
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.long, device=device)],
                dim=1,
            )

            if stop_id is not None and torch.all(next_tokens_t.squeeze(1) == stop_id):
                break

        return out

    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        comma_token_id: Optional[int] = None,  # currently unused for start/periodic scope
        eos_token_id: Optional[int] = None,
        temperature: float = 0.0,
    ) -> torch.LongTensor:
        self.eval()
        device = input_ids.device

        # Batch-size-1 assumption for this version.
        if input_ids.size(0) != 1:
            raise ValueError("generate_cached currently assumes batch_size == 1.")

        out = input_ids.clone()

        if attention_mask is None:
            attention_mask = torch.ones_like(out, dtype=torch.long, device=device)

        max_steps = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        stop_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # 1) Prefill full prompt once and get cache + first next-token logits
        prefill = self._prefill_with_cache(
            input_ids=out,
            attention_mask=attention_mask,
        )
        past = prefill["past_key_values"]                    # cache from full prompt
        next_logits = prefill["next_logits"]                 # [1, V]
        verbal_lens = prefill["verbal_lens"].clone()         # [1]
        augmented_lens = prefill["augmented_lens"].clone()   # [1]

        # First token prediction comes from prompt prefill logits
        next_token = self._sample_next_token(next_logits, temperature=temperature)  # [1,1]

        for _ in range(max_steps):
            # Append predicted verbal token
            out = torch.cat([out, next_token.to(device)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
                dim=1,
            )

            # EOS stop (single sample)
            if stop_id is not None and int(next_token[0, 0].item()) == int(stop_id):
                break

            # The appended token's verbal index is current verbal length before increment
            just_appended_verbal_index = verbal_lens.clone()  # [1]
            verbal_lens = verbal_lens + 1

            # 2) Build incremental augmented chunk for the token just appended
            # token_ids shape must be [B], here [1]
            chunk_lists = self._build_incremental_augmented_chunk(
                token_ids=next_token.view(-1),                    # [1]
                verbal_indices=just_appended_verbal_index,        # [1]
                total_aug_lens_so_far=augmented_lens,             # [1]
            )

            chunk_t = self._lists_to_batch_tensors(
                augmented_ids_list=chunk_lists["augmented_ids_list"],
                position_ids_list=chunk_lists["position_ids_list"],
                attention_mask_list=chunk_lists["attention_mask_list"],
                verbal_mask_list=chunk_lists["verbal_mask_list"],
                device=device,
            )

            # Update augmented length tracker
            augmented_lens = augmented_lens + chunk_t["attention_mask"].sum(dim=1).long()

            # 3) Cached incremental forward on chunk only
            chunk_embeds = self._build_inputs_embeds(chunk_t["augmented_ids"])
            outputs = self.base_model(
                inputs_embeds=chunk_embeds,
                attention_mask=chunk_t["attention_mask"],  # local chunk mask(note that some HF decoder implementations expect attention mask length to cover past + current tokens, not just current chunk)
                position_ids=chunk_t["position_ids"],
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values

            # 4) Select logits at last verbal position in chunk, then sample next token
            next_logits = self._select_next_logit_from_chunk(
                logits=outputs.logits,
                verbal_mask=chunk_t["verbal_mask"],
            )  # [1, V]
            next_token = self._sample_next_token(next_logits, temperature=temperature)  # [1,1]

        return out
    

    def _should_insert_at_verbal_index(self, verbal_index: int) -> bool:
        """
        Decide whether to prepend a latent group before verbal token at `verbal_index`.
        Supported strategies here: start, periodic.
        """
        strategy = self.config.insertion_strategy
        if strategy == "start":
            return verbal_index == 0
        if strategy == "periodic":
            k = int(self.config.insertion_period)
            if k <= 0:
                raise ValueError("insertion_period must be > 0 for periodic strategy.")
            # Match existing semantics: insert at indices 1,2,... where i % k == 0
            return verbal_index > 0 and (verbal_index % k == 0)
        raise ValueError(
            f"_should_insert_at_verbal_index only supports start/periodic here, got: {strategy}"
        )


    def _build_incremental_augmented_chunk(
        self,
        token_ids: torch.LongTensor,          # [B] new verbal token ids for this step
        verbal_indices: torch.LongTensor,     # [B] verbal index of each new token
        total_aug_lens_so_far: Optional[torch.LongTensor] = None,  # [B], only needed if freeze_position_ids=False
    ) -> dict:
        """
        Build one incremental augmented chunk per sample for prepend mode only.
        Output lists are ragged (variable chunk lengths), to be padded by _lists_to_batch_tensors().
        """
        if self.config.append_mode:
            raise ValueError("This helper currently supports prepend mode only (append_mode=False).")

        if token_ids.dim() != 1 or verbal_indices.dim() != 1:
            raise ValueError("token_ids and verbal_indices must be 1D tensors [B].")
        if token_ids.size(0) != verbal_indices.size(0):
            raise ValueError("token_ids and verbal_indices must have same batch size.")

        if (not self.config.freeze_position_ids) and total_aug_lens_so_far is None:
            raise ValueError("total_aug_lens_so_far is required when freeze_position_ids=False.")

        B = token_ids.size(0)

        ids_list: list[list[int]] = []
        pos_list: list[list[int]] = []
        attn_list: list[list[int]] = []
        vmask_list: list[list[int]] = []

        m = int(self.config.num_latent_tokens)
        for b in range(B):
            tok = int(token_ids[b].item())
            v_idx = int(verbal_indices[b].item())
            insert_now = self._should_insert_at_verbal_index(v_idx)

            row_ids: list[int] = []
            row_pos: list[int] = []
            row_attn: list[int] = []
            row_vmask: list[int] = []

            # Prepend latent group (if needed)
            if insert_now:
                for j in range(m):
                    row_ids.append(-1)
                    if self.config.freeze_position_ids:
                        # latent tokens share anchor verbal position
                        row_pos.append(v_idx)
                    else:
                        base = int(total_aug_lens_so_far[b].item())
                        row_pos.append(base + j)
                    row_attn.append(1)
                    row_vmask.append(0)

            # Add verbal token
            row_ids.append(tok)
            if self.config.freeze_position_ids:
                row_pos.append(v_idx)
            else:
                base = int(total_aug_lens_so_far[b].item())
                row_pos.append(base + (m if insert_now else 0))
            row_attn.append(1)
            row_vmask.append(1)

            ids_list.append(row_ids)
            pos_list.append(row_pos)
            attn_list.append(row_attn)
            vmask_list.append(row_vmask)

        return {
            "augmented_ids_list": ids_list,
            "position_ids_list": pos_list,
            "attention_mask_list": attn_list,
            "verbal_mask_list": vmask_list,
        }


    def _lists_to_batch_tensors(
        self,
        augmented_ids_list: list[list[int]],
        position_ids_list: list[list[int]],
        attention_mask_list: list[list[int]],
        verbal_mask_list: list[list[int]],
        device: torch.device,
    ) -> dict:
        """
        Right-pad ragged chunk lists and return batched tensors [B, T_chunk_max].
        """
        B = len(augmented_ids_list)
        if not (len(position_ids_list) == len(attention_mask_list) == len(verbal_mask_list) == B):
            raise ValueError("All input lists must have same batch size length.")

        max_len = max(len(x) for x in augmented_ids_list) if B > 0 else 0

        aug_ids = torch.full((B, max_len), -1, dtype=torch.long, device=device)
        pos_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
        attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
        vmask = torch.zeros((B, max_len), dtype=torch.long, device=device)

        for b in range(B):
            l = len(augmented_ids_list[b])
            aug_ids[b, :l] = torch.tensor(augmented_ids_list[b], dtype=torch.long, device=device)
            pos_ids[b, :l] = torch.tensor(position_ids_list[b], dtype=torch.long, device=device)
            attn[b, :l] = torch.tensor(attention_mask_list[b], dtype=torch.long, device=device)
            vmask[b, :l] = torch.tensor(verbal_mask_list[b], dtype=torch.long, device=device)

        return {
            "augmented_ids": aug_ids,
            "position_ids": pos_ids,
            "attention_mask": attn,
            "verbal_mask": vmask,
        }


    def _select_next_logit_from_chunk(
        self,
        logits: torch.FloatTensor,        # [B, T, V]
        verbal_mask: torch.LongTensor,    # [B, T], 1 verbal, 0 latent/pad
    ) -> torch.FloatTensor:
        """
        Select next-token logits at each row's last verbal position.
        Returns [B, V].
        """
        if logits.dim() != 3 or verbal_mask.dim() != 2:
            raise ValueError("Expected logits [B,T,V] and verbal_mask [B,T].")
        if logits.size(0) != verbal_mask.size(0) or logits.size(1) != verbal_mask.size(1):
            raise ValueError("Batch/time dims of logits and verbal_mask must match.")

        B = logits.size(0)
        next_logits = []
        for b in range(B):
            verbal_pos = torch.where(verbal_mask[b] == 1)[0]
            if verbal_pos.numel() == 0:
                raise RuntimeError(f"No verbal positions found for batch row {b}.")
            idx = int(verbal_pos[-1].item())
            next_logits.append(logits[b, idx, :])

        return torch.stack(next_logits, dim=0)  # [B, V]


    def _sample_next_token(
        self,
        next_logits: torch.FloatTensor,   # [B, V]
        temperature: float = 0.0,
    ) -> torch.LongTensor:
        """
        Greedy (temperature=0) or multinomial sampling (>0).
        Returns [B, 1].
        """
        if next_logits.dim() != 2:
            raise ValueError("next_logits must be [B, V].")

        if temperature > 0.0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # [B,1]
        else:
            nxt = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B,1]

        return nxt.long()


    def _prefill_with_cache(
        self,
        input_ids: torch.LongTensor,                  # [B, T_verbal]
        attention_mask: Optional[torch.LongTensor],   # [B, T_verbal] or None
    ) -> dict:
        """
        Run full prompt once, build cache, and return next-token logits from last verbal positions.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        # Build augmented prompt
        augmented = self._augment_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            loss_mask=None,
            comma_token_id=None if self.config.insertion_strategy != "comma" else None,
            # For your current scope (start/periodic), comma_token_id is unused.
        )

        inputs_embeds = self._build_inputs_embeds(augmented["augmented_ids"])

        # Prefill with cache enabled
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=augmented["attention_mask"],
            position_ids=augmented["position_ids"],
            use_cache=True,
        )

        next_logits = self._select_next_logit_from_chunk(
            logits=outputs.logits,
            verbal_mask=augmented["verbal_mask"],
        )

        verbal_lens = attention_mask.sum(dim=1).long()         # [B]
        augmented_lens = augmented["attention_mask"].sum(dim=1).long()  # [B]

        return {
            "past_key_values": outputs.past_key_values,
            "next_logits": next_logits,          # [B, V]
            "verbal_lens": verbal_lens,          # [B]
            "augmented_lens": augmented_lens,    # [B]
            "augmented_attention_mask": augmented["attention_mask"],  # [B, T_aug]
            "augmented_verbal_mask": augmented["verbal_mask"],        # [B, T_aug]
        }

    def extra_repr(self) -> str:
        cfg = asdict(self.config)
        short = {
            "model_name": cfg["model_name"],
            "num_latent_tokens": cfg["num_latent_tokens"],
            "insertion_strategy": cfg["insertion_strategy"],
            "insertion_period": cfg["insertion_period"],
            "freeze_position_ids": cfg["freeze_position_ids"],
            "append_mode": cfg["append_mode"],
        }
        return ", ".join(f"{k}={v}" for k, v in short.items())

