def _host_tuple_or_none(values: tuple[int, ...] | list[int]) -> tuple[int, ...] | None:
    if len(values) == 0:
        return None
    return tuple(int(v) for v in values)


def _causal_conv1d_custom_torch(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias_opt: torch.Tensor | None,
    query_start_loc_opt: tuple[int, ...] | list[int],
    cache_indices_opt: tuple[int, ...] | list[int],
    initial_state_mode_opt: tuple[int, ...] | list[int],
    num_accepted_tokens_opt: tuple[int, ...] | list[int],
    activation_mode: int,
    pad_slot_id: int,
    run_mode: int,
) -> torch.Tensor:
    """Torch implementation matching npu_causal_conv1d_custom's Python ABI."""
    if x.dim() != 2:
        raise RuntimeError(f"Unsupported x shape for causal_conv1d_custom torch fallback: {tuple(x.shape)}")
    if conv_state.dim() != 3:
        raise RuntimeError(
            f"Unsupported conv_state shape for causal_conv1d_custom torch fallback: {tuple(conv_state.shape)}"
        )

    dim = x.shape[-1]
    if weight.shape[-1] == dim:
        weight_for_conv = weight.transpose(0, 1).contiguous()
    else:
        weight_for_conv = weight.contiguous()
    if weight_for_conv.shape[0] != dim:
        raise RuntimeError(
            f"causal_conv1d_custom torch fallback: weight dim mismatch, "
            f"x dim={dim}, weight.shape={tuple(weight.shape)}"
        )

    width = weight_for_conv.shape[1]
    state_len = width - 1
    activation = "silu" if activation_mode else None
    query_start_loc = _host_tuple_or_none(query_start_loc_opt)
    cache_indices = _host_tuple_or_none(cache_indices_opt)
    has_initial_state = _host_tuple_or_none(initial_state_mode_opt)
    num_accepted_tokens = _host_tuple_or_none(num_accepted_tokens_opt)

    if query_start_loc is None or cache_indices is None:
        raise RuntimeError("causal_conv1d_custom torch fallback requires query_start_loc_opt and cache_indices_opt.")

    seqlens = [end - start for start, end in zip(query_start_loc[:-1], query_start_loc[1:])]
    max_query_len = max(seqlens, default=0)
    output.zero_()

    def _conv(seq_tokens: torch.Tensor, initial_state: torch.Tensor | None) -> torch.Tensor:
        conv_x = seq_tokens.transpose(0, 1).unsqueeze(0).to(weight_for_conv.dtype)
        bias = bias_opt.to(weight_for_conv.dtype) if bias_opt is not None else None
        if initial_state is None:
            conv_out = F.conv1d(conv_x, weight_for_conv.unsqueeze(1), bias, padding=state_len, groups=dim)
        else:
            conv_x = torch.cat([initial_state.to(weight_for_conv.dtype), conv_x], dim=-1)
            conv_out = F.conv1d(conv_x, weight_for_conv.unsqueeze(1), bias, padding=0, groups=dim)
        conv_out = conv_out[..., : seq_tokens.shape[0]]
        if activation in ("silu", "swish"):
            conv_out = F.silu(conv_out)
        return conv_out.squeeze(0).transpose(0, 1).to(output.dtype)

    for i, seq_len in enumerate(seqlens):
        cache_idx = cache_indices[i]
        if cache_idx == pad_slot_id or seq_len <= 0:
            continue

        start = query_start_loc[i]
        end = start + int(seq_len)
        seq_tokens = x[start:end]
        state = conv_state[cache_idx]

        if run_mode == 0:
            use_initial_state = has_initial_state is not None and bool(has_initial_state[i])
            initial_state = state[:state_len].transpose(0, 1).unsqueeze(0) if use_initial_state else None
            output[start:end].copy_(_conv(seq_tokens, initial_state))

            if use_initial_state:
                state_source = torch.cat([initial_state.squeeze(0).transpose(0, 1), seq_tokens], dim=0)
            else:
                state_source = seq_tokens
            final_state = F.pad(state_source.transpose(0, 1), (state_len - state_source.shape[0], 0))
            state[:state_len].copy_(final_state[:, -state_len:].transpose(0, 1).to(conv_state.dtype))
            continue

        if run_mode != 1:
            raise RuntimeError(f"Unsupported causal_conv1d_custom run_mode for torch fallback: {run_mode}")

        if num_accepted_tokens is None:
            state_offset = 0
            effective_state_len = state_len
            shift = seq_len
        else:
            state_offset = num_accepted_tokens[i] - 1
            effective_state_len = state_len + max_query_len - 1 - (max_query_len - seq_len)
            shift = 1

        initial_state = state[state_offset : state_offset + state_len].transpose(0, 1).unsqueeze(0)
        output[start:end].copy_(_conv(seq_tokens, initial_state))

        old_window = state[state_offset : state_offset + effective_state_len].clone()
        if seq_len >= effective_state_len:
            updated = seq_tokens[seq_len - effective_state_len : seq_len]
        else:
            keep = effective_state_len - seq_len
            updated = torch.cat([old_window[shift : shift + keep], seq_tokens], dim=0)
        state[:effective_state_len].copy_(updated.to(conv_state.dtype))

    return output
