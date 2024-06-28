"""
API functions for sampling from anticipatory infilling models.
"""
from typing import List

import math

import torch
import torch.nn.functional as F
import tvm
import numpy as np

from tqdm import tqdm

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *


def safe_logits(logits, idx):
    logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf') # don't generate controls
    logits[SPECIAL_OFFSET:] = -float('inf')               # don't generate special tokens

    # don't generate stuff in the wrong time slot
    if idx % 3 == 0:
        logits[DUR_OFFSET:DUR_OFFSET+MAX_DUR] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET+MAX_NOTE] = -float('inf')
    elif idx % 3 == 1:
        logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET+MAX_NOTE] = -float('inf')
    elif idx % 3 == 2:
        logits[TIME_OFFSET:TIME_OFFSET+MAX_TIME] = -float('inf')
        logits[DUR_OFFSET:DUR_OFFSET+MAX_DUR] = -float('inf')

    return logits


def nucleus(logits, top_p):
    # from HF implementation
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("inf")

    return logits


def future_logits(logits, curtime):
    """ don't sample events in the past """
    if curtime > 0:
        logits[TIME_OFFSET:TIME_OFFSET+curtime] = -float('inf')

    return logits


def instr_logits(logits, full_history):
    """ don't sample more than 16 instruments """
    instrs = ops.get_instruments(full_history)
    if len(instrs) < 16:
        return logits

    for instr in range(MAX_INSTR):
        if instr not in instrs:
            logits[NOTE_OFFSET+instr*MAX_PITCH:NOTE_OFFSET+(instr+1)*MAX_PITCH] = -float('inf')

    return logits


def masked_instr_logits(logits, masked_instrs):
    """ supress the given instruments """
    for instr in masked_instrs:
        logits[NOTE_OFFSET+instr*MAX_PITCH:NOTE_OFFSET+(instr+1)*MAX_PITCH] = -float('inf')

    return logits


def construct_prompt(z, tokens, cache):
    history = tokens.copy()
    lookback = max(len(tokens) - 768 - (len(tokens) % 255), 0) # quantize lookback for caching
    history = history[lookback:] # Markov window
    if len(tokens) % 255 == 0:
        cache = None # if we hopped, flush the cache

    offset = ops.min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    input_ids = torch.tensor(z + history)
    if cache:
        input_ids = input_ids[-1:]

    return input_ids, cache, offset


def add_token(model, z, tokens, top_p, temperature, current_time, masked_instrs, cache, debug=False, use_MLC=False):
    assert len(tokens) % 3 == 0

    new_token = []
    input_ids, cache, offset = construct_prompt(z, tokens, cache)
    with torch.no_grad():
        for i in range(3):
                        
            if not use_MLC:
                input_ids = input_ids.unsqueeze(0).to(model.device)
                output = model(input_ids, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = output.logits[0,-1]
            else:
                logits, cache = debugchat_forward(model, input_ids, cache)
                logits = torch.tensor(logits)[0,0,:]

            idx = len(tokens) + i
            logits = safe_logits(logits, idx)
            if i == 0:
                logits = future_logits(logits, current_time - offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)
            logits = masked_instr_logits(logits, masked_instrs)
            logits = nucleus(logits, top_p)

            probs = F.softmax(logits/temperature, dim=-1)
            input_ids = torch.multinomial(probs, 1)
            new_token.append(int(input_ids))

    new_token[0] += offset # revert to full sequence timing
    if debug:
        print(f'  OFFSET = {offset}, TIME = {tokens[::3][-5:]}')

    return new_token, cache


def generate(model, start_time, end_time, inputs=None, controls=None, top_p=1.0, temperature=1.0, masked_instrs=[], debug=False, delta=DELTA*TIME_RESOLUTION, use_MLC=False):
    if inputs is None:
        inputs = []

    if controls is None:
        controls = []

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, clip_duration=False), start_time)

    # treat events beyond start_time as controls
    future = ops.clip(inputs, start_time+1, ops.max_time(inputs, seconds=False), clip_duration=False)
    if debug:
        print('Future')
        ops.print_tokens(future)

    # clip controls that preceed the sequence
    controls = ops.clip(controls, DELTA, ops.max_time(controls, seconds=False), clip_duration=False)

    if debug:
        print('Controls')
        ops.print_tokens(controls)

    z = [ANTICIPATE] if len(controls) > 0 or len(future) > 0 else [AUTOREGRESS]
    if debug:
        print('AR Mode' if z[0] == AUTOREGRESS else 'AAR Mode')

    # interleave the controls with the events
    tokens, controls = ops.anticipate(prompt, ops.sort(controls + [CONTROL_OFFSET+token for token in future]))

    if debug:
        print('Prompt')
        ops.print_tokens(tokens)

    current_time = ops.max_time(prompt, seconds=False)
    if debug:
        print('Current time:', current_time)

    with tqdm(range(end_time-start_time)) as progress:
        if controls:
            atime, adur, anote = controls[0:3]
            anticipated_tokens = controls[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        cache = None
        while True:
            # anticipate any upcoming controls
            while current_time >= anticipated_time - delta:
                # update the cache
                input_ids, cache, offset = construct_prompt(z, tokens, cache)
                for new_token in [atime, adur, anote]:
                    with torch.no_grad():
                        # run the model as if we were going to use its prediction
                        if not use_MLC:
                            input_ids = input_ids.unsqueeze(0).to(model.device)
                            cache = model(input_ids, past_key_values=cache, use_cache=True).past_key_values
                        else:
                            _, cache = debugchat_forward(model, input_ids, cache)

                    tokens.append(new_token)
                    input_ids = torch.tensor(new_token).unsqueeze(0)

                if debug:
                    note = anote - ANOTE_OFFSET
                    instr = note//2**7
                    print('A', atime - ATIME_OFFSET, adur - ADUR_OFFSET, instr, note - (2**7)*instr)

                if len(anticipated_tokens) > 0:
                    atime, adur, anote = anticipated_tokens[0:3]
                    anticipated_tokens = anticipated_tokens[3:]
                    anticipated_time = atime - ATIME_OFFSET
                else:
                    # nothing more to anticipate
                    anticipated_time = math.inf

            new_token, cache = add_token(model, z, tokens, top_p, temperature, max(start_time,current_time), masked_instrs, cache, use_MLC=use_MLC)
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note//2**7
                new_pitch = new_note - (2**7)*new_instr
                print('C', new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)

    events, _ = ops.split(tokens)
    return ops.sort(ops.unpad(events) + future)


def debugchat_forward(
    dc,
    input_tokens,
    kv_caches
):
    """
    Parameters
    ----------
    dc : DebugChat
        The DebugChat object that contains the model and tokenizer
        for generating the response.
        
    input_tokens : List[str]
        Either a prompt to the model if kv_caches is None, or the last token.

    temperature : float
        Softmax temperature for sampling.
        
    top_p : float
        Nucleus sampling parameter.
    """

    assert((len(input_tokens) == 1 and kv_caches is not None) or (kv_caches is None))

    if kv_caches is None:
        input_tokens = tvm.nd.array(np.array(input_tokens).astype("int32"), device=dc.device)
        embedding, input_len = dc._embed(input_tokens)
        logits, kv_caches = dc._prefill(embedding, input_len)
    else:
        last_token = input_tokens[-1]
        logits = dc._decode(last_token, kv_caches)
    
    return logits.numpy(), kv_caches