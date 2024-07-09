"""
API functions for sampling from anticipatory infilling models.
"""
import math

import torch
import torch.nn.functional as F

from tqdm import tqdm

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import * # TODO: Deprecate this
from anticipation.vocabs.tripletmidi import vocab


def safe_logits(logits, idx, curtime=None, allowed_control_pn=None):
    if allowed_control_pn is None:
        # don't generate controls
        logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf') 
    else:
        # don't generate (pitch,instr) tokens that do not correspond to allowed_control_pn
        instr = allowed_control_pn
        logits[ANOTE_OFFSET:(ANOTE_OFFSET+instr*MAX_PITCH)] = -float('inf')
        logits[(ANOTE_OFFSET+(instr+1)*MAX_PITCH):SPECIAL_OFFSET] = -float('inf')  

        # only generate anti-anticipated atime tokens 
        assert curtime is not None
        logits[ATIME_OFFSET+curtime:ATIME_OFFSET+MAX_TIME] = -float('inf')     
        
    logits[SPECIAL_OFFSET:] = -float('inf') # don't generate special tokens

    # don't generate stuff in the wrong time slot
    if idx % 3 == 0:
        logits[vocab['duration_offset'] : vocab['duration_offset'] + vocab['config']['max_duration']] = -float('inf')
        logits[vocab['note_offset']     : vocab['note_offset']     + vocab['config']['max_note']]     = -float('inf')
    elif idx % 3 == 1:
        logits[vocab['time_offset']     : vocab['time_offset']     + vocab['config']['max_time']]     = -float('inf')
        logits[vocab['note_offset']     : vocab['note_offset']     + vocab['config']['max_note']]     = -float('inf')
    elif idx % 3 == 2:
        logits[vocab['time_offset']     : vocab['time_offset']     + vocab['config']['max_time']]     = -float('inf')
        logits[vocab['duration_offset'] : vocab['duration_offset'] + vocab['config']['max_duration']] = -float('inf')

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

def control_prefix(instruments, human_instruments, task, vocab):
    task = vocab['task'][task]
    instr_offset = vocab['instrument_offset']
    human_instr_offset = vocab['human_instrument_offset']
    separator = vocab['separator']
    pad = vocab['pad']

    # get the list of instruments to condition on
    # by convention, let's provide the list sorted by instrument code
    instr_controls = sorted(instruments)
    instr_controls = [instr_offset + instr for instr in instruments]

    human_instr_controls = sorted(human_instruments)
    human_instr_controls = [human_instr_offset + instr for instr in human_instruments]

    instr_controls = instr_controls + human_instr_controls

    vocab_size = vocab['config']['size']
    assert max(instr_controls) < vocab_size

    # put task last, so the model knows it's time to generate events once it's seen the task token
    z_start = [separator] + instr_controls + [task]
    z_cont = instr_controls + [task]

    # pad the start controls out to an offset of 0 (mod 3)
    if len(z_start) % 3 > 0:
        z_start[1:1] = (3-len(z_start)%3)*[pad]

    # pad the continuation controls out to an offset of 1 (mod 3)
    if len(z_cont) % 3 > 0:
        z_cont[0:0] = (3-len(z_cont)%3)*[pad]
    z_cont = [pad] + z_cont

    return z_start, z_cont

def add_token(model, task, tokens, instruments, human_instruments, top_p, temperature, current_time, masked_instrs, allowed_control_pn=None, debug=False):
    pad = vocab['pad']

    assert len(tokens) % 3 == 0

    # get control global control prefix for the beginning of a sequence and the continuation of a sequence
    task_string = 'autoregress' if task == [AUTOREGRESS] else 'anticipate'
    z_start, z_cont = control_prefix(instruments, human_instruments, task_string, vocab)

    history = tokens.copy()
    prefix = None

    if (len(tokens) + len(z_start) + 1) >= 1024:
        lookback = len(tokens) - (1024 - len(z_cont) - 3) # we generate three tokens at a time
        prefix = z_cont
    else:
        lookback = max(len(tokens) - (1024 - len(z_start) - 1), 0)
        prefix = [pad] + z_start

    history = history[lookback:] # Markov window
    offset = ops.min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]] # relativize time in the history buffer

    new_token = []
    with torch.no_grad():
        for i in range(3):
            input_tokens = torch.tensor(prefix + history + new_token).unsqueeze(0).to(model.device)
            logits = model(input_tokens).logits[0,-1]

            idx = input_tokens.shape[1]-1
            logits = safe_logits(logits, idx, allowed_control_pn)
            if i == 0:
                logits = future_logits(logits, current_time - offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)

            
            logits = masked_instr_logits(logits, masked_instrs)
            
            logits = nucleus(logits, top_p)

            probs = F.softmax(logits/temperature, dim=-1)
            token = torch.multinomial(probs, 1)
            
            new_token.append(int(token))

    new_token[0] += offset # revert to full sequence timing
    if debug:
        print(f'  OFFSET = {offset}, LEN = {len(history)}, TIME = {tokens[::3][-5:]}')

    return new_token

def generate(model, start_time, end_time, inputs=None, chord_controls=None, human_controls=None, instruments=None, human_instruments=None, top_p=1.0, temperature=1.0, masked_instrs=[], debug=False, chord_delta=DELTA*TIME_RESOLUTION, human_delta=HUMAN_DELTA*TIME_RESOLUTION, return_controls=False, allowed_control_pn=None):
    
    if inputs is None:
        inputs = []

    if chord_controls is None:
        chord_controls = []

    if human_controls is None:
        human_controls = []

    if instruments is None:
        raise ValueError('Must provide list of instruments')

    if human_instruments is None:
        raise ValueError('Must provide list of human instruments s')

    start_time = int(TIME_RESOLUTION*start_time)
    end_time = int(TIME_RESOLUTION*end_time)

    # prompt is events up to start_time
    prompt = ops.pad(ops.clip(inputs, 0, start_time, clip_duration=False), start_time) # bug? start_time isn't in seconds, which is the default for ops.clip()

    # treat events beyond start_time as controls
    future = ops.clip(inputs, start_time+1, ops.max_time(inputs, seconds=False), clip_duration=False)
    if debug:
        print('Future')
        ops.print_tokens(future)

    # clip chord controls that preceed the sequence
    chord_controls = ops.clip(chord_controls, DELTA, ops.max_time(chord_controls, seconds=False), clip_duration=False)

    if debug:
        print('Chord Controls')
        ops.print_tokens(chord_controls)
        print('Human Controls')
        ops.print_tokens(human_controls)

    # task = [ANTICIPATE] if len(chord_controls) > 0 or len(future) > 0 or len(human_controls) > 0 else [AUTOREGRESS]
    task = [AUTOREGRESS] # always autoregress for now!
    if debug:
        print('AR Mode' if task[0] == AUTOREGRESS else 'AAR Mode')

    # interleave the chord_controls and human_controls with the events
    # note that we merge future with chord_controls, as they are both anticipated
    # tokens, controls = ops.anticipate(prompt, ops.sort(controls + [CONTROL_OFFSET+token for token in future]))
    tokens, chord_controls, human_controls = ops.anticipate_and_anti_anticipate(prompt, ops.sort(chord_controls + [CONTROL_OFFSET+token for token in future]), human_controls, chord_delta=chord_delta, human_delta=human_delta)
    
    if debug:
        print('Prompt')
        ops.print_tokens(tokens)

    current_time = ops.max_time(prompt, seconds=False)
    if debug:
        print('Current time:', current_time)

    with tqdm(range(end_time-start_time)) as progress:
        if chord_controls:
            atime, adur, anote = chord_controls[0:3]
            anticipated_tokens = chord_controls[3:]
            anticipated_time = atime - ATIME_OFFSET
        else:
            # nothing to anticipate
            anticipated_time = math.inf

        if human_controls:
            aatime, aadur, aanote = human_controls[0:3]
            anti_anticipated_tokens = human_controls[3:]
            anti_anticipated_time = aatime - ATIME_OFFSET
        else:
            # nothing to anti-anticipate
            anti_anticipated_time = math.inf

        while True:
            while (current_time >= anticipated_time - chord_delta) or (current_time >= anti_anticipated_time - human_delta):
                if (anticipated_time - chord_delta <= anti_anticipated_time - human_delta):
                    tokens.extend([atime, adur, anote])

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
                else:
                    tokens.extend([aatime, aadur, aanote])

                    if debug:
                        note = aanote - ANOTE_OFFSET
                        instr = note//2**7
                        print('A', aatime - ATIME_OFFSET, aadur - ADUR_OFFSET, instr, note - (2**7)*instr)

                    if len(anti_anticipated_tokens) > 0:
                        aatime, aadur, aanote = anti_anticipated_tokens[0:3]
                        anti_anticipated_tokens = anti_anticipated_tokens[3:]
                        anti_anticipated_time = aatime - ATIME_OFFSET
                    else:
                        # nothing more to anti-anticipate
                        anti_anticipated_time = math.inf

            new_token = add_token(model, task, tokens, instruments, human_instruments, top_p, temperature, max(start_time,current_time), masked_instrs, allowed_control_pn, debug)
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

    events, controls = ops.split(tokens)
    if return_controls:
        return ops.unpad(events), controls
    else:
        return ops.sort(ops.unpad(events) + future)
