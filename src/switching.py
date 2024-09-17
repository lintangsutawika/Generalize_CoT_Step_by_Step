import torch
import numpy as np
from utils import get_single_sep_position

def switch_random(inputs, labels, tokenizer=None, replace_ratio=1.0, start_id=0, ready_id=0, pause_id=0, eos_id=0, switch_prob=0.5):

    first_sep_position = get_single_sep_position(inputs, start_id)+1
    second_sep_position = get_single_sep_position(inputs, ready_id)
    eos_position = get_single_sep_position(inputs, eos_id)
    delta_sep_position = second_sep_position - first_sep_position

    cot_length = int(delta_sep_position)
    filler_mask = np.random.rand(cot_length) < switch_prob

    if sum(filler_mask) == 0:
        return inputs, labels

    # Find indices where the value changes from False to True
    start_indices = np.where(np.diff(filler_mask.astype(int)) == 1)[0] + 1
    # Find indices where the value changes from True to False
    end_indices = np.where(np.diff(filler_mask.astype(int)) == -1)[0] + 1
    # If the array starts with True, add index 0
    if filler_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    # If the array ends with True, add the last index
    if filler_mask[-1]:
        end_indices = np.append(end_indices, len(filler_mask))
    # Pair start and end indices
    switch_index = np.column_stack((start_indices, end_indices))
    cot_tokens = inputs[first_sep_position:second_sep_position]
    cot_tokens_tmp = []
    token_idx = 0
    for idx, (start, end) in enumerate(switch_index):

        cot_tokens_tmp.append(cot_tokens[token_idx:start])

        seq_leng = end - start
        num_tokens = int(np.ceil(seq_leng * replace_ratio))
        cot_tokens_tmp.append(torch.as_tensor([pause_id]*num_tokens)) #.to(device)
        token_idx = end

        if (idx == len(switch_index)-1) and (end < cot_length):
            cot_tokens_tmp.append(cot_tokens[end:])
    
    cot_tokens_tmp = torch.cat(cot_tokens_tmp)

    inputs_tmp = torch.cat((
        inputs[:first_sep_position],
        cot_tokens_tmp,
        inputs[second_sep_position:]
        ), dim=-1)

    labels_tmp = torch.cat((
        labels[:first_sep_position],
        cot_tokens_tmp,
        labels[second_sep_position:]
        ), dim=-1)
    
    del cot_tokens_tmp
    return inputs_tmp, labels_tmp


def switch_sequence(inputs, labels, tokenizer=None, replace_ratio=1.0, start_id=0, ready_id=0, pause_id=0, eos_id=0, switch_prob=0.5):

    first_sep_position = get_single_sep_position(inputs, start_id)+1
    second_sep_position = get_single_sep_position(inputs, ready_id)
    eos_position = get_single_sep_position(inputs, eos_id)
    delta_sep_position = second_sep_position - first_sep_position

    cot_length = int(delta_sep_position)

    filler_length = sum(np.random.rand(cot_length) < switch_prob)
    filler_mask = np.array([False]*cot_length)
    filler_mask[:filler_length] = True

    if sum(filler_mask) == 0:
        return inputs, labels

    # Find indices where the value changes from False to True
    start_indices = np.where(np.diff(filler_mask.astype(int)) == 1)[0] + 1
    # Find indices where the value changes from True to False
    end_indices = np.where(np.diff(filler_mask.astype(int)) == -1)[0] + 1
    # If the array starts with True, add index 0
    if filler_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    # If the array ends with True, add the last index
    if filler_mask[-1]:
        end_indices = np.append(end_indices, len(filler_mask))
    # Pair start and end indices
    switch_index = np.column_stack((start_indices, end_indices))
    cot_tokens = inputs[first_sep_position:second_sep_position]
    cot_tokens_tmp = []
    token_idx = 0
    for idx, (start, end) in enumerate(switch_index):

        cot_tokens_tmp.append(cot_tokens[token_idx:start])

        seq_leng = end - start
        num_tokens = int(np.ceil(seq_leng * replace_ratio))
        cot_tokens_tmp.append(torch.as_tensor([pause_id]*num_tokens)) #.to(device)
        token_idx = end

        if (idx == len(switch_index)-1) and (end < cot_length):
            cot_tokens_tmp.append(cot_tokens[end:])
    
    cot_tokens_tmp = torch.cat(cot_tokens_tmp)

    inputs_tmp = torch.cat((
        inputs[:first_sep_position],
        cot_tokens_tmp,
        inputs[second_sep_position:]
        ), dim=-1)

    labels_tmp = torch.cat((
        labels[:first_sep_position],
        cot_tokens_tmp,
        labels[second_sep_position:]
        ), dim=-1)
    
    del cot_tokens_tmp
    return inputs_tmp, labels_tmp


# def switch_depth()



