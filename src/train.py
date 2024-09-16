import math
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
import torch
import json

import numpy as np

from torch.utils.data import DataLoader
from transformers import AdamW

from model import ImplicitModel
from configuration_model import ImplicitModelConfig
from data import CoTDataset, CoTDataCollator, extract_answer
from utils import get_sep_position, batch_ids, save_model


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


def compute_lambda_distribution(removal_smoothing_lambda, truncate_length=100):
    if removal_smoothing_lambda == float('inf'):
        lambda_distribution = torch.zeros(truncate_length)
        lambda_distribution[0] = 1
    else:
        positions = torch.arange(truncate_length)
        lambda_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
        cum_prob = lambda_distribution.sum()
        assert cum_prob <= 1
        lambda_distribution[-1] = lambda_distribution[-1] + (1-cum_prob)
    return lambda_distribution

    
@torch.no_grad()
def evaluate(dataloader, tokenizer, device, ctx, model, max_new_tokens, scheduled_to_remove, removal_side, removal_smoothing_lambda, lambda_distribution, keep_position=False, disable_random_removal_offset=False, ready_token="<|ready|>"):
    model.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    position_ids_all = None
    position_ids = None
    ready_id = tokenizer.encode(ready_token)[0]
    # for batch in tqdm.tqdm(dataloader):
    for batch in dataloader:
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        # start_tok = "<|start|>"
        # separator = tokenizer(start_tok, add_special_tokens=False)['input_ids'][0]
        # sep_positions = get_sep_position(input_ids_all, separator) #tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]

        # first_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        # # second_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=1)
        # second_sep_positions = get_sep_position(input_ids_all, ready_id)
        # # eos_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=2)
        # eos_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=1)

        # if scheduled_to_remove > 0 or removal_smoothing_lambda != float('inf'):
        #     if keep_position:
        #         position_ids_all = torch.arange(0, input_ids_all.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        #     input_ids_all_tmp = []
        #     labels_tmp = []
        #     random_removal_offset = torch.multinomial(lambda_distribution, batch_size, replacement=True).to(device)
        #     if disable_random_removal_offset:
        #         random_removal_offset.fill_(0)
        #     to_remove = scheduled_to_remove + random_removal_offset
        #     if removal_side == 'left':
        #         removal_from_positions = first_sep_positions + 1 # remove from, including
        #         removal_to_positions = first_sep_positions + 1 + to_remove # remove to, not including
        #     else: # removal_side == 'right'
        #         removal_to_positions = second_sep_positions
        #         removal_from_positions = second_sep_positions - to_remove

        #     for batch_id in range(input_ids_all.shape[0]):
        #         eos_position = eos_positions[batch_id]
        #         removal_from_position = removal_from_positions[batch_id]
        #         removal_to_position = removal_to_positions[batch_id]
        #         removal_from_position = max(removal_from_position, first_sep_positions[batch_id]+1)
        #         removal_to_position = min(removal_to_position, second_sep_positions[batch_id])
        #         if keep_position:
        #             position_ids_all[batch_id, removal_from_position-1:] += removal_to_position-removal_from_position
        #         input_ids_all_tmp.append(torch.cat((input_ids_all[batch_id, :removal_from_position], input_ids_all[batch_id, removal_to_position:eos_position+1]), dim=-1))
        #         labels_tmp.append(torch.cat((labels[batch_id, :removal_from_position], labels[batch_id, removal_to_position:eos_position+1]), dim=-1))
        #     input_ids_all = batch_ids(input_ids_all_tmp, tokenizer.eos_token_id, device, input_ids_all.dtype)
        #     labels = batch_ids(labels_tmp, -100, device, input_ids.dtype)

        # with ctx:
        #     if keep_position:
        #         position_ids_all = position_ids_all[:, :input_ids_all.shape[-1]]
        #     outputs = model.compute_loss(input_ids=input_ids_all, labels=labels, position_ids=position_ids_all)

        # total_loss += outputs.total_loss.item()
        # total_correct_tokens += outputs.total_correct.item()
        # total_tokens += outputs.total_tokens
        total_instances += batch_size

        # # Generate
        # stop_on_two_eos = True
        # if keep_position:
        #     position_ids = position_ids_all[:, :input_ids.shape[-1]]
        beam_output = model.generate(
            input_ids=input_ids,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            # stop_on_two_eos=stop_on_two_eos,
        )

        # Evaluate
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            sep_position = sep_positions[i].item()
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1

            if i == 0:
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target: {tgt_text}')
                print (f'Predicted: {pred_text}')
                print ('')
    accuracy = total_correct / total_instances
    # token_accuracy = total_correct_tokens / total_tokens
    # loss = total_loss / total_tokens
    # ppl = math.exp(loss)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--train_split', type=str, default="train")
    parser.add_argument('--val_split', type=str, default="valid")
    parser.add_argument('--test_split', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--max_remove_length', type=int, default=20)
    parser.add_argument('--max_len_train', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_size', type=int, default=-1)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--remove_start_from', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_interval', type=int, default=2500)
    parser.add_argument('--eval_interval', type=int, default=2500)
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument('--keep_position', action='store_true')
    parser.set_defaults(keep_position=False)
    parser.add_argument('--reinitialize_weights', action='store_true')
    parser.set_defaults(reinitialize_weights=False)
    # Remove Tokens
    parser.add_argument('--remove_tokens', action='store_true')
    parser.add_argument('--remove_from_step', type=float, default=5000)
    parser.add_argument('--remove_every_n_step', type=float, default=5000)
    parser.add_argument('--remove_all_when_remove_beyond', type=str, default='inf')
    parser.add_argument('--removal_smoothing_lambda', type=float, default=float('inf'))
    parser.add_argument('--removal_side', type=str, choices=['left', 'right'], default='left')
    # Switch Tokens
    parser.add_argument('--switch_tokens', action='store_true')
    parser.add_argument('--switch_from_step', type=float, default=5000)
    parser.add_argument('--switch_every_n_step', type=float, default=5000)
    parser.add_argument('--switch_token_ratio', type=float, default=1.0)
    parser.add_argument('--switch_from_rate', type=float, default=0.1)
    parser.add_argument('--switch_mode', type=str, choices=['depth', 'sequential'], default='sequential')
    args = parser.parse_args()

    if args.save_interval is None:
        save_interval = args.eval_interval
    else:
        save_interval = args.save_interval

    if args.remove_all_when_remove_beyond == 'inf':
        args.remove_all_when_remove_beyond = float('inf')
    else:
        args.remove_all_when_remove_beyond = int(args.remove_all_when_remove_beyond)
    with open(os.path.join(args.save_model, "train_args.json"), 'wt') as f:
        json.dump(vars(args), f, indent=4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    lambda_distribution = compute_lambda_distribution(args.removal_smoothing_lambda)
    print (lambda_distribution.tolist()[:10])

    dtype = 'float32'
    if args.bf16:
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create model
    if args.from_pretrained is None:
        config = ImplicitModelConfig(base_model=args.model)
        model = ImplicitModel(config).to(device).to(ptdtype)
    else:
        print (f'Loading from {args.from_pretrained}')
        model = ImplicitModel.from_pretrained(args.from_pretrained).to(device).to(ptdtype)
    if 'gpt2' in args.model:
        old_length = model.base_model.transformer.wpe.weight.shape[0]
        if args.truncation > old_length and args.from_pretrained is None:
            #import pdb; pdb.set_trace()
            print ('EXPANDING POSITIONs')
            new_wpe = torch.nn.Embedding(args.truncation, model.base_model.transformer.wpe.weight.shape[-1])
            new_wpe.weight.data[:old_length] = model.base_model.transformer.wpe.weight
            new_wpe.weight.data[old_length:] = model.base_model.transformer.wpe.weight[-1].view(1, -1).expand(args.truncation-old_length, -1)
            model.base_model.transformer.wpe = new_wpe

            for block in model.base_model.transformer.h:
                block.attn.register_buffer(
                    "bias",
                    torch.tril(torch.ones((args.truncation, args.truncation), dtype=torch.bool)).view(
                        1, 1, args.truncation, args.truncation
                ),
                persistent=False,
            )
    model = model.to(device).to(ptdtype)
    tokenizer = model.tokenizer
    tokenizer.add_tokens(["<|start|>", "<|pause|>", "<|ready|>"])
    model.tokenizer = tokenizer
    model.base_model.resize_token_embeddings(len(tokenizer)) 

    start_id = tokenizer.encode("<|start|>")[0]
    pause_id = tokenizer.encode("<|pause|>")[0]
    ready_id = tokenizer.encode("<|ready|>")[0]

    if args.reinitialize_weights:
        print ('reinitializing weights')
        model.model.apply(model.model._init_weights)

    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet

    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    print("Building training dataset loader")
    train_dataset = CoTDataset(tokenizer, args.data_path, args.truncation, data_name=args.data_name, max_size=args.max_size, split=args.train_split)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    print("Building validation dataset loader")
    val_dataset = CoTDataset(tokenizer, args.data_path, args.truncation, data_name=args.data_name, split=args.val_split)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    if args.test_split:
        print("Building test dataset loader")
        test_dataset = CoTDataset(tokenizer, args.data_path, args.truncation, data_name=args.data_name, split=args.test_split)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    # Train
    step = 0
    position_ids = None

    steps_per_epoch = len(train_dataloader)
    max_epochs = args.train_steps // steps_per_epoch
    steps_per_removed_token = int(round(steps_per_epoch / args.remove_per_epoch))
    steps_per_switched_token = int(round(steps_per_epoch * (args.epochs - args.switch_start_from)) / 100)
    remove_step_counter = 0
    switch_step_counter = steps_per_switched_token
    switch_ratio = args.switch_ratio
    best_val_accuracy = float('-inf')

    scheduled_to_switch = 0
    if args.switch_tokens > 0:
        print (f'the number of switched CoT tokens starts from {args.switch_start_from}')
        scheduled_to_switch = args.switch_start_from

    scheduled_to_remove = 0
    if args.remove_tokens > 0:
        print (f'the number of removed CoT tokens starts from {args.remove_start_from}')
        scheduled_to_remove = args.remove_start_from

    print(f'Training for {args.train_steps} steps ({max_epochs} epochs)')

    all_cot_removed_in_prev_batch = False
    for epoch in range(0, max_epochs):
        if scheduled_to_remove < float('inf'):
            scheduled_to_remove = int(round(scheduled_to_remove))
        if scheduled_to_remove >= args.remove_all_when_remove_beyond:
            scheduled_to_remove = float('inf') # remove all
        model.train()

        for batch in train_dataloader:

            input_ids = batch['input_ids_all'] #.to(device)
            labels = batch['labels_all'] #.to(device)
            batch_size = input_ids.shape[0]

            if args.remove_tokens:
                prev_scheduled_to_remove = scheduled_to_remove
                if remove_step_counter == steps_per_removed_token or steps_per_removed_token == 0:
                    scheduled_to_remove += 1
                    remove_step_counter = 0
                if scheduled_to_remove > prev_scheduled_to_remove:
                    print (f'Scheduled to remove: {scheduled_to_remove}')
                    print(f" -epoch {epoch}. step {step}. removing: {scheduled_to_remove}")
                    if args.reset_optimizer and (not all_cot_removed_in_prev_batch):
                        print ('RESETTING OPTIMIZER')
                        optimizer.zero_grad(set_to_none=True)
                        del optimizer
                        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
                if scheduled_to_remove >= args.remove_all_when_remove_beyond:
                    scheduled_to_remove = float('inf') # remove all
                elif scheduled_to_remove >= args.max_remove_length:
                    scheduled_to_remove = args.max_remove_length

                first_sep_positions = get_sep_position(input_ids, start_id)
                second_sep_positions = get_sep_position(input_ids, ready_id)
                eos_positions = get_sep_position(input_ids, tokenizer.eos_token_id)

                all_cot_removed_in_batch = False
                if scheduled_to_remove > 0 or args.removal_smoothing_lambda != float('inf'):
                    input_ids_tmp = []
                    labels_tmp = []
                    random_removal_offset = torch.multinomial(lambda_distribution, batch_size, replacement=True) #.to(device)
                    to_remove = scheduled_to_remove + random_removal_offset
                    if args.keep_position:
                        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
                    if args.removal_side == 'left':
                        removal_from_positions = first_sep_positions + 1 # remove from, including
                        removal_to_positions = first_sep_positions + 1 + to_remove # remove to, not including
                    else: # removal_side == 'right'
                        removal_to_positions = second_sep_positions
                        removal_from_positions = second_sep_positions - to_remove

                    all_cot_removed_in_batch = True
                    for batch_id in range(input_ids.shape[0]):
                        eos_position = eos_positions[batch_id]
                        removal_from_position = removal_from_positions[batch_id]
                        removal_to_position = removal_to_positions[batch_id]
                        removal_from_position = max(removal_from_position, first_sep_positions[batch_id]+1)
                        if removal_to_position < second_sep_positions[batch_id]:
                            all_cot_removed_in_batch = False
                        removal_to_position = min(removal_to_position, second_sep_positions[batch_id])
                        if args.keep_position:
                            position_ids[batch_id, removal_from_position-1:] += removal_to_position-removal_from_position
                        input_ids_tmp.append(torch.cat((input_ids[batch_id, :removal_from_position], input_ids[batch_id, removal_to_position:eos_position+1]), dim=-1))
                        labels_tmp.append(torch.cat((labels[batch_id, :removal_from_position], labels[batch_id, removal_to_position:eos_position+1]), dim=-1))
                    input_ids = batch_ids(input_ids_tmp, tokenizer.eos_token_id, device, input_ids.dtype)
                    labels = batch_ids(labels_tmp, -100, device, input_ids.dtype)
                    if not all_cot_removed_in_batch:
                        best_val_accuracy = float('-inf')

                all_cot_removed_in_prev_batch = all_cot_removed_in_batch

            if epoch >= scheduled_to_switch and args.switch_tokens:
                if switch_step_counter == steps_per_switched_token or steps_per_switched_token == 0:
                    model.base_model.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}_step_{step}_switch_rate_{switch_ratio}'), from_pt=True)
                    model.tokenizer.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}_step_{step}_switch_rate_{switch_ratio}'))
                    print(f" -epoch {epoch}. step {step}. switching rate: {switch_ratio}%")
                    switch_ratio += 1.0
                    switch_step_counter = 0
                else:
                    switch_step_counter += 1

                # first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
                first_sep_positions = get_sep_position(input_ids, start_id)
                # second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
                second_sep_positions = get_sep_position(input_ids, ready_id)
                # eos_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)
                eos_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
                delta_sep_positions = second_sep_positions - first_sep_positions
                
                if switch_ratio > 0:

                    input_ids_tmp = []
                    labels_tmp = []

                    switch_prob = switch_ratio/100.0
                    if switch_prob > 1.0:
                        switch_prob = 1.0
                    for batch_id in range(input_ids.shape[0]):
                        cot_length = int(delta_sep_positions[batch_id].cpu())
                        filler_mask = np.random.choice([True, False], cot_length, p=[switch_prob, 1-switch_prob])

                        if sum(filler_mask) == 0:
                            input_ids_tmp.append(input_ids[batch_id])
                            labels_tmp.append(labels[batch_id])
                            continue

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
                        cot_tokens = input_ids[batch_id][first_sep_positions[batch_id]:second_sep_positions[batch_id]]
                        cot_tokens_tmp = []
                        for idx, (start, end) in enumerate(switch_index):
                            if (idx == 0) and (start > 0):
                                cot_tokens_tmp.append(cot_tokens[:start])
                            seq_leng = end - start
                            num_tokens = int(np.ceil(seq_leng/args.switch_token_replace))
                            cot_tokens_tmp.append(torch.as_tensor([pause_id]*num_tokens)) #.to(device)

                            if (idx == len(switch_index)-1) and (end < cot_length):
                                cot_tokens_tmp.append(cot_tokens[end:])
                        
                        cot_tokens_tmp = torch.cat(cot_tokens_tmp)
                        # if cot_tokens_tmp[-1] == pause_id:
                        #     cot_tokens_tmp[-1] = ready_id

                        input_ids_tmp.append(
                            torch.cat((
                                input_ids[batch_id][:first_sep_positions[batch_id]],
                                cot_tokens_tmp,
                                input_ids[batch_id][second_sep_positions[batch_id]:]
                                ), dim=-1)
                            )

                        labels_tmp.append(
                            torch.cat((
                                labels[batch_id][:first_sep_positions[batch_id]],
                                cot_tokens_tmp,
                                labels[batch_id][second_sep_positions[batch_id]:]
                                ), dim=-1)
                            )
                        
                        del cot_tokens_tmp

                    input_ids = batch_ids(input_ids_tmp, tokenizer.eos_token_id, device, input_ids.dtype)
                    labels = batch_ids(labels_tmp, -100, device, input_ids.dtype)

                    del input_ids_tmp, labels_tmp

            # if (not args.switch_tokens) and (not args.remove_tokens):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # if not_printed == False:
            if step == 0:
                print("Sample Input")
                print(input_ids[0])
                print("Sample Label")
                print(labels[0])
                not_printed = True
                
            if args.max_len_train > 0 and input_ids.shape[-1] > args.max_len_train:
                print ('skipped')
                continue
           
            with ctx:
                if args.keep_position:
                    position_ids = position_ids[:, :input_ids.shape[-1]]
                outputs = model.compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)
            loss = outputs.loss
            loss.div(args.accumulate).backward()
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step != 0 and step % 100 == 0:
                token_accuracy = outputs.token_accuracy.item()
                ppl = loss.exp().item()
                print(f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")

            if step != 0 and step % save_interval == 0:
                #Save here
                model.base_model.save_pretrained(os.path.join(args.save_model, f'step_{step}'), from_pt=True)
                model.tokenizer.save_pretrained(os.path.join(args.save_model, f'step_{step}'))

            if step != 0 and step % args.eval_interval == 0:
                accuracy = evaluate(val_dataloader, tokenizer, device, ctx, model, args.max_new_tokens, scheduled_to_remove, args.removal_side, args.removal_smoothing_lambda, lambda_distribution, keep_position=args.keep_position, disable_random_removal_offset=True)
                print (f'Step {step} - Evalation on Valid Split; Accuracy: {accuracy}.')
                if accuracy > best_val_accuracy:
                    print (f'Obtained better val accuracy')
                    best_val_accuracy = accuracy
                    if args.test_split:
                        accuracy = evaluate(test_dataloader, tokenizer, device, ctx, model, args.max_new_tokens, scheduled_to_remove, args.removal_side, args.removal_smoothing_lambda, lambda_distribution, keep_position=args.keep_position, disable_random_removal_offset=True)
                        print(f'Step {step} - Evalation on Test Split; Accuracy: {accuracy}.')

            step += 1

if __name__ == "__main__":
    main()
