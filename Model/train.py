import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer, BertModel
from Data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, T5Translator, R1Translator, BrainBERT
from config import get_config

def train_model(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best='./checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last='./checkpoints/decoding/last/temp_decoding.pt'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'dev']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0

            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels in tqdm(dataloaders[phase]):
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
                    loss = seq2seqLMoutput.loss

                    if phase == 'train':
                        loss.sum().backward()
                        optimizer.step()

                running_loss += loss.sum().item() * input_embeddings_batch.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'Updated best checkpoint: {checkpoint_path_best}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'Updated last checkpoint: {checkpoint_path_last}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    args = get_config('train_decoding')
    dataset_setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    train_input = args['train_input']
    save_path = args['save_path']
    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']
    device_ids = [0]
    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']

    if use_random_init and skip_step_one:
        step2_lr = 5e-4

    print(f'[INFO] Using model: {model_name}')
    save_name = f'{task_name}_finetune_{model_name}_{"skipstep1" if skip_step_one else "2steptraining"}_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    if use_random_init:
        save_name = f'randinit_{save_name}'

    os.makedirs(save_path, exist_ok=True)
    save_path_best = os.path.join(save_path, 'best')
    os.makedirs(save_path_best, exist_ok=True)
    save_path_last = os.path.join(save_path, 'last')
    os.makedir(save_path_last, exist_ok=True)
    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')
    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    print(f'[INFO] Using subjects: {subject_choice}, EEG type: {eeg_type_choice}, bands: {bands_choice}')

    torch.manual_seed(312)
    if torch.cuda.is_available():
        device = torch.device(args['cuda'])
    else:
        device = torch.device('cpu')
    print(f'[INFO] Using device: {device}')

    whole_dataset_dicts = []
    dataset_paths = {
        'task1v1': '/home/graduation project/code/EEG-Code/Data/pickle_file/task1-SR-dataset.pickle',
        'task2v1': '/home/graduation project/code/EEG-Code/Data/pickle_file/task2-NR-dataset.pickle',
        'task2v2': '/home/graduation project/code/EEG-Code/Data/pickle_file/task2-NR v2.0-dataset.pickle',
    }
    for task, path in dataset_paths.items():
        if task in task_name:
            with open(path, 'rb') as handle:
                whole_dataset_dicts.append(pickle.load(handle))

    tokenizer = (
        BartTokenizer.from_pretrained('facebook/bart-large') if model_name in ['BrainTranslator', 'BrainTranslatorNaive', 'R1Translator'] else
        T5Tokenizer.from_pretrained('t5-large') if model_name == 'T5Translator' else
        BertTokenizer.from_pretrained('bert-base-uncased') if model_name == 'BrainBERT' else None
    )
    if tokenizer is None:
        raise ValueError(f'Unsupported model: {model_name}')

    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print(f'[INFO] Train set size: {len(train_set)}, Dev set size: {len(dev_set)}')

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=4)
    dataloaders = {'train': train_dataloader, 'dev': val_dataloader}

    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration(BartConfig.from_pretrained('facebook/bart-large')) if use_random_init else BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslator(pretrained, in_feature=105*len(bands_choice), decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048)
    elif model_name == 'BrainTranslatorNaive':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslatorNaive(pretrained, in_feature=105*len(bands_choice), decoder_embedding_size=1024)
    elif model_name == 'T5Translator':
        pretrained = T5ForConditionalGeneration.from_pretrained('t5-large')
        model = T5Translator(pretrained, in_feature=105*len(bands_choice), decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048)
    elif model_name == 'R1Translator':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = R1Translator(pretrained, in_feature=105*len(bands_choice), decoder_embedding_size=pretrained.config.d_model, rnn_hidden_size=256, num_rnn_layers=2)
    elif model_name == 'BrainBERT':
        bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        bart_decoder = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainBERT(bert_encoder, bart_decoder, in_feature=105*len(bands_choice))

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Step 1: Freeze most of pretrained model parameters
    if model_name in ['BrainTranslator', 'BrainTranslatorNaive', 'T5Translator', 'R1Translator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if any(key in name for key in ['shared', 'embed', 'layer.0']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    if not skip_step_one:
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)
        print('=== Starting Step 1 training ===')
        model = train_model(dataloaders, device, model, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last)
    elif load_step1_checkpoint:
        stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
        print(f'Skipping step 1, loading checkpoint: {stepone_checkpoint}')
        model.load_state_dict(torch.load(stepone_checkpoint))
    else:
        print('Skipping step 1, starting from scratch at step 2')

    # Step 2: Unfreeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = True

    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)
    print('=== Starting Step 2 training ===')
    trained_model = train_model(dataloaders, device, model, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last)