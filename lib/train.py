from lib.utils import tensorsFromPair 
import torch
from torch import nn
import torch.nn.functional as F
import random


MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5

def train_batch(input_tensor, 
          target_tensor, 
          encoder, decoder, 
          encoder_optimizer, 
          decoder_optimizer, 
          criterion, device,
          max_length=MAX_LENGTH):
    
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def train(encoder, decoder, 
          n_iters,
          encoder_optimizer,
          decoder_optimizer,
          train_pairs,
          device,
          source,
          target,
          print_every=10):
#     plot_losses = []
    print_loss_total = 0  
#     plot_loss_total = 0 

    training_pairs = [tensorsFromPair(source, target, random.choice(train_pairs), device)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    train_loss = []
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_batch(input_tensor, target_tensor, encoder, 
                           decoder, encoder_optimizer, 
                           decoder_optimizer, criterion, device)
        print_loss_total += loss
#         plot_loss_total += loss
        train_loss.append(loss)
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(iter, print_loss_avg)
            
    return train_loss