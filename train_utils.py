import numpy as np 
import torch
from torch import nn

def one_hot_encode(arr, n_labels):

    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''

    batch_size = n_seqs * n_steps
    n_batches = len(arr)//batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    np.random.shuffle(arr)
    for n in range(0, arr.shape[1], n_steps):

        # The features
        x = arr[:, n:n+n_steps]

        # The targets, shifted by one
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        
        
        
def train(net, data, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, cuda=False, print_every=30):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        n_seqs: Number of mini-sequences per mini-batch, aka batch size
        n_steps: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        cuda: Train with CUDA on a GPU
        print_every: Number of steps for printing training and validation loss

    '''

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        net.cuda()

    counter = 0
    n_chars = len(net.chars)

    save_path = '/content/drive/MyDrive/UCLA/Courses/NLP/CS 263 Final Project/saved_models/'
    file_name = 'charRNN_scratch.pkl'

    for e in range(epochs):

        h = net.init_hidden(n_seqs)

        for x, y in get_batches(data, n_seqs, n_steps):

            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            net.zero_grad()

            output, h = net.forward(inputs, h)

            loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.cuda.LongTensor))

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()

            if counter % print_every == 1:

                # Get validation loss
                val_h = net.init_hidden(n_seqs)
                val_losses = []

                for x, y in get_batches(val_data, n_seqs, n_steps):

                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.cuda.LongTensor))

                    val_losses.append(val_loss.item())

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
                from datetime import datetime
                save_path = 'saved_models/'
                file_name = 'charRNN_1layer_randdata_' + " {:.2f} ".format(loss.item()) + "{:.2f}".format(np.mean(val_losses)) + '.pkl'
                time = "{:%Y_%m_%d_%H_%M_}".format(datetime.now())
                torch.save( net, save_path+time + file_name)

                
# following functions will probably never be used again, since we will test our language models on real data

def letter_to_emg_sim(key, char_tuple, noise_dist= 1, typing_style='skilled'):
    """ Generate a simulated EMG decoder softmax given the correct key

    Arguments 
    ---------
    key: ground truth key 
    char_tuple: tuple of characters that represent the key that each element in the softmax corresponds to
    """
    keyboard = {0:'qwertyuiop',1:'asdfghjkl',2:'zxcvbnm',3:' '}

    int2char = dict(enumerate(char_tuple))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # find the location of key on physical keyboard
    for row in range(4):
        argmax_key_column = keyboard[row].rfind(key)
        if argmax_key_column != -1:
            argmax_key_row = row
            break
    # create some parameters for archetypal typists
    if typing_style == 'skilled':
        accuracy = .5
        softmax_range = 1 # keys
    if typing_style == 'unskilled':
        accuracy = .25
        softmax_range = 2 # keys

    # set the peak probability at the true key "accuracy"% of the time,
    # otherwise it is uniformly randomly assigned to a key less than "softmax_range" keys away
    if np.random.rand() > accuracy:
        r_shift = np.random.choice([ i  for i in range(-softmax_range,softmax_range+1) if i != 0])
        c_shift = np.random.choice([ i  for i in range(-softmax_range,softmax_range+1) if i != 0])
        while keyboard_index_is_lowercaseletter(argmax_key_row+r_shift,argmax_key_column+c_shift) is False:
            r_shift = np.random.choice([ i  for i in range(-softmax_range,softmax_range+1) if i != 0])
            c_shift = np.random.choice([ i  for i in range(-softmax_range,softmax_range+1) if i != 0])
        argmax_key_row = argmax_key_row + r_shift
        argmax_key_column = argmax_key_column + c_shift
    max_key = keyboard[argmax_key_row][argmax_key_column]

    p = np.zeros((len(char_tuple)))
    # space key has no errors
    if key == ' ':
    # make the space key correct 80% of the time.
        for char in ['c','v','b','n','m']:
            p[char2int[char]] = np.random.random()
        p[char2int[key]] = np.random.random()+.65 # 80% correct space bar
        return p/np.sum(p)

    # add noise to softmax for keys within "softmax_range" of the peak prob key
    for i in range(-softmax_range, softmax_range+1):
        for j in range(-softmax_range, softmax_range+1):
            if not keyboard_index_is_lowercaseletter(argmax_key_row+i, argmax_key_column+j):
                continue
            # add the noise to the element in p corresponding to the key
            noise_key = keyboard[argmax_key_row+i][argmax_key_column+j]
            distance = np.max([abs(i),abs(j)])
            noise = 2*np.random.random()-1
            p[char2int[noise_key]] = ((softmax_range-distance+1) + noise) /(softmax_range+1)

    p[char2int[max_key]] = 1
    return p/np.sum(p)

def keyboard_index_is_lowercaseletter(row_index, column_index):
    """ Return False if row and column index are out of bounds on a keyboard
    """
    # top, left, and bottom of keyboard cases
    if row_index < 0 or row_index > 2 or column_index < 0 :
        return False
    # right boundaries, manually defined for each row
    if row_index == 0 and column_index > 9:
        return False
    if row_index == 1 and column_index > 8:
        return False
    if row_index == 2 and column_index > 6:
        return False
    return True
