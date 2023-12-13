import numpy as np
import string
import matplotlib.pyplot as plt

# Letters contain the tokens, except for the blank token
LETTERS = [' '] + list(string.ascii_lowercase)

def gt_label():
    chars = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
    chars = chars.split(' ')

    words = ['bin', 'lay', 'place', 'set', 'blue', 'green', 'red', 'white', 'at', 'by', 'in', 'with', 'a', 'an', 'the', 'no', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'again', 'now', 'please', 'soon']

    words.extend(chars)
    return words


def text2idx(text, start=1):
        """
        Compute the corresponding array given a text
        """
        array = []
        for character in list(text):
            array.append(LETTERS.index(character) + start)
        return np.array(array)


def idx2text(array, start=1):
        """
        Compute the truth text given a truth array of indexes
        """
        text = []
        # Parse the array
        for idx in array:
            # Append only if not blank token (start = 1)
            if(idx >= start):
                text.append(LETTERS[idx - start])     
        return ''.join(text).strip()

 
def ctc_idx2text(array_index_letters, start=1):
        """
        Compute the text prediction given a ctc output
        """
        # Previous letter
        previous_letter = -1
        outputed_text = []
        for n in array_index_letters:
            # Not same letters successively, and if n is lower than start it does not correspond to anything
            if(previous_letter != n and n >= start):
                # Append the corresponding letter
                outputed_text.append(LETTERS[n - start]) 
            # Update previous letter              
            previous_letter = n
        return ''.join(outputed_text).strip()

def ctc_decoder(output):
    output = output.argmax(-1) # (B=64, T=75, Emb=28) -> (B=64, T=75)
    # print(output)
    length = output.size(0)
    text_list = []
    for _ in range(length):
        text_list.append(ctc_idx2text(output[_]))
    return text_list

def plot_error_curves_comparison(output, mode='test'):
    iters = len(output['gru']['train_wer'])
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    for i, k in enumerate(output):
        plt.plot(range(iters), output[k]['train_wer'], label=k)
    plt.title('{} Word Error Rate (WER)'.format(mode))
    plt.xlabel('Iteration')
    plt.ylabel('WER')
    plt.legend()    

    plt.subplot(122)
    for i, k in enumerate(output):
        plt.plot(range(iters), output[k]['train_cer'], label=k)
    plt.title('{} Character Error Rate (CER)'.format(mode))
    plt.xlabel('Iteration')
    plt.ylabel('CER')
    plt.legend()
    plt.savefig('./log/{}_error_curve.png'.format(mode))