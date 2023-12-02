def gt_label():
    chars = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
    chars = chars.split(' ')

    words = ['bin', 'lay', 'place', 'set', 'blue', 'green', 'red', 'white', 'at', 'by', 'in', 'with', 'a', 'an', 'the', 'no', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'again', 'now', 'please', 'soon']

    words.extend(chars)
    return words

def word2idx(word):
    gt_words = gt_label()
    return gt_words.index(word)

def ctc_idx2word(idx,start = 1):
    gt_words = gt_label()
    output_text = []
    prev = -1
    for i in idx:
        if i != prev and i != start:
            output_text.append(gt_words[i-start])
        prev = i
    
    return ' '.join(output_text).strip()

def idx2word(idx,start = 1):
    gt_words = gt_label()
    output_text = []
    for i in idx:
        if i >= start:
            output_text.append(gt_words[i])
    return ' '.join(output_text).strip()


