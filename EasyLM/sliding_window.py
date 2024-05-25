# from more_itertools import windowed, repeat_last
import numpy as np

from more_itertools import windowed


def sliding_window(tok_input_ids, width, stride,
                   padding_value=-100,
                   bos_token_id=-2,
                   append_eos=False):
    """
    Generates sliding window elements from input token IDs.

    Args:
        tok_input_ids: List of input token IDs.
        width: Width of the sliding window.
        stride: Stride of the sliding window.
        padding_value: Value to use for padding.
        bos_token_id: ID of the beginning-of-sentence token.

    Returns:
        List of sliding window elements, where each element is a list 
        containing:
            - input_tokens: The token IDs within the window.
            - attention_mask: Attention mask for the window.
            - targets: The target token IDs for the window.
            - loss_mask: Mask indicating which tokens contribute to loss.
    """
    if append_eos:
        tok_input_ids = np.r_[tok_input_ids, bos_token_id]
    start = 0
    in_start = 0
    end = width
    while True:
        
        assert end-width>=0, "end-width must be greater than or equal to 0"
        curr_tokens = tok_input_ids[:end]
        input_tokens = np.r_[bos_token_id, curr_tokens[:-1][in_start:]]
        targets = curr_tokens[start:]
        loss_mask = np.ones_like(targets).astype(bool)
        attention_mask = np.ones_like(input_tokens).astype(bool)
        
        yield dict(input_tokens=np.pad(input_tokens,(width-input_tokens.shape[0],0),
                                       constant_values=padding_value),
                    targets=np.pad(targets,(width-targets.shape[0],0),
                                   constant_values=padding_value),
                    attention_mask=np.pad(attention_mask,(width-attention_mask.shape[0],0),
                                          constant_values=False),
                    loss_mask=np.pad(loss_mask,(width-loss_mask.shape[0],0),
                                     constant_values=False),
                    )

        end += stride
        start = max(end-stride,0)
        in_start = max(end-width,0)
        if end>len(tok_input_ids):
            break

    input_tokens = np.r_[bos_token_id, tok_input_ids[-width:][:-1]]
    targets = tok_input_ids[:end][start:]
    if len(targets)==0:
        return
    loss_mask = np.ones_like(targets).astype(bool)
    attention_mask = np.ones_like(input_tokens).astype(bool)
    
    yield dict(input_tokens=np.pad(input_tokens,(width-input_tokens.shape[0],0),
                                       constant_values=padding_value),
                    targets=np.pad(targets,(width-targets.shape[0],0),
                                   constant_values=padding_value),
                    attention_mask=np.pad(attention_mask,(width-attention_mask.shape[0],0),
                                          constant_values=False),
                    loss_mask=np.pad(loss_mask,(width-loss_mask.shape[0],0),
                                     constant_values=False),
                    )
        


def padded_sliding_window(tok_input_ids, width, stride, chunk_length,
                          padding_value = -100, bos_token_id = -2):
    tok_input_ids = np.array(tok_input_ids, dtype=np.int32)
    addition = (tok_input_ids.shape[0]%(2*chunk_length))
    tok_input_ids = np.pad(tok_input_ids,(2*chunk_length-addition,0),constant_values=padding_value)
    tok_input_ids = np.r_[tok_input_ids, bos_token_id]
    yield from sliding_window(tok_input_ids, width, stride, padding_value, bos_token_id)
     
    
if __name__=="__main__":

    input_ids = np.arange(65)
    # input_ids = np.arange(4097)
    width = 16
    stride = 4

    for element in sliding_window(np.arange(32), width=8, stride=4):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        print(targets[loss_mask])
        print(element)
    print("@"*100)
    for element in sliding_window(np.arange(32), width=8, stride=0):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        print(targets[loss_mask])
        print(element)
        
    print("@"*100)
    seq = np.arange(5)

    for element in sliding_window(seq, width=8, stride=0):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        print(element)
        print(targets[loss_mask])
    print("@"*100)
    seq = np.arange(9)

    for element in sliding_window(seq, width=8, stride=0):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        print(element)
        print(targets[loss_mask])
    print("@"*100)
    seq = np.arange(17)

    for element in sliding_window(seq, width=8, stride=0):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        print(element)
        print(targets[loss_mask])



    list(windowed(seq, n=width, step=stride,fillvalue=-100))
    print()
    print("@"*50+"padded_sliding_window"+"@"*50)
    seq = np.arange(17)

    for element in padded_sliding_window(seq, width=8, stride=0, chunk_length=2):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        print(element)
        print(targets[loss_mask])
    list(windowed(seq, n=width, step=stride,fillvalue=-100))
    print()
    print("@"*50+"padded_sliding_window"+"@"*50)
    seq = np.arange(19)

    for element in padded_sliding_window(seq, width=8, stride=6, chunk_length=2):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        print(element)
        print(targets[loss_mask])




    print("@"*50+"padded_sliding_window"+"@"*50)
    seq = np.arange(59)

    for element in padded_sliding_window(seq, width=16, stride=8, chunk_length=4):
        input_tokens = element["input_tokens"]
        targets = element["targets"]
        attention_mask = element["attention_mask"]
        loss_mask = element["loss_mask"]
        chunk_mask = loss_mask.reshape(-1,4).sum(axis=1).astype(bool)
        print(element)
        print(input_tokens.reshape(-1,4)[chunk_mask])




