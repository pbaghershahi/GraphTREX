
def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.
    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def texts_from_locs(data, token_starts, token_ends):
    mention_texts = []
    token2startchar = data.tokenization['token2startchar']
    token2endchar = data.tokenization['token2endchar']
    for s, e in zip(token_starts, token_ends):
        start_char, end_char = token2startchar[int(s)], token2endchar[int(e)]
        text = data.text[start_char: end_char]
        mention_texts.append(text)
    return mention_texts
