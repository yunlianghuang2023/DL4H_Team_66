from transformers import AdamW

def adam(params, config=None):
    if config is None:
        config = {
            'lr': 3e-5,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': config['weight_decay']},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optim = AdamW(optimizer_grouped_parameters, lr=config['lr'], betas=(0.9, 0.999), eps=1e-8, correct_bias=True)
    return optim
