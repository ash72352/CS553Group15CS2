import torch
from torch import nn, cuda, load, device
from ps4_models.classifiers import PS4_Mega, PS4_Conv


def load_trained_model(load_path, model_name='PS4_Mega'):

    if model_name.lower() not in ['ps4_conv', 'ps4_mega']:
        raise ValueError(f'Model name {model_name} not recognised, please choose from PS4_Conv, PS4_Mega')

    model: nn.Module = PS4_Mega() if model_name.lower() == 'ps4_mega' else PS4_Conv()

    if load_path != '':
        try:
            if cuda.is_available():
                model.load_state_dict(load(load_path)['model_state_dict'])
            else:
                model.load_state_dict(load(load_path, map_location=device('cpu'))['model_state_dict'])
            print("loded params from", load_path)
        except:
            raise ImportError(f'No file located at {load_path}, could not load parameters')
    print(model)

    pytorch_total_params = sum(par.numel() for par in model.parameters() if par.requires_grad)
    print(pytorch_total_params)

    return model


# MARK: sampling from new sequence
def sample_new_sequence(embs, weights_load_path, model_name='PS4_Mega'):

    model = load_trained_model(weights_load_path, model_name)

    seq_size = len(embs)
    R = embs.view(1, seq_size, -1)

    pred_ss = ''

    with torch.no_grad():
        y_hat = model(R)
        probs = torch.softmax(y_hat, 2)
        _, ss_preds = torch.max(probs, 2)

        for i in range(seq_size):
            ss = ss_preds[0][i].item()
            ss = ss_tokeniser(ss, reverse=True)
            pred_ss += ss

    return pred_ss


def ss_tokeniser(ss, reverse=False):

    ss_set = ['C', 'T', 'G', 'H', 'S', 'B', 'I', 'E', 'C']

    if reverse:
        return inverse_ss_tokeniser(ss)
    else:
        return 0 if (ss == 'P' or ss == ' ') else ss_set.index(ss)


def inverse_ss_tokeniser(ss):

    ss_set = ['C', 'T', 'G', 'H', 'S', 'B', 'I', 'E', 'C', 'C']

    return ss_set[ss]