from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np
import time
import os


def generate_embedings(input_seq, output_path=None):

    # Create directories
    protT5_path = "ps4_data/data/protT5"
    # where to store the embeddings
    per_residue_path = "ps4_data/data/protT5/output/per_residue_embeddings" if output_path is None else output_path
    for dir_path in [protT5_path, per_residue_path]:
        __create_dir(dir_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))

    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    model, tokenizer = __get_T5_model(device)

    # Load fasta.
    all_seqs = {"0": input_seq}
    # Compute embeddings and/or secondary structure predictions
    results = __get_embeddings(model, tokenizer, all_seqs, device)

    return results


def __get_T5_model(device):

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


def __save_embeddings(emb_dict,out_path):
    np.savez_compressed(out_path, **emb_dict)


def __get_embeddings(model, tokenizer, seqs, device, per_residue=True,
                   max_residues=4000, max_seq_len=1000, max_batch=100):

    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_residue:  # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().squeeze()
                    print("emb_count:", len(results["residue_embs"]))

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time / 60, avg_time))
    print('\n############# END #############')
    return results


def __create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)