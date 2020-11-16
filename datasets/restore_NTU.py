
import pickle
import numpy as np

with open('/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU60/NTU60/xsub/train_label.pkl', 'rb') as f:
    _, labels = pickle.load(f, encoding='latin1')

data = np.load('/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU60/NTU60/xsub/train_data.npy')
data = data.transpose([0,4,2,3,1])


def transform_data(data):
    N_sequences, n_people, max_length, n_joints, coords = data.shape
    
    nulls = (np.linalg.norm(data, axis=(3,4))>0)*1
    seq_len = nulls.sum(axis=2)

    # sequences with one person
    one_person = (seq_len[:,1] == 0)*1

    seq_len_max = seq_len.max(1) 

    all_sequences = []
    indices = []

    idx_count = 0

    for i in range(N_sequences):
        if one_person[i]==1:
            all_sequences.append(data[i,0,:seq_len_max[i]])
            indices.append(idx_count)
            idx_count += seq_len_max[i]
        else:
            all_sequences.append(data[i,0,:seq_len_max[i]])
            all_sequences.append(data[i,1,:seq_len_max[i]])
            
            indices.append(idx_count)
            idx_count += seq_len_max[i]*2

    all_sequences = np.concatenate(all_sequences, 0)

    data_dict = {
        'indices': indices,
        'seq_len': seq_len_max,
        'one_person': one_person,
        'labels': labels,
    }

    return all_sequences, data_dict

all_sequences, data_dict = transform_data(data)

np.save('NTU_train_seqs.npy', all_sequences)
np.save('NTU_train_labels.npy', data_dict)