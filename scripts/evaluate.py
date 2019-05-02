import argparse
import os
import pickle
import sys

sys.path.append(".")

from torch.utils.data import DataLoader

from src.model import *
from src.sampler import *
from src.dataset import MidiDataset
from src.transform import Transform

from src.reconstruction import MidiBuilder

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='X_test.pickle')
parser.add_argument('--model_type', type=str, default='lstm')
parser.add_argument('--mode', type=str, default=None, choices=['eval', 'interpolate', 'reconstruct'])
parser.add_argument('--song_id_a', type=str, default=None)
parser.add_argument('--song_id_b', type=str, default=None)
parser.add_argument('--song_names', type=str, default='data/test_paths.pickle')
parser.add_argument('--temp_path', type=str, default=None)
parser.add_argument('--reconstruction_path', type=str, default='midi_reconstruction')
parser.add_argument('--model_path', type=str)

# Data settings
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--tempo_file', type=str, default='data/T_test.pickle')

# Model parameters
parser.add_argument('--num_subsequences', type=int, default=16)
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--sequence_length', type=int, default=16)
parser.add_argument('--encoder_input_size', type=int, default=61)
parser.add_argument('--decoder_input_size', type=int, default=61)
parser.add_argument('--encoder_hidden_size', type=int, default=2048)
parser.add_argument('--decoder_hidden_size', type=int, default=1024)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--encoder_num_layers', type=int, default=2)
parser.add_argument('--decoder_num_layers', type=int, default=2)

# Trainer parameters
parser.add_argument('--KL_rate', type=float, default=0.9999)
parser.add_argument('--free_bits', type=int, default=256)
parser.add_argument('--sampling_rate', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--output_dir', type=str, default='outputs')


def load_model(model_type, params):
    if model_type == 'lstm':
        model = MusicLSTMVAE(**params)
    elif model_type == 'gru':
        model = MusicGRUVAE(**params)
    else:
        raise Exception("Invalid model type. Expected lstm or gru")
    return model


def load_data(test_data, batch_size, song_paths=None):
    X_test = pickle.load(open(test_data, 'rb'))
    song_names = [os.path.basename(x) for x in pickle.load(open(song_paths, 'rb'))]
    test_data = MidiDataset(X_test, Transform(1), song_paths=song_names)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return test_loader

def load_tempo(tempo_path, song_id):
    if temp_path is None:
        raise ValueError('Tempo file unspecified')
    else:
        tempos = pickle.load(open(tempo_path, 'rb'))
        return tempos[song_id]
    
def evaluate(sampler, model, args):
    data_path = os.path.join(args.data_dir, args.data)
    data = load_data(data_path, args.batch_size, args.song_names)
    loss_tf, loss = sampler.evaluate(model, data)
    print("Loss with teacher forcing: %.4f, loss without teacher forcing: %.4f" % (loss_tf, loss))
    
def reconstruct(sampler, model, args):
    data_path = os.path.join(args.data_dir, args.data)
    builder = MidiBuilder()
    song_id = args.song_id_a
    data = load_data(data_path, args.batch_size, args.song_names)
    song = data.dataset.get_tensor_by_name(song_id)
    # Generate reconstruction from the samples
    reconstructed = sampler.reconstruct(model, song)
    # Get tempo
    tempos = pickle.load(open(args.tempo_file, 'rb'))
    tempo = data.dataset.song_to_idx[song_id]
    # Reconstruct into midi form
    midi = builder.midi_from_piano_roll(reconstructed) #TODO should include tempo
    reconstruction_dir = args.reconstruction_path
    if not os.path.exists(reconstruction_dir):
        os.makedirs(reconstruction_dir)
    path = os.path.join(reconstruction_dir, song_id)
    midi.write(path + ".mid")
    print('Saved reconstruction for %s' % song_id)
    
def interpolate():
    raise ValueError("Not implemented")

def main(args):
    model_params = {
        'num_subsequences': args.num_subsequences,
        'max_sequence_length': args.max_sequence_length,
        'sequence_length': args.sequence_length,
        'encoder_input_size': args.encoder_input_size,
        'decoder_input_size': args.decoder_input_size,
        'encoder_hidden_size': args.encoder_hidden_size,
        'decoder_hidden_size': args.decoder_hidden_size,
        'latent_dim': args.latent_dim,
        'encoder_num_layers': args.encoder_num_layers,
        'decoder_num_layers': args.decoder_num_layers
    }
    model = load_model(args.model_type, model_params)

    sampler_params = {
        'free_bits': args.free_bits,
        'sampling_rate': args.sampling_rate,
        'batch_size': args.batch_size,
        'output_dir': args.output_dir
    }
    sampler = Sampler(**sampler_params)
    
    model.load_state_dict(torch.load(args.model_path, map_location='cpu').state_dict(), strict=False)
    print(model)
    model.eval()
            
    mode = args.mode
    if mode == 'eval':
        evaluate(sampler, model, args)
    elif mode == 'reconstruct':
        reconstruct(sampler, model, args)
        
#     elif mode == 'interpolate':
#         song_id_A = args.song_id_a
#         song_id_B = args.song_id_b
#         data = load_data(data_path, args.batch_size)
#         song_a = data.get_tensor_by_name(song_id_a)
#         song_b = data.get_tensor_by_name(song_id_b)
#         interpolated = sampler.interpolate(model, song_a, song_b)
#         # TODO save interpolated
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
