import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
import os
import json
from tqdm import tqdm

# Function to extract blocks from a single JSON file
def extract_blocks_from_json(file_path):
    print("\t", file_path)
    blocks = []
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        nodes = data.get("Nodes", [])
        for node in nodes:
            features_encoding = node.get("Features_encoding", [])
            blocks.append(features_encoding)
    return blocks

# Recursive function to search for JSON files and get block data
def get_block_data_from_JSON(dir_path):
    all_blocks = []
    for root, _, files in os.walk(dir_path):
        print(root)
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                blocks = extract_blocks_from_json(file_path)
                all_blocks.extend(blocks)
    return all_blocks

# Function to convert blocks data to numpy array and return padded sequences
def generate_data_from_blocks(blocks, num_samples, expected_length=128, n_instructions=512):
    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(blocks), desc="Processing Blocks", unit="block")
    progress_bar_inst = tqdm(total=len(blocks), desc="Processing Blocks Instructions", unit="block")

    # Check if the length of 'blocks' is less than n_instructions, and pad if necessary
    for i in range(len(blocks)):
        if len(blocks[i]) < n_instructions:
            padding_needed = n_instructions - len(blocks[i])
            for _ in range(padding_needed):
                blocks[i].append([0] * expected_length)  # Pad with zero blocks
        elif len(blocks[i]) > n_instructions:
            # Reduce the elements to n_instructions by truncating
            blocks[i] = blocks[i][:n_instructions]

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Process the blocks as before
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            if len(blocks[i][j]) < expected_length:
                # Pad the block with zeros to the expected length
                padding = expected_length - len(blocks[i][j])
                blocks[i][j].extend([0] * padding)  # Add zeros to the end
            elif len(blocks[i][j]) > expected_length:
                # If the block is longer than the expected length, truncate it
                blocks[i][j] = blocks[i][j][:expected_length]
        progress_bar_inst.update(1)
    blocks_arr = np.array(blocks)

    # Extend 'blocks_arr' to have 'num_samples' rows
    while blocks_arr.shape[0] < num_samples:
        blocks_arr = np.concatenate((blocks_arr, blocks), axis=0)

    # Trim 'blocks_arr' to have 'num_samples' rows
    blocks_arr = blocks_arr[:num_samples]

    X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(blocks_arr, padding='post', dtype='float32')
    Y_train = np.random.rand(num_samples, 64)

    return X_train_padded, Y_train


def attention_mechanism(inputs):
    # Step 1: Calculate the attention scores using a dense layer
    attention_probs = layers.Dense(1, activation='tanh')(inputs)

    # Step 2: Use a softmax activation to convert these scores to weights
    attention_probs = layers.Flatten()(attention_probs)
    attention_probs = layers.Activation('softmax')(attention_probs)
    attention_probs = layers.RepeatVector(inputs.shape[-1])(attention_probs)
    attention_probs = layers.Permute([2, 1])(attention_probs)

    # Step 3: Use a weighted sum of the original sequence based on these weights
    attended_sequence = layers.Multiply()([inputs, attention_probs])

    return attended_sequence

def create_conv1d_encoder(input_shape=(None, 128)):
    model_input = layers.Input(shape=input_shape)
    conv_layer = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(model_input)

    # Apply attention mechanism after the convolution layer
    attended_sequence = attention_mechanism(conv_layer)

    global_avg_pooling = layers.GlobalAveragePooling1D()(attended_sequence)
    dense_layer = layers.Dense(64, activation='relu')(global_avg_pooling)

    return model_input, dense_layer


def create_decoder(output_length):
    # Decoder input shape should match encoder's output shape
    decoder_input = layers.Input(shape=(64,))

    # Begin decoding
    x = layers.Dense(output_length * 64, activation='relu')(decoder_input)
    x = layers.Reshape((output_length, 64))(x)
    x = layers.Conv1DTranspose(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)

    return decoder_input, x

def create_autoencoder(input_length):
    # Create encoder
    encoder_input, encoder_output = create_conv1d_encoder()
    encoder = Model(inputs=encoder_input, outputs=encoder_output)

    # Create decoder
    decoder_input, decoder_output = create_decoder(input_length)
    decoder = Model(inputs=decoder_input, outputs=decoder_output)

    # Connect encoder and decoder
    autoencoder_input = layers.Input(shape=(input_length, 128))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)

    autoencoder = Model(inputs=autoencoder_input, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def encode_blocks(blocks, encoder):
    embeddings = []
    for block in blocks:
        block_embedding = encoder.predict(np.expand_dims(block, axis=0))
        embeddings.append(block_embedding)
    return embeddings

def plot_learning_graph(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def main():
    # Directory path to search for JSON files
    directory_path = 'C:\\Users\Saqib\PycharmProjects\GAGE\\output\CEG_data_small\\subfolder'
    #directory_path = 'C:\\Users\Saqib\PycharmProjects\GAGE\\output\CEG_data_try'

    # Get blocks from JSON files
    blocks = get_block_data_from_JSON(directory_path)
    num_samples = 100
    input_length = 512
    X_train, _ = generate_data_from_blocks(blocks,
                                           num_samples)  # Ignore Y_train since autoencoder reconstructs its input
    print(X_train.shape)

    # Use the dynamic length for creating the autoencoder

    #autoencoder = create_autoencoder(input_length)
    autoencoder = load_model('models/AED_autoencoder_202310030112_Ori_v3.h5')

    # Train the autoencoder
    history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    autoencoder.save('models/AED_autoencoder_202310030112_Ori_v3.h5')

    # Plot the learning graph
    plot_learning_graph(history)

    # call the model
    loaded_model = load_model('models/AED_autoencoder_202310030112_Ori_v3.h5')

    # Create a sample block and pad it to match the model's expected input length
    sample_block = np.random.rand(20, 128)
    sample_block_padded = tf.keras.preprocessing.sequence.pad_sequences([sample_block], maxlen=input_length, padding='post', dtype='float32')[
        0]
    # Predict using the loaded model
    reconstructed_block = loaded_model.predict(np.expand_dims(sample_block_padded, axis=0))

    # Print the reconstructed block for the sample block
    print("Reconstructed block for the sample block:\n", reconstructed_block)


# Run the main function
main()

# 11022
# 33805
# 40439
# 45188
# 53197
# 45188
# 29
# 3819
# 11384
# 10695
# 13421
# 5310
# 799
# 1310
# 47833
# 32201
# 16065
# 23813
# 20375
# 13815
# 7323
# 17894
# 19226
# 19180
# 20131
# 11795
# 33144



# 493331 total blocks trained AED

# (False, False, False, False) 0
# (False, False, False, True) 1
# (False, False, True, False) 2
# (False, False, True, True) 3
# (False, True, False, False) 4
# (False, True, False, True) 5
# (False, True, True, False) 6
# (False, True, True, True) 7
# (True, False, False, False) 8
# (True, False, False, True) 9
# (True, False, True, False) 10
# (True, False, True, True) 11
# (True, True, False, False) 12
# (True, True, False, True) 13
# (True, True, True, False) 14
# (True, True, True, True) 15