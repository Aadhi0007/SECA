"""
Model Builder for SECA

This module converts a genome representation into a neural network model.
The architecture is dynamically created based on genome parameters.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models





def build_model(genome, input_shape, num_classes):
    """
    Build a neural network model from a genome.
    If genome["network_type"] == "llm", build a Transformer model.
    Else, build a Conv2D model.
    """
    net_type = genome.get("network_type", "cnn")

    if net_type in ["llm", "gpt", "bert", "llama"]:
        # Build Transformer (LLM) model
        # For simplicity, we assume input_shape is (seq_len,) of integer tokens
        inputs = layers.Input(shape=input_shape)
        
        # We need a vocabulary size. Let's assume num_classes + some buffer, or max from data.
        # Since this is a demo, giving it a fixed vocab of 10000.
        vocab_size = 10000 
        embed_dim = genome["embed_dim"]
        
        x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
        
        for block in genome["blocks"]:
            # Multi-Head Attention
            attn_output = layers.MultiHeadAttention(
                num_heads=block["num_heads"], 
                key_dim=embed_dim
            )(x, x)
            # Add & Norm
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed Forward
            ff_output = layers.Dense(block["ff_dim"], activation="relu")(x)
            ff_output = layers.Dense(embed_dim)(ff_output)
            
            # Add & Norm
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
        x = layers.GlobalAveragePooling1D()(x)
        
        if genome["dropout"] > 0:
            x = layers.Dropout(genome["dropout"])(x)
            
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
    else:
        # Build CNN model
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))

        is_1d = (len(input_shape) == 1)
        if is_1d:
            # Reshape (steps,) to (steps, channels=1) for 1D convolution
            model.add(layers.Reshape((input_shape[0], 1)))
            current_size_x = input_shape[0]
            current_size_y = input_shape[0]
        else:
            current_size_x = input_shape[0]
            current_size_y = input_shape[1]

        for stage in genome["stages"]:
            filters = stage["filters"]
            kernel = stage["kernel"]
            pool = stage["pool"]

            if is_1d:
                model.add(
                    layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel,
                        activation="relu",
                        padding="same"
                    )
                )
            else:
                model.add(
                    layers.Conv2D(
                        filters=filters,
                        kernel_size=(kernel, kernel),
                        activation="relu",
                        padding="same"
                    )
                )
                
            model.add(layers.BatchNormalization())

            if pool == 2:
                if is_1d:
                    if current_size_x >= 2:
                        model.add(layers.MaxPooling1D(2))
                        current_size_x //= 2
                else:
                    if current_size_x >= 2 and current_size_y >= 2:
                        model.add(layers.MaxPooling2D((2, 2)))
                        current_size_x //= 2
                        current_size_y //= 2

        model.add(layers.Flatten())
        model.add(layers.Dense(genome["dense"], activation="relu"))

        if genome["dropout"] > 0:
            model.add(layers.Dropout(genome["dropout"]))

        model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model