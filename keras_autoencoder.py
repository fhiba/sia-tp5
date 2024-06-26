import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tf_keras.layers import Input, Dense
from tf_keras.models import Model
from tf_keras.activations import elu, hard_sigmoid
from tf_keras.losses import mean_squared_error

from tensorflow.python.framework.ops import disable_eager_execution

from data.font import Font3

disable_eager_execution()

b = [[
    float(bit)
    for byte in font
    for bit in bin(byte)[2:].zfill(5)
  ]
  for font in Font3
]

input_dim = len(b[0])

input_enc = Input(input_dim)
enc = Dense(35, activation=elu)(input_enc)
enc = Dense(28, activation=elu)(enc)
enc = Dense(22, activation=elu)(enc)
enc = Dense(17, activation=elu)(enc)
enc = Dense(10, activation=elu)(enc)

latent = Dense(3, activation=elu)(enc)

dec = Dense(10, activation=elu)(latent)
dec = Dense(17, activation=elu)(dec)
dec = Dense(22, activation=elu)(dec)
dec = Dense(28, activation=elu)(dec)
output_dec = Dense(input_dim, activation=hard_sigmoid)(dec)

autoencoder = Model(input_enc, output_dec)
autoencoder.summary()
autoencoder.compile(optimizer="adam", loss=mean_squared_error)

b = np.reshape(b, (len(b), np.prod(input_dim))).astype(np.float32)

autoencoder.fit(b, b,
        shuffle=False,
        epochs=2000,
        batch_size=32)

axes = plt.subplots(4, 8)[1]
for i, font in enumerate(autoencoder.predict(b)):
    axes[int(i/8), i%8].imshow(font.reshape((7,5)), cmap=matplotlib.colormaps['plasma'])
  
plt.show()
