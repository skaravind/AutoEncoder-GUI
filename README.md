# AutoEncoder-GUI
A GUI (Python3) for manually changing latent dimensions of an Adversarial AutoEncoder. Its made using Tkinter

The model included with the code is the Decoder of an AAE trained on Ganesha paintings (or should i say GANesha ;P)

## Running Script

In terminal: <code> $ python3 gui.py </code>

## Model

There are 100 latent dimensions, so I have added 100 scrollers for changing each dimension. It starts with mean=0 std=1, but you can set individual scroller value upto 5.0/-5.0.
