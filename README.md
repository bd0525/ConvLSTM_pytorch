# ConvLSTM_pytorch

The original implementation was made by [Andrea Palazzi](https://github.com/ndrplz) and [Davide Abati](https://github.com/DavideA), , based on [this](https://github.com/rogertrullo/pytorch_convlstm/blob/master/conv_lstm.py) implementation, which they heavily refactored and extended with additional features to meet their needs.

Please note that this repository implements the following dynamics:
![CLSTM_dynamics](https://user-images.githubusercontent.com/7113894/59357391-15c73e00-8d2b-11e9-8234-9d51a90be5dc.png)

which differ slightly from those in the original [paper](https://arxiv.org/pdf/1506.04214.pdf).

### How to Use
The ConvLSTM class supports an arbitrary number of layers. In this case, it can be specified the hidden dimension (that is, the number of channels) and the kernel size of each layer. In the case more layers are present but a single value is provided, this is replicated for all the layers. For example, in the following snippet each of the three layers has a different hidden dimension but the same kernel size.

Example usage:
```
model = ConvLSTM(input_dim=channels,
                 hidden_dim=[64, 64, 128],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
```

### TODO (in progress...)
- Add docs
- Add example usage on a toy problem (need test)
- Implement stateful mechanism (need test)
- ...