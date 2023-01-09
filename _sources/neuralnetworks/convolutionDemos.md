# Animations of Convolution and Deconvolution


The concepts convolution, deconvolution (=transposed convolution), strides and padding have been introduced in the [previous section](03ConvolutionNeuralNetworks.ipynb). Below, these concepts are demonstrated. The animations are from {cite}`dumoulin2016guide`. In the demos only a single channel is at the input and only a single feature map is calculated. In a convolution- or deconvolution-layer typically many feature maps are calculated from many channels at the input.



## Convolution

::::{grid} 2
:::{grid-item-card} 
padding = 0, stride = 1
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/no_padding_no_strides.gif" style="width:200px" align="center">
</figure>
:::
:::{grid-item-card} 
padding = 1, stride = 1
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/same_padding_no_strides.gif" style="width:200px" align="center">
</figure>
:::
:::{grid-item-card}
padding = 0, stride = 2
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/no_padding_strides.gif" style="width:200px" align="center">
</figure>
:::
:::{grid-item-card}
padding = 1, stride = 2
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/padding_strides.gif" style="width:200px" align="center">
</figure>
:::
::::



## Deconvolution

::::{grid} 2
:::{grid-item-card} 

padding = 0, stride = 1, transposed
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/no_padding_no_strides_transposed.gif" style="width:200px" align="center">
</figure>
:::
:::{grid-item-card} 
padding = 1, stride = 1, transposed
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/same_padding_no_strides_transposed.gif" style="width:200px" align="center">
</figure>
:::
:::{grid-item-card} 
padding = 0, stride = 2, transposed
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/no_padding_strides_transposed.gif" style="width:200px" align="center">
</figure>
:::
:::{grid-item-card} 
padding = 1, stride = 2, transposed
^^^
<figure align="center">
<img src="https://maucher.home.hdm-stuttgart.de/Pics/gif/padding_strides_transposed.gif" style="width:200px" align="center">
</figure>
:::
::::

