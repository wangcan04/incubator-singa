This example trains the [MLP model](http://arxiv.org/abs/1003.0358) over the
MNIST dataset. The purpose of this example is to show uses how to implement
and use their own layers in SINGA.
Note: The *mnist/job.conf* trains the same MLP model, but only uses SINGA built-in
layers.

## File description

The files in this folder include:

* myproto, definition of the configuration protocol of the `HiddenLayer`.
* hidden_layer.h, declaration of the `HiddenLayer`.
* hidden_layer.cc, definition of the `HiddenLayer`.
* main.cc, main function that register the `HiddenLayer`.
* Makefile.exmaple, Makefile for compiling all source code in this folder.
* deep.conf, the job configuration for training the MLP model.
* job.conf, similar to deep.conf except that only one hidden layer is used.


## Data preparation

To use the MNIST dataset, we can reuse the preparation code in *mnist/* folder.

    # inside mnist/ folder
    cp Makefile.example Makefile
    make download
    make create

## Compilation

The *Makefile.example* contains instructions for compiling the source code.

    # in mlp/ folder
    cp Makefile.example Makefile
    make

It will generate an executable file *mlp.bin*.

## Running

There are two job configuration files,

1. *job.conf* for a shallow model with only one hidden layer.
2. *deep.conf* for original MLP model with 6 hidden layers.

Before running the following scripts, we need to export the `LD_LIBRARY_PATH` to
include the libsinga.so

    # at the root folder of SINGA
    export LD_LIBRARY_PATH=.libs:$LD_LIBRARY_PATH

### Shallow model

    # at the root folder of SINGA
    ./bin/singa-run.sh -exec examples/mlp/mlp.bin -conf examples/mlp/job.conf

You will notice that the training is fast since the model is simple (only one
hidden layer).


### Deep model

    # at the root folder of SINGA
    ./bin/singa-run.sh -exec examples/mlp/mlp.bin -conf examples/mlp/deep.conf

You will notice that the training is slower as the model is more complex (6
hidden layers). But the accuracy increases faster per iteration than the shallow
model.
