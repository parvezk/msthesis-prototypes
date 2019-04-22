
import * as tf from '@tensorflow/tfjs';

import {IMAGE_H, IMAGE_W} from './data';

export function createConvModel() {

    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));

    // Our third layer is another convolution, this time with 32 filters.
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // Max pooling again.
    model.add(tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));

    model.add(tf.layers.flatten({}));

    model.add(tf.layers.dense({
        units: 10,
        KernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    return model;
}