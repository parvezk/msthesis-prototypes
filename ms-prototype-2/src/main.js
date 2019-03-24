/**
 * Based on
 * https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
 */

 import * as tf from "@tensorflow/tfjs";
 import {writeInternalActivationAndGetOutput} from "./filters"

 /**
 * Calcuate and save the maximally-activating input images for a covn2d layer.
 */

 const CONFIG = {
     layerNames: "block1_conv1,block2_conv1,block3_conv2,block4_conv2,block5_conv3",
     filters: 8,
     inputImage: "",
     outputDir: "./dist/activation"
 }

export async function processing(model, inputTensor, activationsDiv) {

    const imageHeight = model.inputs[0].shape[1];
    const imageWidth = model.inputs[0].shape[2];
    const { filters, outputDir } = CONFIG;

    let x = inputTensor.resizeNearestNeighbor([imageHeight, imageWidth]);
    x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]]);
    //x = x.as4D([1, x.shape.length[0], x.shape.length[1], x.shape.length[2]]);
    const layerNames = CONFIG.layerNames.split(',');

    await writeInternalActivationAndGetOutput(
        model, layerNames, x, filters, outputDir, activationsDiv);
 }