
/**
 * Algorithms for analyzing and visualizing the convolutional filters
 * internal to a convnet.
 * 
 * 1. Retrieving internal activations of a convnet.
 * See function `writeInternalActivationAndGetOutput()`.
 **/

import * as path from "path";
import * as utils from "./utils";
import * as tf from "@tensorflow/tfjs";

async function writeInternalActivationAndGetOutput(
    model, layerNames, inputImage, numFilters, outputDir) {
}

module.exports = {
    writeInternalActivationAndGetOutput
  };
  