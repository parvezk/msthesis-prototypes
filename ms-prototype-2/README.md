 # Thesis Prototype: 

## Visualize What Deep Network Learns
> A visual exploration tool for a visual classifier that explains reasons for a classification decision to an end user.
<!-- visual classifier that explains the reasons for a classification decision to an end user -->

The tool jointly predicts a class label, and shows why the predicted label is appropriate for a given image using visual evidence. 
# [//] another comment option
- Visualizing intermediate outputs: Interpret how successive layers transform the input image.
- Visualizing filters: See what visual pattern or concept each filter in a network is receptive.
- Visualizing heatmaps: Recognize which part of an input image attributed most to the classification decision by localizing the object in the image.


## Building and running on localhost

First install dependencies:

```sh
npm install
```

To create a production build:

```sh
npm run build-prod
```

To create a development build:

```sh
npm run build-dev
```

## Running

```sh
node dist/bundle.js
```

