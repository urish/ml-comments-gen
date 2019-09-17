import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import { ITokenizersJson } from './comment-predictor';

export async function loadModel(modelPath: string) {
  const modelDir = path.resolve(modelPath);
  const modelFile = path.join(modelDir, 'model.json');
  const model = await tf.loadLayersModel('file://' + modelFile);
  const tokenizers = require(path.join(modelDir, 'tokenizers.json')) as ITokenizersJson;
  return {
    model,
    tokenizers
  };
}
