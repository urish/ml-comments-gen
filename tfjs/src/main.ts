import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import { ITokenizersJson, CommentPredictor } from './comment-predictor';

const modelDir = path.resolve('../runs/870/tfjsmodel');
const modelFile = path.join(modelDir, 'model.json');
const tokenizers: ITokenizersJson = require(path.join(modelDir, 'tokenizers.json'));
const testFn = 'function printSum(a:number,b:number) { console.log(a + b); }';

async function main() {
  const model = await tf.loadLayersModel('file://' + modelFile);
  const predictor = new CommentPredictor(model, tokenizers);

  console.log('*** INPUT AST ***');
  console.log(predictor.ast(testFn));
  console.log('');
  console.log('*** OUTPUT ***');
  for (const text of predictor.predict(testFn)) {
    process.stdout.write(text);
  }
}

main().catch(console.error);
