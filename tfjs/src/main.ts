import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';

const myDir = path.resolve('../runs/870/tfjsmodel/model.json');

async function main() {
  const model = await tf.loadLayersModel('file://' + myDir);
  console.log('Model loaded successfully!', !!model);
}

main().catch(console.log);
