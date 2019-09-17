import { CommentPredictor } from './comment-predictor';
import { loadModel } from './load-model';

const modelDir = '../runs/js-500/tfjsmodel';
const testFn = 'function printSum(a:number,b:number) { console.log(a + b); }';

async function main() {
  const {model, tokenizers} = await loadModel(modelDir);
  const predictor = new CommentPredictor(model, tokenizers);

  console.log('*** INPUT AST ***');
  console.log(predictor.ast(testFn));
  console.log('');
  console.log('*** OUTPUT ***');

  let predictedComment = [];
  for (const token of predictor.predict(testFn)) {
    predictedComment.push(token);
  }

  return predictedComment.join(' ');
}

main()
  .then(console.log)
  .catch(console.error);
