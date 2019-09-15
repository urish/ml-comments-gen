import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import { dumpAst } from '../../prepare/src/dump-ast';

interface ITokenizersJson {
  params: { max_seq_len: number };
  ast: { [key: string]: number };
  comments: { [key: string]: string };
}

const modelDir = path.resolve('../runs/870/tfjsmodel');
const modelFile = path.join(modelDir, 'model.json');
const { params, ast: astTokens, comments: commentTokens }: ITokenizersJson = require(path.join(
  modelDir,
  'tokenizers.json'
));
const startToken = parseInt(
  Object.keys(commentTokens).find((k) => commentTokens[k] === '<start>')!,
  10
);

function zeroPad(v: number[], minSize: number) {
  const result = [...v];
  while (result.length < minSize) {
    result.push(0);
  }
  return result;
}

async function main() {
  const model = await tf.loadLayersModel('file://' + modelFile);
  const ast = dumpAst(
    'export function SocketFactory(config: SocketIoConfig) {\n    return new WrappedSocket(config);\n}',
    true
  );
  console.log('*** INPUT AST ***');
  console.log(ast);
  console.log('');
  console.log('*** OUTPUT ***');
  const astVector = ast
    .split(' ')
    .map((token) => (token in astTokens ? astTokens[token] : astTokens['UNK']));
  const x1Pad = [zeroPad(astVector, params.max_seq_len)];
  const x2 = [startToken];
  for (let i = 0; i < 100; i++) {
    const x2Pad = [zeroPad(x2, params.max_seq_len)];
    const yHat = model.predict([tf.tensor(x1Pad), tf.tensor(x2Pad)]) as tf.Tensor2D;
    const nextCommentToken = (yHat.argMax(1).arraySync() as number[])[0];
    x2.push(nextCommentToken);
    const tokenValue = commentTokens[nextCommentToken];
    console.log(tokenValue);
    if (tokenValue === '<eos>') {
      break;
    }
  }
}

main().catch(console.log);
