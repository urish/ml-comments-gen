import * as tf from '@tensorflow/tfjs-node';
import { dumpAst } from '../../prepare/src/dump-ast';

export interface ITokenizersJson {
  params: { max_seq_len: number };
  ast: { [key: string]: number };
  comments: { [key: string]: string };
}

function zeroPad(v: number[], minSize: number) {
  const result = [...v];
  while (result.length < minSize) {
    result.push(0);
  }
  return result;
}

export class CommentPredictor {
  constructor(private model: tf.LayersModel, private tokenizers: ITokenizersJson) {}

  ast(functionDecl: string) {
    return dumpAst(functionDecl, true);
  }

  *predict(functionDecl: string) {
    const { ast: astTokens, comments: commentTokens, params } = this.tokenizers;
    const startToken = parseInt(
      Object.keys(commentTokens).find((k) => commentTokens[k] === '<start>')!,
      10
    );
    const ast = this.ast(functionDecl);
    const astVector = ast
      .split(' ')
      .map((token) => (token in astTokens ? astTokens[token] : astTokens['UNK']));
    const x1Pad = [zeroPad(astVector, params.max_seq_len)];
    const x2 = [startToken];
    for (let i = 0; i < 200; i++) {
      const x2Pad = [zeroPad(x2, params.max_seq_len)];
      const yHat = this.model.predict([tf.tensor(x1Pad), tf.tensor(x2Pad)]) as tf.Tensor2D;
      const nextCommentToken = (yHat.argMax(1).arraySync() as number[])[0];
      x2.push(nextCommentToken);
      const tokenValue = commentTokens[nextCommentToken];
      yield tokenValue === '<eos>' ? '\n' : tokenValue + ' ';
      if (tokenValue === '*/' || tokenValue === '<end>') {
        return;
      }
    }
  }
}
