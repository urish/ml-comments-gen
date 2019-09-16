import * as tf from '@tensorflow/tfjs-node';
import { dumpAst } from '../../prepare/src/dump-ast';

export interface ITokenizersJson {
  params: {
    ast_vocab_size: number;
    comment_vocab_size: number;
    max_ast_len: number;
    max_comment_len: number;
    lstm_layer_size: number;
  };
  ast: { [key: string]: number };
  comments: { [key: string]: string };
}

export class CommentPredictor {
  private encoderModel: tf.LayersModel;
  private decoderModel: tf.LayersModel;

  constructor(model: tf.LayersModel, private tokenizers: ITokenizersJson) {
    const { lstm_layer_size } = tokenizers.params;
    const encoderInputs = (model.input as tf.SymbolicTensor[])[0];
    const [, stateHEnc, stateCEnc] = model.layers[2].output as tf.SymbolicTensor[]; // lstm_1
    const encoderStates = [stateHEnc, stateCEnc];
    this.encoderModel = tf.model({ inputs: encoderInputs, outputs: encoderStates });

    const decoder_inputs = (model.input as tf.SymbolicTensor[])[1]; // input_2
    const decoder_state_input_h = tf.input({ shape: [lstm_layer_size], name: 'input_3' });
    const decoder_state_input_c = tf.input({ shape: [lstm_layer_size], name: 'input_4' });
    const decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c];
    const decoder_lstm = model.layers[3];
    const [decoder_lstm_outputs, state_h_dec, state_c_dec] = decoder_lstm.apply(decoder_inputs, {
      initial_state: decoder_states_inputs
    }) as tf.SymbolicTensor[];
    const decoder_states = [state_h_dec, state_c_dec];
    const decoder_dense = model.layers[4];
    const decoder_outputs = decoder_dense.apply(decoder_lstm_outputs) as tf.SymbolicTensor;
    this.decoderModel = tf.model({
      inputs: [decoder_inputs, ...decoder_states_inputs],
      outputs: [decoder_outputs, ...decoder_states]
    });
  }

  ast(functionDecl: string) {
    return dumpAst(functionDecl, true);
  }

  *predict(functionDecl: string) {
    const { ast: astTokens, comments: commentTokens } = this.tokenizers;
    const startToken = parseInt(
      Object.keys(commentTokens).find((k) => commentTokens[k] === '<start>')!,
      10
    );
    const ast = this.ast(functionDecl);
    const astVector = ast
      .split(' ')
      .map((token) => (token in astTokens ? astTokens[token] : astTokens['UNK']));

    const { max_comment_len, ast_vocab_size, comment_vocab_size } = this.tokenizers.params;
    const inputSeq = tf.oneHot(astVector, ast_vocab_size).expandDims();
    let statesValues = this.encoderModel.predict(inputSeq) as tf.Tensor[];

    // Populate the first character of target sequence with the start character.
    let targetSeq = tf.oneHot([startToken], comment_vocab_size).expandDims();
    for (let i = 0; i < max_comment_len; i++) {
      const [outputTokens, h, c] = this.decoderModel.predict([
        targetSeq,
        ...statesValues
      ]) as tf.Tensor[];

      // Sample a token
      const sampled_token_index = (outputTokens.argMax(2).arraySync() as number[][])[0][0];
      const next_word = commentTokens[sampled_token_index as number];

      if (next_word == '<end>') {
        return;
      }

      if (next_word == '<eol>' || next_word === '<eos>') {
        yield '\n';
      } else {
        yield next_word;
      }

      targetSeq = tf.oneHot([sampled_token_index], comment_vocab_size).expandDims();
      statesValues = [h, c];
    }
  }
}
