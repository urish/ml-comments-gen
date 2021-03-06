import * as tf from '@tensorflow/tfjs-node';
import { dumpAst } from './dump-ast';
import { getSubstitutionsDict } from './rename-args-in-comments';
import { loadModel } from './load-model';

export interface ITokenizersJson {
  params: {
    ast_vocab_size: number;
    comment_vocab_size: number;
    max_ast_len: number;
    max_comment_len: number;
    lstm_layer_size: number;
    character_tokenizer: number;
  };
  ast: { [key: string]: number };
  comments: { [key: string]: string };
}

function* commentTokenizer(comment: Iterable<string>) {
  let buffer = '';
  for (let item of comment) {
    buffer += item;
    while (buffer.length > 0) {
      const matches = buffer.match(/\W+|\w+/g) || [];
      if (matches.length < 2) {
        break;
      }
      yield matches[0];
      buffer = buffer.slice(matches[0].length);
    }
  }
  const matches = buffer.match(/\W+|\w+/) || [];
  for (let match of matches) {
    yield match;
  }
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
      initialState: decoder_states_inputs
    }) as tf.SymbolicTensor[];
    const decoder_states = [state_h_dec, state_c_dec];
    const decoder_dense = model.layers[4];
    const decoder_outputs = decoder_dense.apply(decoder_lstm_outputs) as tf.SymbolicTensor;
    this.decoderModel = tf.model({
      inputs: [decoder_inputs, ...decoder_states_inputs],
      outputs: [decoder_outputs, ...decoder_states]
    });
  }

  static async loadFrom(modelPath: string) {
    const { model, tokenizers } = await loadModel(modelPath);
    return new CommentPredictor(model, tokenizers);
  }

  ast(functionDecl: string) {
    return dumpAst(functionDecl, { functionOrMethod: true });
  }

  private *predictInternal(functionDecl: string) {
    const { ast: astTokens, comments: commentTokens } = this.tokenizers;
    const {
      max_ast_len,
      max_comment_len,
      ast_vocab_size,
      comment_vocab_size,
      character_tokenizer
    } = this.tokenizers.params;

    const ast = this.ast(functionDecl);
    const astVector = ast
      .split(' ')
      .slice(0, max_ast_len)
      .map((token) => (token in astTokens ? astTokens[token] : astTokens['UNK']));
    const inputSeq = tf
      .oneHot(astVector, ast_vocab_size)
      .pad([[0, max_ast_len - astVector.length], [0, 0]])
      .expandDims();
    let statesValues = this.encoderModel.predict(inputSeq) as tf.Tensor[];

    const startToken = character_tokenizer ? '/' : '<start>';
    const startTokenValue = parseInt(
      Object.keys(commentTokens).find((k) => commentTokens[k] === startToken)!,
      10
    );
    if (character_tokenizer) {
      yield startToken;
    }

    // Populate the first character of target sequence with the start character.
    let targetSeq = tf.oneHot([startTokenValue], comment_vocab_size).expandDims();
    let firstInLine = true;
    let prevWord = '';

    for (let i = 0; i < max_comment_len; i++) {
      const [outputTokens, h, c] = this.decoderModel.predict([
        targetSeq,
        ...statesValues
      ]) as tf.Tensor[];

      // Sample a token
      const sampled_token_index = (outputTokens.argMax(2).arraySync() as number[][])[0][0];
      const nextWord = commentTokens[sampled_token_index as number];

      if (nextWord == '<end>' || !nextWord) {
        return;
      }

      if (character_tokenizer) {
        yield nextWord;
      } else if (nextWord == '<eol>' || nextWord === '<eos>') {
        yield '\n';
        firstInLine = true;
      } else if (nextWord[0] === '>') {
        const space = firstInLine || prevWord[0] === '>' ? '' : ' ';
        yield space + nextWord[1];
        firstInLine = false;
      } else {
        const space = firstInLine ? '' : ' ';
        yield space + nextWord;
        firstInLine = false;
      }
      prevWord = nextWord;

      targetSeq = tf.oneHot([sampled_token_index], comment_vocab_size).expandDims();
      statesValues = [h, c];
    }
  }

  *predictTokens(functionDecl: string) {
    const substitutions = getSubstitutionsDict(functionDecl, true);
    for (let key of Object.keys(substitutions)) {
      substitutions[key.toLowerCase()] = substitutions[key];
    }
    for (let token of commentTokenizer(this.predictInternal(functionDecl))) {
      if (token in substitutions) {
        yield substitutions[token];
      } else {
        yield token;
      }
    }
  }

  predict(functionDecl: string) {
    return Array.from(this.predictTokens(functionDecl)).join('');
  }
}
