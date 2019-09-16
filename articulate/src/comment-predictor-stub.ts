const stubComment = '/* this is a stub comment \n * second line */';

export class CommentPredictorStub {
  constructor() {}

  *predict(functionDecl: string) {
    for (const token of stubComment.split(' ')) {
      yield token + ' ';
    }
  }
}
