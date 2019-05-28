import { dumpAst } from './dump-ast';

describe('dumpAst', () => {
  it('should dump AST for a simple expression', () => {
    expect(dumpAst('a = 5')).toEqual([
      'ExpressionStatement BinaryExpression Identifier',
      'ExpressionStatement BinaryExpression FirstAssignment',
      'ExpressionStatement BinaryExpression FirstLiteralToken',
      'EndOfFileToken',
    ]);
  });

  it('should dump AST for block expression', () => {
    expect(dumpAst(`for (x = 0; x < 10; x++) { console.log(x) }`)).toEqual([
      'ForStatement ForKeyword',
      'ForStatement OpenParenToken',
      'ForStatement BinaryExpression Identifier',
      'ForStatement BinaryExpression FirstAssignment',
      'ForStatement BinaryExpression FirstLiteralToken',
      'ForStatement SemicolonToken',
      'ForStatement BinaryExpression Identifier',
      'ForStatement BinaryExpression FirstBinaryOperator',
      'ForStatement BinaryExpression FirstLiteralToken',
      'ForStatement SemicolonToken',
      'ForStatement PostfixUnaryExpression Identifier',
      'ForStatement PostfixUnaryExpression PlusPlusToken',
      'ForStatement CloseParenToken',
      'ForStatement Block FirstPunctuation',
      'ForStatement Block ExpressionStatement CallExpression PropertyAccessExpression Identifier',
      'ForStatement Block ExpressionStatement CallExpression PropertyAccessExpression DotToken',
      'ForStatement Block ExpressionStatement CallExpression PropertyAccessExpression Identifier',
      'ForStatement Block ExpressionStatement CallExpression OpenParenToken',
      'ForStatement Block ExpressionStatement CallExpression Identifier',
      'ForStatement Block ExpressionStatement CallExpression CloseParenToken',
      'ForStatement Block CloseBraceToken',
      'EndOfFileToken',
    ]);
  });
});
