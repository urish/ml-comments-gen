import { dumpAst } from './dump-ast';

describe('dumpAst', () => {
  it('should dump AST for a simple expression', () => {
    expect(dumpAst('a = 5')).toEqual(
      '( ExpressionStatement ( BinaryExpression ( Identifier EqualsToken NumericLiteral ) ) ) EndOfFileToken',
    );
  });

  it('should dump AST for block expression', () => {
    expect(dumpAst(`for (x = 0; x < 10; x++) { console.log(x) }`)).toEqual(
      '( ForStatement ( ForKeyword OpenParenToken BinaryExpression ( Identifier EqualsToken NumericLiteral ) ' +
        'SemicolonToken BinaryExpression ( Identifier LessThanToken NumericLiteral ) ' +
        'SemicolonToken PostfixUnaryExpression ( Identifier PlusPlusToken ) CloseParenToken ' +
        'Block ( OpenBraceToken SyntaxList ( ExpressionStatement ( ' +
        'CallExpression ( PropertyAccessExpression ( Identifier DotToken Identifier ) OpenParenToken ' +
        'SyntaxList ( Identifier ) CloseParenToken ) ) ) CloseBraceToken ) ) ) ' +
        'EndOfFileToken',
    );
  });
});
