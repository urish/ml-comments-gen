import { dumpAst } from './dump-ast';

describe('dumpAst', () => {
  it('should dump AST for a simple expression', () => {
    expect(dumpAst('a = 5')).toEqual(
      '( ExpressionStatement ( BinaryExpression ( Identifier EqualsToken NumericLiteral ) ) ) EndOfFileToken'
    );
  });

  it('should include identifier names when instructed to', () => {
    expect(dumpAst('a = 5', { includeIdentifiers: true })).toEqual(
      '( ExpressionStatement ( BinaryExpression ( Identifier [a] EqualsToken NumericLiteral ) ) ) EndOfFileToken'
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
        'EndOfFileToken'
    );
  });

  it('should correctly dump the AST of functions', () => {
    expect(
      dumpAst('function myFunc(foo) { return foo + "bar"; }', { functionOrMethod: true })
    ).toEqual(
      'FunctionDeclaration ( FunctionKeyword Identifier OpenParenToken SyntaxList ( ' +
        'Parameter ( Identifier ) ) CloseParenToken Block ( OpenBraceToken SyntaxList (' +
        ' ReturnStatement ( ReturnKeyword BinaryExpression ( Identifier PlusToken StringLiteral' +
        ' ) SemicolonToken ) ) CloseBraceToken )'
    );
  });

  it('should correctly dump the AST of class methods', () => {
    expect(dumpAst('myMethod(foo) { return foo + "bar"; }', { functionOrMethod: true })).toEqual(
      'MethodDeclaration ( Identifier OpenParenToken SyntaxList ( Parameter ( Identifier ) ) ' +
        'CloseParenToken Block ( OpenBraceToken SyntaxList ( ReturnStatement ( ReturnKeyword' +
        ' BinaryExpression ( Identifier PlusToken StringLiteral ) SemicolonToken ) ) CloseBraceToken )'
    );
  });
});
