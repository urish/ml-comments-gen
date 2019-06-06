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

  it('should include comment nodes in the AST', () => {
    expect(dumpAst('a = 5 /* magic */;', true)).toEqual([
      'ExpressionStatement BinaryExpression Identifier',
      'ExpressionStatement BinaryExpression FirstAssignment',
      'ExpressionStatement BinaryExpression FirstLiteralToken',
      '/* magic */',
      'ExpressionStatement SemicolonToken',
      'EndOfFileToken',
    ]);
  });

  it('should support multiple comments', () => {
    expect(dumpAst('/*start*/a// two\n = /*five*/5 /* magic */ /* more magic */;//EOF', true)).toEqual([
      '/*start*/',
      'ExpressionStatement BinaryExpression Identifier',
      '// two',
      'ExpressionStatement BinaryExpression FirstAssignment',
      '/*five*/',
      'ExpressionStatement BinaryExpression FirstLiteralToken',
      '/* magic */',
      '/* more magic */',
      'ExpressionStatement SemicolonToken',
      '//EOF',
      'EndOfFileToken',
    ]);
  });

  it('should support functions with docstrings', () => {
    expect(dumpAst('/**\n * test\n */ function f() { return 5; }', true)).toEqual([
      '/**\n * test\n */',
      'FunctionDeclaration FunctionKeyword',
      'FunctionDeclaration Identifier',
      'FunctionDeclaration OpenParenToken',
      'FunctionDeclaration SyntaxList',
      'FunctionDeclaration CloseParenToken',
      'FunctionDeclaration Block FirstPunctuation',
      'FunctionDeclaration Block ReturnStatement ReturnKeyword',
      'FunctionDeclaration Block ReturnStatement FirstLiteralToken',
      'FunctionDeclaration Block ReturnStatement SemicolonToken',
      'FunctionDeclaration Block CloseBraceToken',
      'EndOfFileToken',
    ]);
  });
});
