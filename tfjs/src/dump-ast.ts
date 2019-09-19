import { tsquery } from '@phenomnomnominal/tsquery';
import 'array-flat-polyfill';
import { Node, isIdentifier } from 'typescript';
import { functionOrMethodAST } from './ast-utils';

export interface IDumpAstOptions {
  functionOrMethod?: boolean;
  includeIdentifiers?: boolean;
}

function traverse(node: Node, options: IDumpAstOptions): string {
  let nodeKind = tsquery.syntaxKindName(node.kind);
  if (options.includeIdentifiers && isIdentifier(node)) {
    nodeKind += ` [${node.text}]`;
  }
  if (!node.getChildren().length) {
    return nodeKind;
  }
  return (
    nodeKind +
    ' ( ' +
    node
      .getChildren()
      .map((child) => traverse(child, options))
      .join(' ') +
    ' )'
  );
}

export function dumpAst(source: string, options: IDumpAstOptions = {}) {
  const ast = options.functionOrMethod ? functionOrMethodAST(source) : tsquery.ast(source);
  return traverse(ast, options).replace(/^SourceFile \( SyntaxList | \)$/g, '');
}
