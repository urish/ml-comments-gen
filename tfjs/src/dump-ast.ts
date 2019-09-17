import { tsquery } from '@phenomnomnominal/tsquery';
import 'array-flat-polyfill';
import { Node } from 'typescript';
import { functionOrMethodAST } from './ast-utils';

function traverse(node: Node): string {
  const nodeKind = tsquery.syntaxKindName(node.kind);
  if (!node.getChildren().length) {
    return nodeKind;
  }
  return (
    nodeKind +
    ' ( ' +
    node
      .getChildren()
      .map((child) => traverse(child))
      .join(' ') +
    ' )'
  );
}

export function dumpAst(source: string, functionOrMethod = false) {
  const ast = functionOrMethod ? functionOrMethodAST(source) : tsquery.ast(source);
  return traverse(ast).replace(/^SourceFile \( SyntaxList | \)$/g, '');
}
