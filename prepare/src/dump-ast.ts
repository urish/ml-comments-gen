import 'array-flat-polyfill';
import { createSourceFile, Node, ScriptTarget, SyntaxKind } from 'typescript';

function traverse(node: Node, includeComments: boolean): string {
  const nodeKind = SyntaxKind[node.kind];
  if (!node.getChildren().length) {
    return nodeKind;
  }
  return (
    nodeKind +
    ' ( ' +
    node
      .getChildren()
      .map((child) => traverse(child, includeComments))
      .join(' ') +
    ' )'
  );
}

export function dumpAst(source: string, includeComments = false) {
  const ast = createSourceFile('', source, ScriptTarget.Latest, true);
  return traverse(ast, includeComments).replace(/^SourceFile \( SyntaxList | \)$/g, '');
}
