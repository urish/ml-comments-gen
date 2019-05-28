import { createSourceFile, ScriptTarget, SyntaxKind, Node, isSourceFile } from 'typescript';
import 'array-flat-polyfill';

function traverse(node: Node): string[] {
  const nodeKind = SyntaxKind[node.kind];
  if (!node.getChildCount()) {
    return [nodeKind];
  }
  const prefix = !isSourceFile(node) && node.kind != SyntaxKind.SyntaxList ? nodeKind + ' ' : '';
  return node
    .getChildren()
    .flatMap((child) => traverse(child))
    .map((kindName) => prefix + kindName);
}

export function dumpAst(source: string) {
  const ast = createSourceFile('', source, ScriptTarget.Latest, true);
  return traverse(ast);
}
