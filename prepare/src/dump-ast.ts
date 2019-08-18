import 'array-flat-polyfill';
import { tsquery } from '@phenomnomnominal/tsquery';
import { MethodDeclaration, Node } from 'typescript';
import { syntaxKindMap } from './syntax-kind';

function traverse(node: Node): string {
  const nodeKind = syntaxKindMap[node.kind];
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
  let ast: Node = tsquery.ast(source);
  if (functionOrMethod) {
    const functionNode = tsquery.query(ast, 'SourceFile>FunctionDeclaration');
    if (functionNode[0]) {
      ast = functionNode[0];
    } else {
      ast = tsquery.ast(`class __DUMMY { ${source} }`);
      ast = tsquery.query<MethodDeclaration>(ast, 'SourceFile>ClassDeclaration>MethodDeclaration')[0];
    }
  }
  return traverse(ast).replace(/^SourceFile \( SyntaxList | \)$/g, '');
}
