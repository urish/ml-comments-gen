import 'array-flat-polyfill';
import {
  createSourceFile,
  getLeadingCommentRanges,
  getTrailingCommentRanges,
  isSourceFile,
  Node,
  ScriptTarget,
  SyntaxKind,
} from 'typescript';

function traverse(node: Node, includeComments: boolean): string[] {
  const nodeKind = SyntaxKind[node.kind];
  const source = node.getSourceFile().getFullText();
  const leadingComments = getLeadingCommentRanges(source, node.pos) || [];
  const trailingComments = getTrailingCommentRanges(source, node.pos) || [];
  const allComments = [...leadingComments, ...trailingComments].map((range) =>
    source.substr(range.pos, range.end - range.pos),
  );
  if (node.kind === SyntaxKind.JSDocComment) {
    return [];
  }
  if (!node.getChildCount()) {
    return [...allComments, nodeKind];
  }
  const prefix = !isSourceFile(node) && node.kind != SyntaxKind.SyntaxList ? nodeKind + ' ' : '';
  return node
    .getChildren()
    .flatMap((child) => traverse(child, includeComments))
    .map((kindName) => (kindName[0] === '/' ? kindName : prefix + kindName));
}

function removeDuplicates(items: string[]) {
  return items.reduce((acc, item) => (acc[acc.length - 1] !== item ? [...acc, item] : acc), []);
}

export function dumpAst(source: string, includeComments = false) {
  const ast = createSourceFile('', source, ScriptTarget.Latest, true);
  return removeDuplicates(traverse(ast, includeComments));
}
