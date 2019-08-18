import {
  forEachChild,
  FunctionDeclaration,
  isFunctionDeclaration,
  isMethodDeclaration,
  MethodDeclaration,
  Node,
} from 'typescript';

export const supportedLanguageIds = ['javascript', 'javascriptreact', 'typescript', 'typescriptreact'];

export function isSupportedLanguage(langId: string) {
  return supportedLanguageIds.indexOf(langId) >= 0;
}

export function getNodeAtFileOffset(node: Node, offset: number) {
  let result = null as Node | null;
  const visit = (childNode: Node) => {
    forEachChild(childNode, visit);
    if (!result && (childNode.getStart() <= offset && childNode.getEnd() > offset)) {
      result = childNode;
    }
  };
  visit(node);
  return result;
}

export function findParentFunction(node: Node | null): FunctionDeclaration | MethodDeclaration | null {
  if (!node) {
    return node;
  }
  if (isFunctionDeclaration(node) || isMethodDeclaration(node)) {
    return node;
  }
  return findParentFunction(node.parent);
}
