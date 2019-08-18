import { functionOrMethodAST } from './ast-utils';

export function renameArgsInComments(comments: string, source: string) {
  const targetNode = functionOrMethodAST(source);
  const argNames = targetNode.parameters.map((p) => p.name.getText());
  const tokens = comments.match(/\W+|\w+/g) || [];
  return tokens
    .map((token) => {
      const index = argNames.indexOf(token);
      return index >= 0 ? `ArgumentNumber${index}` : token;
    })
    .join('');
}
