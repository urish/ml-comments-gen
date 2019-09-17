import { functionOrMethodAST } from './ast-utils';

const functionNamePlaceholder = 'FunctionNamePlaceholder';
const argumentPlaceholderPrefix = 'ArgumentNumber';

function createDict(argNames: string[], functionName: string | undefined, reverse: boolean) {
  const dict: { [key: string]: string } = {};
  for (let i = 0; i < argNames.length; i++) {
    const placeholder = argumentPlaceholderPrefix + i;
    if (reverse) {
      dict[placeholder] = argNames[i];
    } else {
      dict[argNames[i]] = placeholder;
    }
  }
  if (reverse) {
    dict[functionNamePlaceholder] = functionName || '';
  } else {
    if (functionName) {
      dict[functionName] = functionNamePlaceholder;
    }
  }
  return dict;
}

export function renameArgsInComments(comments: string, source: string, reverse = false) {
  const targetNode = functionOrMethodAST(source);
  const argNames = targetNode.parameters.map((p) => p.name.getText());
  const functionName = targetNode.name && targetNode.name.getText();
  const tokens = comments.match(/\W+|\w+/g) || [];
  const dict = createDict(argNames, functionName, reverse);
  return tokens.map((token) => (token in dict ? dict[token] : token)).join('');
}
